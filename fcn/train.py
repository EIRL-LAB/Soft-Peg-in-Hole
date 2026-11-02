import argparse
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ANGLE_DEGREES, ANGLE_LABEL_POLICY, RectangleDataset
from losses import HeatmapFocalLoss, heatmap_peak_mask
from models import FCN
from rotation_equivariant import (
    RotationEquivariantFCN,
    pad_to_square,
    rotate_tensor,
)


TRAINING_OBJECTIVE = "heatmap_focal_all_equivalent_angles_v4"
MODEL_ARCHITECTURE = "resnet18_fcn_groupnorm_heatmap_only_v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the single-channel rotation-equivariant hole FCN."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--checkpoint",
        default="best_heatmap_all_angles_v4.pth",
        help="Best validation checkpoint used for final testing and inference.",
    )
    parser.add_argument(
        "--last-checkpoint",
        default="last_heatmap_all_angles_v4.pth",
        help="Latest completed epoch checkpoint used by --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer and scheduler state from --last-checkpoint.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional old checkpoint used for initialization.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=2,
        help=(
            "Number of angle branches processed together. All 36 branches "
            "are still evaluated in every training and inference scan."
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Lightly compile scan path with "
            "aot_eager (use --no-compile to disable)."
        ),
    )
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--canvas-size", type=int, default=400)
    parser.add_argument(
        "--background-loss-weight",
        type=float,
        default=1.0,
        help="Weight for background pixels inside the positive heatmap branch.",
    )
    parser.add_argument(
        "--valid-angle-weight",
        type=float,
        default=2.0,
        help="Loss multiplier for symmetry-equivalent valid angle branches.",
    )
    parser.add_argument(
        "--invalid-angle-peak",
        type=float,
        default=0.1,
        help="Target center-heatmap peak for an invalid angle branch.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Maximum training batches per epoch; 0 uses the full loader.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="Limit full-angle validation samples; 0 evaluates the full split.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit final full-angle test samples; 0 evaluates the full split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_initial_weights(
    base_fcn: FCN,
    rotation_model: RotationEquivariantFCN,
    path: str,
    device: torch.device,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if any(key.startswith("base_fcn.") for key in state_dict):
        incompatible = rotation_model.load_state_dict(state_dict, strict=False)
    elif any(key.startswith("fcn.") for key in state_dict):
        migrated = {
            (
                "base_fcn." + key.removeprefix("fcn.")
                if key.startswith("fcn.")
                else key
            ): value
            for key, value in state_dict.items()
        }
        incompatible = rotation_model.load_state_dict(migrated, strict=False)
    else:
        incompatible = base_fcn.load_state_dict(state_dict, strict=False)

    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("base_fcn.base_fcn.")
    ]
    if unexpected or disallowed_missing:
        details = []
        if disallowed_missing:
            details.append(f"missing keys: {disallowed_missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise RuntimeError(
            "initial checkpoint is incompatible ("
            + "; ".join(details) + ")"
        )


@dataclass
class TrainingMetricsAccumulator:
    """Accumulate loss components and diagnostic probabilities."""

    sample_count: int = 0
    step_count: int = 0
    loss_sums: dict[str, float] = field(
        default_factory=lambda: {
            "loss": 0.0,
            "positive_loss": 0.0,
            "background_loss": 0.0,
        }
    )
    positive_probability_sum: float = 0.0
    positive_probability_count: int = 0
    positive_probability_min: float = math.inf
    valid_angle_probability_sum: float = 0.0
    valid_angle_probability_count: int = 0
    valid_angle_probability_min: float = math.inf
    valid_angle_view_count: int = 0
    invalid_angle_probability_sum: float = 0.0
    invalid_angle_probability_count: int = 0
    invalid_angle_probability_max: float = -math.inf
    invalid_angle_view_count: int = 0
    background_probability_sum: float = 0.0
    background_probability_count: int = 0
    background_probability_max: float = -math.inf
    logit_min: float = math.inf
    logit_max: float = -math.inf
    gradient_norm_sum: float = 0.0

    def add_batch_losses(
        self,
        values: dict[str, float],
        batch_size: int,
    ) -> None:
        self.sample_count += int(batch_size)
        for name in self.loss_sums:
            self.loss_sums[name] += float(values[name]) * int(batch_size)

    def add_gradient_norm(self, value: float) -> None:
        self.step_count += 1
        self.gradient_norm_sum += float(value)

    def merge(self, other: "TrainingMetricsAccumulator") -> None:
        self.sample_count += other.sample_count
        self.step_count += other.step_count
        for name in self.loss_sums:
            self.loss_sums[name] += other.loss_sums[name]
        self.positive_probability_sum += other.positive_probability_sum
        self.positive_probability_count += other.positive_probability_count
        self.positive_probability_min = min(
            self.positive_probability_min,
            other.positive_probability_min,
        )
        self.valid_angle_probability_sum += other.valid_angle_probability_sum
        self.valid_angle_probability_count += other.valid_angle_probability_count
        self.valid_angle_probability_min = min(
            self.valid_angle_probability_min,
            other.valid_angle_probability_min,
        )
        self.valid_angle_view_count += other.valid_angle_view_count
        self.invalid_angle_probability_sum += other.invalid_angle_probability_sum
        self.invalid_angle_probability_count += other.invalid_angle_probability_count
        self.invalid_angle_probability_max = max(
            self.invalid_angle_probability_max,
            other.invalid_angle_probability_max,
        )
        self.invalid_angle_view_count += other.invalid_angle_view_count
        self.background_probability_sum += other.background_probability_sum
        self.background_probability_count += other.background_probability_count
        self.background_probability_max = max(
            self.background_probability_max,
            other.background_probability_max,
        )
        self.logit_min = min(self.logit_min, other.logit_min)
        self.logit_max = max(self.logit_max, other.logit_max)
        self.gradient_norm_sum += other.gradient_norm_sum

    def add_tensor_statistics(
        self,
        logits: torch.Tensor,
        center_mask: torch.Tensor,
        is_valid_angle: torch.Tensor,
    ) -> None:
        """Collect center probabilities for valid and invalid angle views."""

        if logits.ndim != 4:
            raise ValueError(
                f"expected logits [B,1,H,W], got {tuple(logits.shape)}"
            )
        if center_mask.shape != (
            logits.shape[0],
            1,
            logits.shape[-2],
            logits.shape[-1],
        ):
            raise ValueError(
                "center_mask must have shape [B,1,H,W], got "
                f"{tuple(center_mask.shape)} for logits {tuple(logits.shape)}"
            )
        if is_valid_angle.shape != (logits.shape[0],):
            raise ValueError(
                "is_valid_angle must have shape [N], got "
                f"{tuple(is_valid_angle.shape)}"
            )
        detached_logits = logits.detach()
        probabilities = torch.sigmoid(detached_logits)
        positive_centers = center_mask
        background = ~center_mask

        positive_values = probabilities.masked_select(positive_centers)
        if positive_values.numel():
            self.positive_probability_sum += float(positive_values.sum())
            self.positive_probability_count += int(positive_values.numel())
            self.positive_probability_min = min(
                self.positive_probability_min,
                float(positive_values.min()),
            )

        valid_selector = is_valid_angle.to(
            device=center_mask.device,
            dtype=torch.bool,
        )[:, None, None, None]
        self.valid_angle_view_count += int(is_valid_angle.sum())
        self.invalid_angle_view_count += int((~is_valid_angle).sum())
        valid_values = probabilities.masked_select(
            positive_centers & valid_selector
        )
        if valid_values.numel():
            self.valid_angle_probability_sum += float(valid_values.sum())
            self.valid_angle_probability_count += int(valid_values.numel())
            self.valid_angle_probability_min = min(
                self.valid_angle_probability_min,
                float(valid_values.min()),
            )

        invalid_values = probabilities.masked_select(
            positive_centers & ~valid_selector
        )
        if invalid_values.numel():
            self.invalid_angle_probability_sum += float(invalid_values.sum())
            self.invalid_angle_probability_count += int(invalid_values.numel())
            self.invalid_angle_probability_max = max(
                self.invalid_angle_probability_max,
                float(invalid_values.max()),
            )

        self.background_probability_sum += float(
            probabilities.masked_select(background).sum()
        )
        self.background_probability_count += int(background.sum())
        self.background_probability_max = max(
            self.background_probability_max,
            float(probabilities.masked_fill(~background, -math.inf).max()),
        )
        self.logit_min = min(self.logit_min, float(detached_logits.min()))
        self.logit_max = max(self.logit_max, float(detached_logits.max()))

    def summary(self) -> dict[str, float | None]:
        sample_denominator = max(self.sample_count, 1)

        def mean_or_none(total: float, count: int) -> float | None:
            return total / count if count else None

        return {
            **{
                name: total / sample_denominator
                for name, total in self.loss_sums.items()
            },
            "positive_center_prob_mean": mean_or_none(
                self.positive_probability_sum,
                self.positive_probability_count,
            ),
            "positive_center_prob_min": (
                self.positive_probability_min
                if self.positive_probability_count
                else None
            ),
            "valid_angle_center_prob_mean": mean_or_none(
                self.valid_angle_probability_sum,
                self.valid_angle_probability_count,
            ),
            "valid_angle_center_prob_min": (
                self.valid_angle_probability_min
                if self.valid_angle_probability_count
                else None
            ),
            "valid_angle_views": float(self.valid_angle_view_count),
            "invalid_angle_center_prob_mean": mean_or_none(
                self.invalid_angle_probability_sum,
                self.invalid_angle_probability_count,
            ),
            "invalid_angle_center_prob_max": (
                self.invalid_angle_probability_max
                if self.invalid_angle_probability_count
                else None
            ),
            "invalid_angle_views": float(self.invalid_angle_view_count),
            "background_prob_mean": mean_or_none(
                self.background_probability_sum,
                self.background_probability_count,
            ),
            "background_prob_max": (
                self.background_probability_max
                if self.background_probability_count
                else None
            ),
            "logit_min": self.logit_min if self.sample_count else None,
            "logit_max": self.logit_max if self.sample_count else None,
            "gradient_norm": (
                self.gradient_norm_sum / self.step_count
                if self.step_count
                else None
            ),
            "samples": float(self.sample_count),
            "steps": float(self.step_count),
        }


def _format_training_window(
    batch_index: int,
    batch_count: int,
    metrics: dict[str, float | None],
    learning_rate: float,
) -> str:
    def value(name: str) -> str:
        number = metrics[name]
        return (
            "n/a"
            if number is None or not math.isfinite(number)
            else f"{number:.6g}"
        )

    return (
        f"  train batch {batch_index}/{batch_count} "
        f"loss={value('loss')} pos={value('positive_loss')} "
        f"bg={value('background_loss')} "
        f"valid_prob={value('valid_angle_center_prob_mean')}/"
        f"{value('valid_angle_center_prob_min')} "
        f"invalid_prob={value('invalid_angle_center_prob_mean')}/"
        f"{value('invalid_angle_center_prob_max')} "
        f"views={value('valid_angle_views')}/"
        f"{value('invalid_angle_views')} "
        f"bg_prob={value('background_prob_mean')}/"
        f"{value('background_prob_max')} "
        f"logit={value('logit_min')}..{value('logit_max')} "
        f"grad={value('gradient_norm')} lr={learning_rate:.3g}"
    )


def _update_collapse_guard(
    metrics: dict[str, float | None],
    consecutive_windows: int,
) -> int:
    positive_probability = metrics["valid_angle_center_prob_mean"]
    gradient_norm = metrics["gradient_norm"]
    if gradient_norm is None:
        raise ValueError("collapse guard requires a finite gradient norm")
    if positive_probability is None:
        return consecutive_windows
    collapsed = (
        positive_probability <= 1e-6
        and gradient_norm <= 1e-8
    )
    consecutive_windows = consecutive_windows + 1 if collapsed else 0
    if consecutive_windows >= 3:
        raise RuntimeError(
            "suspected all-background collapse: positive center "
            "probability <= 1e-6 and pre-clip gradient norm <= 1e-8 "
            "for three consecutive log windows; do not resume this run"
        )
    return consecutive_windows


@dataclass
class CollapseGuard:
    consecutive_windows: int = 0

    def update(self, metrics: dict[str, float | None]) -> None:
        self.consecutive_windows = _update_collapse_guard(
            metrics,
            self.consecutive_windows,
        )


def _validate_valid_angle_masks(valid_angle_masks: torch.Tensor) -> torch.Tensor:
    """Validate and return a bool ``[B,36]`` equivalent-angle mask."""
    expected_width = len(ANGLE_DEGREES)
    if valid_angle_masks.ndim != 2 or valid_angle_masks.shape[1] != expected_width:
        raise ValueError(
            "valid_angle_masks must have shape "
            f"[B,{expected_width}], got {tuple(valid_angle_masks.shape)}"
        )
    valid_angle_masks = valid_angle_masks.to(dtype=torch.bool)
    valid_counts = valid_angle_masks.sum(dim=1)
    if not bool(torch.all(valid_counts >= 1)):
        raise ValueError(
            "each image must have at least one valid angle bin; got "
            f"{valid_counts.tolist()}"
        )
    return valid_angle_masks


def _build_training_targets(
    heatmaps: torch.Tensor,
    valid_angle_masks: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    canvas_size: int,
    align_corners: bool,
    invalid_angle_peak: float,
) -> torch.Tensor:
    """Rotate center targets into K raw FCN views and scale their peaks."""
    if not math.isfinite(float(invalid_angle_peak)) or not 0.0 <= float(
        invalid_angle_peak
    ) <= 0.1:
        raise ValueError("invalid_angle_peak must be in [0, 0.1]")
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 1:
        raise ValueError(
            f"heatmaps must have shape [B,1,H,W], got {tuple(heatmaps.shape)}"
        )
    if angles_deg.ndim != 1 or angles_deg.numel() == 0:
        raise ValueError("angles_deg must be a non-empty one-dimensional tensor")
    batch_size = int(heatmaps.shape[0])
    num_angles = int(angles_deg.numel())
    if valid_angle_masks.shape != (batch_size, num_angles):
        raise ValueError(
            "valid_angle_masks must match the requested angle chunk; got "
            f"{tuple(valid_angle_masks.shape)}, expected "
            f"({batch_size}, {num_angles})"
        )

    padded, _ = pad_to_square(heatmaps, canvas_size=canvas_size, value=0.0)
    square_h, square_w = padded.shape[-2:]
    expanded_targets = (
        padded.unsqueeze(1)
        .expand(batch_size, num_angles, 1, square_h, square_w)
        .reshape(batch_size * num_angles, 1, square_h, square_w)
    )
    expanded_angles = (
        angles_deg.unsqueeze(0)
        .expand(batch_size, num_angles)
        .reshape(batch_size * num_angles)
    )
    rotated_targets = rotate_tensor(
        expanded_targets,
        expanded_angles,
        mode="bilinear",
        align_corners=align_corners,
        padding_value=0.0,
    )
    maxima = rotated_targets.amax(dim=(-2, -1), keepdim=True)
    if not bool(torch.all(maxima > 0.0)):
        raise ValueError("every rotated training heatmap must contain a peak")
    normalized = rotated_targets / maxima
    peak_scales = torch.where(
        valid_angle_masks.reshape(-1),
        1.0,
        float(invalid_angle_peak),
    ).to(dtype=heatmaps.dtype)
    return normalized * peak_scales[:, None, None, None]


def train_one_epoch(
    model: RotationEquivariantFCN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: HeatmapFocalLoss,
    device: torch.device,
    epoch: int,
    max_train_batches: int = 0,
    log_every: int = 25,
    collapse_guard: CollapseGuard | None = None,
    scan_chunk_size: int = 2,
    valid_angle_weight: float = 2.0,
    invalid_angle_peak: float = 0.1,
) -> dict[str, float | None]:
    if scan_chunk_size <= 0:
        raise ValueError("scan_chunk_size must be positive")
    if not math.isfinite(float(valid_angle_weight)) or valid_angle_weight <= 0:
        raise ValueError("valid_angle_weight must be finite and positive")
    model.train()
    epoch_metrics = TrainingMetricsAccumulator()
    window_metrics = TrainingMetricsAccumulator()
    if collapse_guard is None:
        collapse_guard = CollapseGuard()
    batch_count = len(dataloader)
    if max_train_batches:
        batch_count = min(batch_count, max_train_batches)
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= batch_count:
            break

        images = batch["image"].to(device)
        heatmaps = batch["hole_label"].to(device)
        valid_angle_masks = _validate_valid_angle_masks(
            batch["valid_angle_mask"].to(device)
        )

        batch_size = int(images.shape[0])
        angle_values = torch.as_tensor(
            ANGLE_DEGREES,
            device=device,
            dtype=torch.float32,
        )

        optimizer.zero_grad(set_to_none=True)
        batch_metrics = TrainingMetricsAccumulator()
        loss_values = {"loss": 0.0, "positive_loss": 0.0, "background_loss": 0.0}
        view_denominator = float(batch_size * len(ANGLE_DEGREES))

        for angle_start in range(0, len(ANGLE_DEGREES), scan_chunk_size):
            angle_end = min(angle_start + scan_chunk_size, len(ANGLE_DEGREES))
            angles = angle_values[angle_start:angle_end]
            valid_chunk = valid_angle_masks[:, angle_start:angle_end]
            raw_logits = model.scan_raw(
                images,
                chunk_size=int(angles.numel()),
                angles_deg=angles,
            )
            flat_logits = raw_logits.reshape(
                batch_size * int(angles.numel()),
                1,
                raw_logits.shape[-2],
                raw_logits.shape[-1],
            )
            targets = _build_training_targets(
                heatmaps,
                valid_chunk,
                angles,
                canvas_size=model.canvas_size,
                align_corners=model.align_corners,
                invalid_angle_peak=invalid_angle_peak,
            )
            flat_valid = valid_chunk.reshape(-1)
            view_weights = torch.where(
                flat_valid,
                float(valid_angle_weight),
                1.0,
            )

            per_view_loss, components = criterion(
                flat_logits,
                targets,
                return_components=True,
                reduction="none",
            )
            weighted_loss = (per_view_loss * view_weights).sum()
            weighted_positive = (components["positive"] * view_weights).sum()
            weighted_background = (components["background"] * view_weights).sum()
            chunk_loss = weighted_loss / view_denominator
            if not bool(torch.isfinite(chunk_loss)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "non-finite training loss at "
                    f"epoch={epoch + 1}, batch={batch_index}, "
                    f"angles={angle_start}:{angle_end}"
                )
            chunk_loss.backward()
            loss_values["loss"] += float(chunk_loss.detach())
            loss_values["positive_loss"] += float(
                (weighted_positive / view_denominator).detach()
            )
            loss_values["background_loss"] += float(
                (weighted_background / view_denominator).detach()
            )
            batch_metrics.add_tensor_statistics(
                flat_logits,
                heatmap_peak_mask(targets),
                flat_valid,
            )

        batch_metrics.add_batch_losses(
            loss_values,
            batch_size,
        )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        gradient_norm_value = float(gradient_norm)
        if not math.isfinite(gradient_norm_value):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "non-finite gradient norm at "
                f"epoch={epoch + 1}, batch={batch_index}"
            )
        optimizer.step()
        batch_metrics.add_gradient_norm(gradient_norm_value)

        epoch_metrics.merge(batch_metrics)
        window_metrics.merge(batch_metrics)
        should_log = (
            batch_index % log_every == 0
            or batch_index + 1 == batch_count
        )
        if should_log:
            summary = window_metrics.summary()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            print(
                _format_training_window(
                    batch_index,
                    batch_count,
                    summary,
                    learning_rate,
                )
            )
            collapse_guard.update(summary)
            window_metrics = TrainingMetricsAccumulator()

    if epoch_metrics.sample_count == 0:
        raise RuntimeError("training loader produced no samples")
    return epoch_metrics.summary()


def _score_predictions(
    center_errors: list[float],
    angle_errors: list[float],
    shapes: list[str],
) -> dict[str, float]:
    sample_count = max(len(center_errors), 1)
    center_hits = sum(1 for e in center_errors if e <= 10.0)
    angle_hits = sum(1 for error in angle_errors if abs(error) <= 10.0)
    joint_hits = sum(
        1
        for center_error, angle_error in zip(center_errors, angle_errors)
        if center_error <= 10.0 and abs(angle_error) <= 10.0
    )

    non_circle_errors = [
        abs(ae) for ae, s in zip(angle_errors, shapes) if s != "circle"
    ]
    return {
        "center_mean_px": float(np.mean(center_errors)) if center_errors else math.inf,
        "center_p95_px": (
            float(np.percentile(center_errors, 95)) if center_errors else math.inf
        ),
        "center_hit_10px": center_hits / sample_count,
        "angle_mae_deg": (
            float(np.mean(np.abs(angle_errors))) if angle_errors else math.inf
        ),
        "non_circle_angle_mae_deg": (
            float(np.mean(non_circle_errors)) if non_circle_errors else 0.0
        ),
        "angle_hit_10deg": angle_hits / sample_count,
        "joint_success": joint_hits / sample_count,
    }


def validate_full_scan(
    model: RotationEquivariantFCN,
    dataloader: DataLoader,
    device: torch.device,
    scan_chunk_size: int,
    sample_limit: int = 0,
):
    model.eval()
    center_errors: list[float] = []
    angle_errors: list[float] = []
    shapes: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            centers = batch["center"].float()
            valid_angle_masks = _validate_valid_angle_masks(
                batch["valid_angle_mask"]
            )

            results = model.find_best_angle(
                images,
                chunk_size=scan_chunk_size,
            )

            for sample_index in range(images.shape[0]):
                result = results[sample_index]
                pred_x, pred_y = result["center"]
                pred_rotation = float(result["rotation_deg"])
                target_x = float(centers[sample_index, 0])
                target_y = float(centers[sample_index, 1])
                scale = float(batch["downsample_factor"][sample_index])
                center_errors.append(
                    math.hypot(pred_x - target_x, pred_y - target_y) * scale
                )

                valid_bins = torch.nonzero(
                    valid_angle_masks[sample_index],
                    as_tuple=False,
                ).flatten().tolist()
                candidate_diffs = [
                    (
                        pred_rotation
                        - float(ANGLE_DEGREES[bin_index])
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                    for bin_index in valid_bins
                ]
                diff = min(candidate_diffs, key=lambda value: (abs(value), value))
                if abs(diff) < 1e-9:
                    diff = 0.0
                angle_errors.append(diff)

                shapes.append(batch["shape"][sample_index])

                if sample_limit and len(center_errors) >= sample_limit:
                    break
            if sample_limit and len(center_errors) >= sample_limit:
                break

    if not center_errors:
        raise RuntimeError("validation split is empty")
    return _score_predictions(center_errors, angle_errors, shapes), {}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: RotationEquivariantFCN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: dict,
    args,
    *,
    run_id: str,
    training_metrics: dict | None = None,
    best_joint_success: float = -math.inf,
    checkpoint_kind: str = "best",
    data_generator: torch.Generator | None = None,
    collapse_guard_windows: int = 0,
):
    if not run_id:
        raise ValueError("run_id must be a non-empty string")
    if checkpoint_kind not in {"best", "last"}:
        raise ValueError("checkpoint_kind must be 'best' or 'last'")
    if collapse_guard_windows < 0:
        raise ValueError("collapse_guard_windows cannot be negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        ),
        "data_generator": (
            data_generator.get_state() if data_generator is not None else None
        ),
    }
    payload = {
        "model_architecture": MODEL_ARCHITECTURE,
        "training_objective": TRAINING_OBJECTIVE,
        "angle_label_policy": ANGLE_LABEL_POLICY,
        "checkpoint_kind": checkpoint_kind,
        "run_id": run_id,
        "resolved_data_dir": str(Path(args.data_dir).resolve()),
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "angles_deg": [float(value) for value in ANGLE_DEGREES],
        "canvas_size": args.canvas_size,
        "downsample_factor": args.downsample_factor,
        "metrics": metrics,
        "training_metrics": training_metrics or {},
        "best_joint_success": float(best_joint_success),
        "collapse_guard_windows": int(collapse_guard_windows),
        "rng_state": rng_state,
        "config": vars(args),
    }
    temporary_path = path.with_name(f".{path.name}.{run_id}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _validate_training_args(args) -> None:
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.scan_chunk_size <= 0:
        raise ValueError("--scan-chunk-size must be positive")
    if args.full_val_every <= 0:
        raise ValueError("--full-val-every must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.max_train_batches < 0:
        raise ValueError("--max-train-batches cannot be negative")
    if args.val_limit < 0 or args.test_limit < 0:
        raise ValueError("--val-limit and --test-limit cannot be negative")
    for name in (
        "learning_rate",
        "background_loss_weight",
        "valid_angle_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
        if name == "learning_rate" and value <= 0:
            raise ValueError("--learning-rate must be positive")
        if name != "learning_rate" and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    invalid_angle_peak = float(args.invalid_angle_peak)
    if (
        not math.isfinite(invalid_angle_peak)
        or not 0.0 <= invalid_angle_peak <= 0.1
    ):
        raise ValueError("--invalid-angle-peak must be in [0, 0.1]")
    if args.downsample_factor <= 0:
        raise ValueError("--downsample-factor must be positive")
    if Path(args.checkpoint).resolve() == Path(args.last_checkpoint).resolve():
        raise ValueError("--checkpoint and --last-checkpoint must be different files")


def _load_resume_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"resume checkpoint not found: {path}; "
            "start without --resume or provide --last-checkpoint"
        )
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"cannot resume {path}: expected a training checkpoint dictionary"
        )
    objective = checkpoint.get("training_objective")
    if objective != TRAINING_OBJECTIVE:
        found = "missing" if objective is None else repr(objective)
        raise ValueError(
            f"cannot resume {path}: training_objective is {found}, "
            f"expected {TRAINING_OBJECTIVE!r}; do not resume a legacy or "
            "collapsed run"
        )
    if checkpoint.get("angle_label_policy") != ANGLE_LABEL_POLICY:
        raise ValueError(
            f"cannot resume {path}: angle_label_policy must be "
            f"{ANGLE_LABEL_POLICY!r}"
        )
    checkpoint_kind = checkpoint.get("checkpoint_kind")
    if checkpoint_kind != "last":
        found = "missing" if checkpoint_kind is None else repr(checkpoint_kind)
        raise ValueError(
            f"cannot resume {path}: checkpoint_kind is {found}, expected "
            "'last'; --resume must use the latest completed epoch checkpoint"
        )
    required = (
        "run_id",
        "resolved_data_dir",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "best_joint_success",
        "collapse_guard_windows",
        "config",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            f"cannot resume {path}: missing fields {', '.join(missing)}"
        )
    return checkpoint


RESUME_CONFIG_FIELDS = (
    "data_dir",
    "seed",
    "batch_size",
    "canvas_size",
    "downsample_factor",
    "learning_rate",
    "background_loss_weight",
    "valid_angle_weight",
    "invalid_angle_peak",
    "scan_chunk_size",
    "max_train_batches",
    "log_every",
    "full_val_every",
    "val_limit",
)


def _validate_resume_config(checkpoint: dict, args) -> None:
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("cannot resume checkpoint: missing saved training config")

    mismatches = []
    for name in RESUME_CONFIG_FIELDS:
        if name not in saved_config:
            mismatches.append(f"{name}=missing")
            continue
        saved_value = saved_config[name]
        current_value = getattr(args, name)
        if name == "data_dir":
            saved_value = str(checkpoint["resolved_data_dir"]).casefold()
            current_value = str(Path(current_value).resolve()).casefold()
        if saved_value != current_value:
            mismatches.append(
                f"{name}: checkpoint={saved_value!r}, current={current_value!r}"
            )
    if mismatches:
        raise ValueError(
            "cannot exactly resume because training configuration changed: "
            + "; ".join(mismatches)
            + ". Use the original arguments, or use --init-checkpoint for "
            "a deliberate weights-only restart."
        )


def _load_best_checkpoint(
    path: Path,
    device: torch.device,
    run_id: str,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"best checkpoint not found: {path}")
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid best checkpoint dictionary: {path}")
    expected = {
        "training_objective": TRAINING_OBJECTIVE,
        "angle_label_policy": ANGLE_LABEL_POLICY,
        "checkpoint_kind": "best",
        "run_id": run_id,
    }
    mismatches = [
        f"{name}={checkpoint.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if checkpoint.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            f"best checkpoint {path} does not belong to the active run: "
            + "; ".join(mismatches)
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"best checkpoint {path} is missing model_state_dict")
    return checkpoint


def _restore_training_rng_state(
    checkpoint: dict,
    data_generator: torch.Generator,
) -> None:
    rng_state = checkpoint["rng_state"]
    if not isinstance(rng_state, dict):
        raise ValueError("cannot exactly resume checkpoint: invalid RNG state")
    required = ("python", "numpy", "torch", "data_generator")
    missing = [name for name in required if rng_state.get(name) is None]
    if missing:
        raise ValueError(
            "cannot exactly resume checkpoint: missing RNG state "
            + ", ".join(missing)
        )
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in cuda_state]
        )
    data_generator.set_state(rng_state["data_generator"].cpu())


def main():
    args = parse_args()
    _validate_training_args(args)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_dataset = RectangleDataset(
        args.data_dir,
        split="train",
        downsample_factor=args.downsample_factor,
    )
    val_dataset = RectangleDataset(
        args.data_dir,
        split="val",
        downsample_factor=args.downsample_factor,
    )
    test_dataset = RectangleDataset(
        args.data_dir,
        split="test",
        downsample_factor=args.downsample_factor,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    base_fcn = FCN(num_input_channels=1, num_output_channels=1)
    model = RotationEquivariantFCN(
        base_fcn,
        angles_deg=ANGLE_DEGREES,
        canvas_size=args.canvas_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    criterion = HeatmapFocalLoss(
        background_weight=args.background_loss_weight,
    )
    checkpoint_path = Path(args.checkpoint)
    last_checkpoint_path = Path(args.last_checkpoint)
    start_epoch = 0
    best_joint_success = -math.inf
    latest_val_metrics: dict = {}
    run_id = uuid.uuid4().hex
    collapse_guard = CollapseGuard()

    if args.resume:
        checkpoint = _load_resume_checkpoint(last_checkpoint_path, device)
        _validate_resume_config(checkpoint, args)
        run_id = str(checkpoint["run_id"])
        if not run_id:
            raise ValueError("cannot resume checkpoint with an empty run_id")
        saved_best_joint_success = float(checkpoint["best_joint_success"])
        collapse_guard.consecutive_windows = int(
            checkpoint["collapse_guard_windows"]
        )
        if collapse_guard.consecutive_windows < 0:
            raise ValueError("checkpoint collapse_guard_windows cannot be negative")
        if checkpoint_path.is_file():
            best_checkpoint = _load_best_checkpoint(
                checkpoint_path,
                device,
                run_id,
            )
            file_best_joint_success = float(
                best_checkpoint.get("metrics", {}).get(
                    "joint_success",
                    -math.inf,
                )
            )
            if not math.isclose(
                saved_best_joint_success,
                file_best_joint_success,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "latest and best checkpoints disagree about "
                    "best_joint_success; use a matching checkpoint pair"
                )
            best_joint_success = saved_best_joint_success
        elif math.isfinite(saved_best_joint_success):
            raise FileNotFoundError(
                f"best checkpoint for resumed run is missing: {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_training_rng_state(checkpoint, generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        latest_val_metrics = dict(checkpoint.get("metrics", {}))
        print(f"resumed from {last_checkpoint_path} at epoch {start_epoch}")
    else:
        existing_outputs = [
            path
            for path in (checkpoint_path, last_checkpoint_path)
            if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "refusing to overwrite checkpoint files from another run: "
                + ", ".join(str(path) for path in existing_outputs)
                + "; choose new --checkpoint/--last-checkpoint paths or "
                "remove the old files explicitly"
            )
        if args.init_checkpoint:
            _load_initial_weights(
                base_fcn,
                model,
                args.init_checkpoint,
                device,
            )
            print(f"initialized from {args.init_checkpoint}")

    active_learning_rate = float(optimizer.param_groups[0]["lr"])
    if args.compile:
        model.compile_scan()
        print("compile=aot_eager (scan_raw and scan)")
    print(
        f"objective={TRAINING_OBJECTIVE} run_id={run_id} "
        f"lr={active_learning_rate:g} "
        f"downsample={args.downsample_factor}x "
        f"background_weight={args.background_loss_weight:g} "
        f"valid_angle_weight={args.valid_angle_weight:g} "
        f"invalid_angle_peak={args.invalid_angle_peak:g}"
    )

    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            max_train_batches=args.max_train_batches,
            log_every=args.log_every,
            collapse_guard=collapse_guard,
            scan_chunk_size=args.scan_chunk_size,
            valid_angle_weight=args.valid_angle_weight,
            invalid_angle_peak=args.invalid_angle_peak,
        )
        print(
            "train="
            + json.dumps(
                train_metrics,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        should_validate = (
            (epoch + 1) % args.full_val_every == 0
            or epoch + 1 == args.epochs
        )
        if should_validate:
            metrics, _ = validate_full_scan(
                model,
                val_loader,
                device,
                args.scan_chunk_size,
                sample_limit=args.val_limit,
            )
            latest_val_metrics = metrics
            scheduler.step(metrics["joint_success"])
            print(
                "val="
                + json.dumps(metrics, ensure_ascii=False, sort_keys=True)
            )

            if metrics["joint_success"] > best_joint_success:
                best_joint_success = metrics["joint_success"]
                save_checkpoint(
                    checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    metrics,
                    args,
                    run_id=run_id,
                    training_metrics=train_metrics,
                    best_joint_success=best_joint_success,
                    checkpoint_kind="best",
                    data_generator=generator,
                    collapse_guard_windows=(
                        collapse_guard.consecutive_windows
                    ),
                )
                print(f"saved best checkpoint to {checkpoint_path}")

        save_checkpoint(
            last_checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            latest_val_metrics,
            args,
            run_id=run_id,
            training_metrics=train_metrics,
            best_joint_success=best_joint_success,
            checkpoint_kind="last",
            data_generator=generator,
            collapse_guard_windows=collapse_guard.consecutive_windows,
        )
        print(f"saved latest checkpoint to {last_checkpoint_path}")

    if not checkpoint_path.exists():
        raise RuntimeError(
            "training finished without producing a best checkpoint; "
            "increase --epochs or run without a completed --resume state"
        )

    best_checkpoint = _load_best_checkpoint(
        checkpoint_path,
        device,
        run_id,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics, _ = validate_full_scan(
        model,
        test_loader,
        device,
        args.scan_chunk_size,
        sample_limit=args.test_limit,
    )
    print("test=" + json.dumps(test_metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
