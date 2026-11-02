import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from dataset import ANGLE_DEGREES, ANGLE_LABEL_POLICY, rescale_pixel_coordinate
from models import FCN
from rotation_equivariant import RotationEquivariantFCN


BASE_NETWORK_HEIGHT = 480
BASE_NETWORK_WIDTH = 640
EXPECTED_MODEL_ARCHITECTURE = "resnet18_fcn_groupnorm_heatmap_only_v2"
EXPECTED_TRAINING_OBJECTIVE = "heatmap_focal_all_equivalent_angles_v4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate one hole and select the highest-peak rotation branch."
    )
    parser.add_argument("image_path")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="best_heatmap_all_angles_v4.pth",
    )
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=None,
        help="Override the checkpoint downsample factor.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Lightly compile scan path with aot_eager "
            "(use --no-compile to disable)."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_model(
    model_path: Path,
    device: torch.device,
    compile_scan: bool = True,
    downsample_factor: int | None = None,
):
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        objective = checkpoint.get("training_objective")
        architecture = checkpoint.get("model_architecture")
        if objective != EXPECTED_TRAINING_OBJECTIVE:
            raise ValueError(
                f"incompatible checkpoint objective {objective!r}; expected "
                f"{EXPECTED_TRAINING_OBJECTIVE!r}. Legacy v3 weights may only "
                "be used with train.py --init-checkpoint and retrained."
            )
        if architecture != EXPECTED_MODEL_ARCHITECTURE:
            raise ValueError(
                f"incompatible model architecture {architecture!r}; expected "
                f"{EXPECTED_MODEL_ARCHITECTURE!r}"
            )
        if checkpoint.get("angle_label_policy") != ANGLE_LABEL_POLICY:
            raise ValueError(
                "incompatible checkpoint angle_label_policy; expected "
                f"{ANGLE_LABEL_POLICY!r}"
            )
        state_dict = checkpoint["model_state_dict"]
        angles_deg = checkpoint.get("angles_deg")
        canvas_size = int(checkpoint.get("canvas_size", 400))
        saved_downsample_factor = int(checkpoint.get("downsample_factor", 2))
    else:
        raise ValueError(
            "inference requires a v4 training checkpoint dictionary; raw or "
            "legacy weights may only initialize a new training run"
        )

    resolved_angles = (
        [] if angles_deg is None else [float(value) for value in angles_deg]
    )
    expected_angles = [float(value) for value in ANGLE_DEGREES]
    if resolved_angles != expected_angles:
        raise ValueError(
            "checkpoint must contain the fixed 36 scanner angles 0..350 "
            "in 10-degree steps"
        )

    resolved_downsample_factor = (
        saved_downsample_factor
        if downsample_factor is None
        else int(downsample_factor)
    )
    if resolved_downsample_factor <= 0:
        raise ValueError("--downsample-factor must be positive")
    if resolved_downsample_factor != saved_downsample_factor:
        canvas_size = max(
            1,
            round(
                canvas_size
                * saved_downsample_factor
                / resolved_downsample_factor
            ),
        )

    base_fcn = FCN(
        num_input_channels=1,
        num_output_channels=1,
    )
    model = RotationEquivariantFCN(
        base_fcn,
        angles_deg=angles_deg,
        canvas_size=canvas_size,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    if compile_scan:
        model.compile_scan()
    return (
        model,
        resolved_angles,
        resolved_downsample_factor,
    )


def prepare_image(
    image_path: Path,
    device: torch.device,
    downsample_factor: int,
):
    image = Image.open(image_path).convert("L")
    original_size = image.size
    resized = image.resize(
        (
            max(1, round(BASE_NETWORK_WIDTH / downsample_factor)),
            max(1, round(BASE_NETWORK_HEIGHT / downsample_factor)),
        ),
        resample=Image.Resampling.BILINEAR,
    )
    tensor = vision_functional.to_tensor(resized).unsqueeze(0).to(device)
    return image, resized, tensor, original_size


def predict_single_image(
    model: RotationEquivariantFCN,
    image_tensor: torch.Tensor,
    chunk_size: int,
):
    """Run one raw 36-angle scan and return decoded visualization data."""

    with torch.no_grad():
        results, raw_probabilities, aligned_best = model.scan_and_decode(
            image_tensor,
            chunk_size=chunk_size,
        )
        result = results[0]

    raw_probabilities = raw_probabilities[0].cpu()
    per_angle_scores = raw_probabilities.amax(dim=(-2, -1)).numpy()
    return result, per_angle_scores, aligned_best[0].cpu()


def map_result_to_original(
    decoded: dict,
    original_size: tuple[int, int],
    network_size: tuple[int, int],
):
    original_width, original_height = original_size
    network_width, network_height = network_size
    network_x, network_y = decoded["center"]
    return {
        "center": [
            rescale_pixel_coordinate(network_x, network_width, original_width),
            rescale_pixel_coordinate(network_y, network_height, original_height),
        ],
        "angle_index": int(decoded["angle_index"]),
        "rotation_deg": float(decoded["rotation_deg"]),
        "correction_deg": float(decoded["correction_deg"]),
        "center_score": float(decoded["center_score"]),
    }


def save_visualization(
    resized_image: Image.Image,
    decoded: dict,
    best_heatmap: torch.Tensor,
    per_angle_scores: np.ndarray,
    angles_deg,
    output_path: Path,
    show: bool,
):
    image_array = np.asarray(resized_image)
    center_x, center_y = decoded["center"]
    correction = float(decoded["correction_deg"])
    corrected = vision_functional.rotate(
        resized_image,
        correction,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        center=[float(center_x), float(center_y)],
        fill=0,
    )

    best_heatmap = best_heatmap.numpy()

    signed_angles = np.array(
        [((float(value) + 180.0) % 360.0) - 180.0 for value in angles_deg]
    )
    order = np.argsort(signed_angles)

    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_array, cmap="gray")
    axes[0].scatter([center_x], [center_y], marker="+", s=180, c="red")
    axes[0].set_title("Input and predicted center")
    axes[0].axis("off")

    heatmap_plot = axes[1].imshow(best_heatmap, cmap="jet")
    axes[1].scatter([center_x], [center_y], marker="+", s=180, c="white")
    axes[1].set_title(
        f"Heatmap @ best angle ({correction:.1f} deg)"
    )
    axes[1].axis("off")
    figure.colorbar(heatmap_plot, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(corrected, cmap="gray")
    axes[2].set_title(f"Corrected preview ({correction:.1f} deg CCW)")
    axes[2].axis("off")

    axes[3].plot(
        signed_angles[order],
        per_angle_scores[order],
        marker="o",
        markersize=3,
    )
    axes[3].axvline(correction, color="red", linestyle="--")
    axes[3].set_xlabel("rotation angle (deg)")
    axes[3].set_ylabel("peak heatmap score")
    axes[3].set_title("Per-angle center peak score")
    axes[3].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close(figure)


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    model_path = Path(args.model_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, angles_deg, downsample_factor = load_model(
        model_path,
        device,
        compile_scan=args.compile,
        downsample_factor=args.downsample_factor,
    )
    _, resized, image_tensor, original_size = prepare_image(
        image_path,
        device,
        downsample_factor,
    )
    decoded, per_angle_scores, best_heatmap = predict_single_image(
        model,
        image_tensor,
        args.chunk_size,
    )
    result = map_result_to_original(decoded, original_size, resized.size)
    print(json.dumps(result, ensure_ascii=False))

    output_path = (
        Path(args.output)
        if args.output
        else image_path.with_name(f"{image_path.stem}_rotation_result.png")
    )
    save_visualization(
        resized,
        decoded,
        best_heatmap,
        per_angle_scores,
        angles_deg,
        output_path,
        args.show,
    )
    print(f"visualization={output_path}")


if __name__ == "__main__":
    main()
