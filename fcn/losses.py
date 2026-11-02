from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def heatmap_peak_mask(targets: torch.Tensor) -> torch.Tensor:
    """Return every spatial maximum of each non-empty target heatmap.

    The final two dimensions are treated as the spatial dimensions.  Keeping
    every equal maximum is important for Gaussian targets whose projected
    center lies exactly between two or four pixels.
    """

    if targets.ndim < 2:
        raise ValueError(
            "targets must have at least two spatial dimensions, "
            f"got {tuple(targets.shape)}"
        )
    if targets.shape[-2] == 0 or targets.shape[-1] == 0:
        raise ValueError("target heatmaps must have non-empty spatial dimensions")

    target_maximum = targets.amax(dim=(-2, -1), keepdim=True)
    return targets.eq(target_maximum) & target_maximum.gt(0.0)


def _loss_normalizer(
    value: torch.Tensor | float | None,
    default: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        value = default
    return torch.as_tensor(
        value,
        dtype=reference.dtype,
        device=reference.device,
    ).clamp_min(1.0)


class HeatmapFocalLoss(nn.Module):
    """Numerically stable CenterNet-style focal loss with soft center peaks.

    Center peaks and spatial background pixels are normalized independently.
    A peak target of ``1.0`` retains the standard CenterNet positive loss,
    while a soft peak such as ``0.1`` is optimized toward that probability
    instead of being treated as a hard positive.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        background_weight: float = 1.0,
    ):
        super().__init__()
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if background_weight < 0.0:
            raise ValueError("background_weight must be non-negative")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.background_weight = float(background_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        normalizer: torch.Tensor | float | None = None,
        background_normalizer: torch.Tensor | float | None = None,
        return_components: bool = False,
        reduction: str = "mean",
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape, got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        if reduction == "none" and (
            normalizer is not None or background_normalizer is not None
        ):
            raise ValueError(
                "explicit normalizers are not supported with reduction='none'"
            )

        targets = targets.to(dtype=logits.dtype)
        probabilities = torch.sigmoid(logits)
        positive_mask = heatmap_peak_mask(targets)
        background_mask = ~positive_mask
        positive_weights = positive_mask.to(dtype=logits.dtype)
        background_weights = (
            torch.pow(1.0 - targets, self.beta)
            * background_mask.to(dtype=logits.dtype)
        )

        # Soft-target BCE makes the optimum center probability equal to the
        # target peak. For target=1 this is exactly the original
        # softplus(-logit) * (1 - probability)^alpha positive focal term.
        positive_bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        positive_unreduced = (
            positive_bce
            * torch.pow(torch.abs(targets - probabilities), self.alpha)
            * positive_weights
        )
        background_unreduced = (
            F.softplus(logits)
            * torch.pow(probabilities, self.alpha)
            * background_weights
        )

        if reduction == "none":
            if logits.ndim < 2:
                raise ValueError("reduction='none' requires a batch dimension")
            positive_counts = positive_mask.reshape(logits.shape[0], -1).sum(1)
            background_counts = background_mask.reshape(logits.shape[0], -1).sum(1)
            positive_loss = positive_unreduced.reshape(
                logits.shape[0], -1
            ).sum(1) / positive_counts.clamp_min(1).to(logits.dtype)
            background_loss = (
                self.background_weight
                * background_unreduced.reshape(logits.shape[0], -1).sum(1)
                / background_counts.clamp_min(1).to(logits.dtype)
            )
        else:
            positive_denominator = _loss_normalizer(
                normalizer,
                positive_mask.sum(),
                logits,
            )
            background_denominator = _loss_normalizer(
                background_normalizer,
                background_mask.sum(),
                logits,
            )
            positive_loss = positive_unreduced.sum() / positive_denominator
            background_loss = (
                self.background_weight
                * background_unreduced.sum()
                / background_denominator
            )
        total_loss = positive_loss + background_loss

        if return_components:
            return total_loss, {
                "positive": positive_loss,
                "background": background_loss,
            }
        return total_loss
