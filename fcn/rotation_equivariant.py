"""Rotation-equivariant scanning and decoding for a single-channel FCN.

The public angle convention used by this module is:

* positive angles rotate an image visually counter-clockwise;
* a scan branch at ``alpha`` rotates the input by ``alpha`` before the FCN;
* the branch logit is rotated back by ``-alpha`` before being returned; and
* consequently, the decoded branch angle is the counter-clockwise correction
  that should be applied to the original image/workpiece.

All rotations are implemented with differentiable PyTorch operations.  No
``torchvision`` angle convention is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


AngleInput = Union[Tensor, Sequence[float], Iterable[float]]


@dataclass(frozen=True)
class SquarePadInfo:
    """Geometry needed to crop a centered square back to its source size."""

    original_height: int
    original_width: int
    square_size: int
    top: int
    bottom: int
    left: int
    right: int


def pad_to_square(
    tensor: Tensor,
    canvas_size: int = 800,
    value: float = 0.0,
) -> Tuple[Tensor, SquarePadInfo]:
    """Center-pad ``tensor`` to a square without resizing its content.

    The last two dimensions are interpreted as ``(height, width)``.  Images
    smaller than ``canvas_size`` are padded to exactly that size.  Larger
    images are padded to the smallest square that contains them.
    """

    if tensor.ndim < 2:
        raise ValueError("tensor must have at least two dimensions (..., H, W)")
    if canvas_size <= 0:
        raise ValueError("canvas_size must be a positive integer")

    height, width = (int(tensor.shape[-2]), int(tensor.shape[-1]))
    if height <= 0 or width <= 0:
        raise ValueError("input height and width must both be positive")

    square_size = max(int(canvas_size), height, width)
    delta_h = square_size - height
    delta_w = square_size - width
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    info = SquarePadInfo(
        original_height=height,
        original_width=width,
        square_size=square_size,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    )
    return F.pad(tensor, (left, right, top, bottom), value=float(value)), info


def crop_from_square(tensor: Tensor, pad_info: SquarePadInfo) -> Tensor:
    """Crop the last two dimensions using geometry returned by
    :func:`pad_to_square`.
    """

    if tensor.ndim < 2:
        raise ValueError("tensor must have at least two dimensions (..., H, W)")
    if (
        int(tensor.shape[-2]) != pad_info.square_size
        or int(tensor.shape[-1]) != pad_info.square_size
    ):
        raise ValueError(
            "tensor spatial size does not match pad_info.square_size: "
            f"got {tuple(tensor.shape[-2:])}, expected "
            f"({pad_info.square_size}, {pad_info.square_size})"
        )

    y0 = pad_info.top
    x0 = pad_info.left
    return tensor[
        ...,
        y0 : y0 + pad_info.original_height,
        x0 : x0 + pad_info.original_width,
    ]


def _coerce_angles(
    angles_deg: Union[float, int, AngleInput],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    angles = torch.as_tensor(angles_deg, device=device, dtype=dtype).flatten()
    if angles.numel() == 1:
        return angles.expand(batch_size)
    if angles.numel() != batch_size:
        raise ValueError(
            "angles_deg must be a scalar or contain one angle per image; "
            f"got {angles.numel()} angles for batch size {batch_size}"
        )
    return angles


def rotate_tensor(
    tensor: Tensor,
    angles_deg: Union[float, int, AngleInput],
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    """Rotate a batch around its image center.

    Positive angles produce a *visual counter-clockwise* rotation.  A scalar
    angle is broadcast across the batch; otherwise one angle per batch element
    is required.  Spatial size is preserved, and pixels outside the source
    image are filled with ``padding_value``.

    Unlike a naive normalized-coordinate affine matrix, this implementation
    includes the height/width ratio, so it also represents a true pixel-space
    rotation for non-square tensors.
    """

    if tensor.ndim != 4:
        raise ValueError(
            "rotate_tensor expects a [B, C, H, W] tensor; "
            f"got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError("rotate_tensor requires a floating-point tensor")
    if mode not in {"nearest", "bilinear", "bicubic"}:
        raise ValueError("mode must be 'nearest', 'bilinear', or 'bicubic'")

    batch_size, _, height, width = tensor.shape
    if batch_size == 0:
        return tensor.clone()
    if height <= 0 or width <= 0:
        raise ValueError("input height and width must both be positive")

    # Grid construction in float32 is supported on CPU and is sufficiently
    # precise for image coordinates.  CUDA float64 inputs retain float64.
    grid_dtype = (
        torch.float64
        if tensor.dtype == torch.float64
        else torch.float32
    )
    angles = _coerce_angles(
        angles_deg,
        device=tensor.device,
        dtype=grid_dtype,
        batch_size=batch_size,
    )
    radians = torch.deg2rad(angles)
    cosines = torch.cos(radians)
    sines = torch.sin(radians)

    if align_corners:
        x_scale = float(max(width - 1, 1))
        y_scale = float(max(height - 1, 1))
    else:
        x_scale = float(width)
        y_scale = float(height)

    # affine_grid maps output coordinates to input coordinates.  This inverse
    # mapping yields a visual CCW transformation for a positive input angle.
    theta = torch.zeros(
        (batch_size, 2, 3),
        device=tensor.device,
        dtype=grid_dtype,
    )
    theta[:, 0, 0] = cosines
    theta[:, 0, 1] = -sines * (y_scale / x_scale)
    theta[:, 1, 0] = sines * (x_scale / y_scale)
    theta[:, 1, 1] = cosines

    grid = F.affine_grid(theta, tensor.shape, align_corners=align_corners)
    sample_input = tensor
    if sample_input.dtype != grid.dtype:
        # CPU grid_sample does not support fp16/bfloat16.  Convert only for the
        # geometric operation and restore the model's dtype afterwards.
        sample_input = sample_input.to(grid.dtype)
    if padding_value != 0.0:
        sample_input = sample_input - float(padding_value)
    rotated = F.grid_sample(
        sample_input,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    if padding_value != 0.0:
        rotated = rotated + float(padding_value)
    return rotated.to(tensor.dtype)


def inverse_rotate_tensor(
    tensor: Tensor,
    angles_deg: Union[float, int, AngleInput],
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    """Apply the inverse of :func:`rotate_tensor` for the supplied angles."""

    angles = torch.as_tensor(angles_deg, device=tensor.device)
    return rotate_tensor(
        tensor,
        -angles,
        mode=mode,
        align_corners=align_corners,
        padding_value=padding_value,
    )


def _default_angles(device: Optional[torch.device] = None) -> Tensor:
    return torch.arange(0.0, 360.0, 10.0, device=device, dtype=torch.float32)


class RotationEquivariantFCN(Module):
    """Run a shared single-channel FCN over rotated views of an image."""

    def __init__(
        self,
        base_fcn: Module,
        angles_deg: Optional[AngleInput] = None,
        canvas_size: int = 800,
        padding_value: float = 0.0,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(base_fcn, Module):
            raise TypeError("base_fcn must be a torch.nn.Module")
        if canvas_size <= 0:
            raise ValueError("canvas_size must be a positive integer")

        angles = (
            _default_angles()
            if angles_deg is None
            else torch.as_tensor(angles_deg, dtype=torch.float32).flatten()
        )
        if angles.numel() == 0:
            raise ValueError("angles_deg must contain at least one angle")
        if not torch.isfinite(angles).all():
            raise ValueError("angles_deg must contain only finite values")

        self.base_fcn = base_fcn
        self.canvas_size = int(canvas_size)
        self.padding_value = float(padding_value)
        self.align_corners = bool(align_corners)
        self.register_buffer("angles_deg", angles.clone(), persistent=True)

    def compile_scan(self, backend: str = "aot_eager") -> None:
        """Compile the hot scan paths without wrapping the module.

        Keeping the module itself unwrapped preserves checkpoint state-dict
        names.  ``aot_eager`` has much lower startup cost than Inductor and is
        therefore a good fit for this very small network.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requires PyTorch 2.0 or newer")
        self.scan_raw = torch.compile(self.scan_raw, backend=backend)
        self.scan = torch.compile(self.scan, backend=backend)

    def _resolve_angles(self, angles_deg: Optional[AngleInput], x: Tensor) -> Tensor:
        if angles_deg is None:
            return self.angles_deg.to(device=x.device)
        angles = torch.as_tensor(
            angles_deg,
            device=x.device,
            dtype=torch.float32,
        ).flatten()
        if angles.numel() == 0:
            raise ValueError("angles_deg must contain at least one angle")
        if not torch.isfinite(angles).all():
            raise ValueError("angles_deg must contain only finite values")
        return angles

    def _scan_raw_chunk(
        self,
        padded: Tensor,
        angles: Tensor,
    ) -> Tensor:
        batch_size, channels, square_h, square_w = padded.shape
        num_angles = int(angles.numel())

        expanded = (
            padded.unsqueeze(1)
            .expand(batch_size, num_angles, channels, square_h, square_w)
            .reshape(batch_size * num_angles, channels, square_h, square_w)
        )
        expanded_angles = (
            angles.unsqueeze(0)
            .expand(batch_size, num_angles)
            .reshape(batch_size * num_angles)
        )
        rotated_inputs = rotate_tensor(
            expanded,
            expanded_angles,
            mode="bilinear",
            align_corners=self.align_corners,
            padding_value=self.padding_value,
        )

        logits = self.base_fcn(rotated_inputs)
        if not isinstance(logits, Tensor) or logits.ndim != 4:
            raise ValueError("base_fcn must return a [B, 1, H, W] tensor")
        if logits.shape[0] != batch_size * num_angles:
            raise ValueError(
                "base_fcn changed the batch size: "
                f"expected {batch_size * num_angles}, got {logits.shape[0]}"
            )
        if logits.shape[1] != 1:
            raise ValueError(
                "RotationEquivariantFCN requires a single-channel FCN; "
                f"got {logits.shape[1]} output channels"
            )
        if tuple(logits.shape[-2:]) != (square_h, square_w):
            logits = F.interpolate(
                logits,
                size=(square_h, square_w),
                mode="bilinear",
                align_corners=self.align_corners,
            )

        return logits.reshape(
            batch_size,
            num_angles,
            square_h,
            square_w,
        )

    def _align_raw_logits(
        self,
        raw_logits: Tensor,
        pad_info: SquarePadInfo,
        angles: Tensor,
    ) -> Tensor:
        batch_size, num_angles, square_h, square_w = raw_logits.shape
        expanded_angles = (
            angles.unsqueeze(0)
            .expand(batch_size, num_angles)
            .reshape(batch_size * num_angles)
        )
        aligned_logits = inverse_rotate_tensor(
            raw_logits.reshape(batch_size * num_angles, 1, square_h, square_w),
            expanded_angles,
            mode="bilinear",
            align_corners=self.align_corners,
            padding_value=0.0,
        ).reshape(batch_size, num_angles, square_h, square_w)
        return crop_from_square(aligned_logits, pad_info)

    def _scan_chunk(
        self,
        padded: Tensor,
        pad_info: SquarePadInfo,
        angles: Tensor,
    ) -> Tensor:
        raw_logits = self._scan_raw_chunk(padded, angles)
        return self._align_raw_logits(raw_logits, pad_info, angles)

    def scan_raw(
        self,
        x: Tensor,
        chunk_size: Optional[int] = None,
        angles_deg: Optional[AngleInput] = None,
    ) -> Tensor:
        """Return FCN logits before inverse rotation as ``[B,K,S,S]``.

        ``S`` is the padded square canvas size.  These are the direct outputs
        for the K rotated inputs and therefore the only logits that should be
        used to compare center-peak confidence between rotation branches.
        """
        if x.ndim != 4:
            raise ValueError(
                "scan_raw expects a [B, C, H, W] tensor; "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("scan_raw requires a non-empty batch")
        if not x.is_floating_point():
            raise TypeError("scan_raw requires a floating-point input tensor")

        angles = self._resolve_angles(angles_deg, x)
        num_angles = int(angles.numel())
        if chunk_size is None:
            chunk_size = num_angles
        if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size:
            raise ValueError("chunk_size must be a positive integer or None")
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer or None")

        padded, _ = pad_to_square(
            x,
            canvas_size=self.canvas_size,
            value=self.padding_value,
        )
        chunks = [
            self._scan_raw_chunk(padded, angles[start : start + chunk_size])
            for start in range(0, num_angles, chunk_size)
        ]
        return torch.cat(chunks, dim=1)

    def scan(
        self,
        x: Tensor,
        chunk_size: Optional[int] = None,
        angles_deg: Optional[AngleInput] = None,
    ) -> Tensor:
        """Return aligned raw logits with shape ``[B, K, H, W]``.

        ``chunk_size`` limits how many angle branches are passed through the
        shared FCN at once.  It changes memory use only, not output ordering or
        numerical semantics.  ``angles_deg`` may supply a common subset of
        angles for all items in the batch for diagnostics and unit tests.
        """

        if x.ndim != 4:
            raise ValueError(
                "scan expects a [B, C, H, W] tensor; "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("scan requires a non-empty batch")
        if not x.is_floating_point():
            raise TypeError("scan requires a floating-point input tensor")

        angles = self._resolve_angles(angles_deg, x)
        num_angles = int(angles.numel())
        if chunk_size is None:
            chunk_size = num_angles
        if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size:
            raise ValueError("chunk_size must be a positive integer or None")
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer or None")

        padded, pad_info = pad_to_square(
            x,
            canvas_size=self.canvas_size,
            value=self.padding_value,
        )
        chunks = [
            self._scan_chunk(padded, pad_info, angles[start : start + chunk_size])
            for start in range(0, num_angles, chunk_size)
        ]
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        x: Tensor,
        chunk_size: Optional[int] = None,
        angles_deg: Optional[AngleInput] = None,
    ) -> Tensor:
        return self.scan(x, chunk_size=chunk_size, angles_deg=angles_deg)

    def _decode_raw_probabilities(
        self,
        x: Tensor,
        probabilities: Tensor,
        angles: Tensor,
    ) -> tuple[List[Dict[str, Any]], Tensor]:
        """Decode direct FCN probabilities and align only the winning branch."""
        batch_size = int(x.shape[0])
        num_angles = int(angles.numel())
        padded, pad_info = pad_to_square(
            x,
            canvas_size=self.canvas_size,
            value=self.padding_value,
        )
        square_h, square_w = padded.shape[-2:]
        expected_shape = (batch_size, num_angles, square_h, square_w)
        if tuple(probabilities.shape) != expected_shape:
            raise ValueError(
                "raw probabilities must have shape "
                f"{expected_shape}, got {tuple(probabilities.shape)}"
            )

        max_per_angle = probabilities.reshape(
            batch_size, num_angles, -1
        ).amax(dim=2)
        # torch.argmax returns the first occurrence, so exact ties follow the
        # configured scanner order without applying a symmetry/minimum prior.
        best_k = max_per_angle.argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=x.device)
        selected_angles = angles[best_k]
        selected_probabilities = probabilities[
            batch_indices,
            best_k,
        ].unsqueeze(1)
        aligned_best = inverse_rotate_tensor(
            selected_probabilities,
            selected_angles,
            mode="bilinear",
            align_corners=self.align_corners,
            padding_value=0.0,
        )
        aligned_best = crop_from_square(aligned_best, pad_info)[:, 0]

        _, height, width = aligned_best.shape
        aligned_indices = aligned_best.reshape(batch_size, -1).argmax(dim=1)
        results: List[Dict[str, Any]] = []
        for batch_index in range(batch_size):
            k = int(best_k[batch_index])
            flat_idx = int(aligned_indices[batch_index])
            pred_y = flat_idx // width
            pred_x = flat_idx % width

            rotation_deg = float(angles[k].item()) % 360.0
            corr_deg = (rotation_deg + 180.0) % 360.0 - 180.0
            if abs(corr_deg) < 1e-9:
                corr_deg = 0.0

            results.append(
                {
                    "center": [pred_x, pred_y],
                    "angle_index": k,
                    "rotation_deg": rotation_deg,
                    "correction_deg": corr_deg,
                    "center_score": float(
                        max_per_angle[batch_index, k].item()
                    ),
                }
            )
        return results, aligned_best

    @torch.no_grad()
    def scan_and_decode(
        self,
        x: Tensor,
        chunk_size: Optional[int] = None,
        angles_deg: Optional[AngleInput] = None,
    ) -> tuple[List[Dict[str, Any]], Tensor, Tensor]:
        """Scan direct FCN heatmaps, decode their raw peaks, and align the winner.

        For each image in the batch, the input is rotated at K angles.  The
        FCN produces a heatmap for every rotated view, and the view whose
        direct, pre-alignment heatmap has the highest maximum value is selected.

        Returns ``(results, raw_probabilities, aligned_best_probabilities)``.
        Each result contains:
          - ``center``: ``[x, y]`` pixel coordinates in the **original** image
          - ``angle_index``: winning index in the configured scanner order,
          - ``rotation_deg``: winning scanner rotation in ``[0, 360)``,
          - ``correction_deg``: signed equivalent in ``[-180, 180)``,
          - ``center_score``: direct FCN peak before inverse interpolation.
        """
        if x.ndim != 4:
            raise ValueError(
                "scan_and_decode expects a [B, C, H, W] tensor; "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("scan_and_decode requires a non-empty batch")
        if not x.is_floating_point():
            raise TypeError("scan_and_decode requires floating-point inputs")

        angles = self._resolve_angles(angles_deg, x)
        raw_logits = self.scan_raw(
            x,
            chunk_size=chunk_size,
            angles_deg=angles,
        )
        raw_probabilities = torch.sigmoid(raw_logits)
        results, aligned_best = self._decode_raw_probabilities(
            x,
            raw_probabilities,
            angles,
        )
        return results, raw_probabilities, aligned_best

    @torch.no_grad()
    def find_best_angle(
        self,
        x: Tensor,
        chunk_size: Optional[int] = None,
        angles_deg: Optional[AngleInput] = None,
    ) -> List[Dict[str, Any]]:
        """Return the branch whose direct FCN center-heatmap peak is highest."""
        results, _, _ = self.scan_and_decode(
            x,
            chunk_size=chunk_size,
            angles_deg=angles_deg,
        )
        return results

    def export_config(self) -> Dict[str, Any]:
        """Return serializable geometry settings suitable for a checkpoint."""

        return {
            "angles_deg": [float(value) for value in self.angles_deg.cpu().tolist()],
            "canvas_size": self.canvas_size,
            "padding_value": self.padding_value,
            "align_corners": self.align_corners,
        }


def _signed_degrees(angles: Tensor) -> Tensor:
    """Normalize degrees to ``[-180, 180)``."""

    return torch.remainder(angles + 180.0, 360.0) - 180.0


__all__ = [
    "SquarePadInfo",
    "pad_to_square",
    "crop_from_square",
    "rotate_tensor",
    "inverse_rotate_tensor",
    "RotationEquivariantFCN",
]
