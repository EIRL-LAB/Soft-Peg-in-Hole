"""Manifest-backed data loading for the rotation-equivariant FCN."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


ANGLE_STEP_DEG = 10
NUM_ANGLE_BINS = 360 // ANGLE_STEP_DEG
ANGLE_DEGREES = tuple(range(0, 360, ANGLE_STEP_DEG))
ANGLE_LABEL_POLICY = "all_equivalent_angle_bins_v1"
LEGACY_CAMERA_OFFSET = 0.35
DEFAULT_GAUSSIAN_RADIUS_PIXELS = 6.0


def rescale_pixel_coordinate(
    value: float,
    source_size: int,
    target_size: int,
) -> float:
    """Rescale a pixel-center coordinate using half-pixel semantics.

    This matches PIL resizing and PyTorch's ``align_corners=False`` geometry:
    pixel centers are mapped through the continuous image domain rather than
    by matching the first and last integer indices.
    """
    if source_size <= 0 or target_size <= 0:
        raise ValueError("source_size and target_size must be positive")
    return (float(value) + 0.5) * target_size / source_size - 0.5


def create_gaussian_heatmap(
    height: int,
    width: int,
    center: Sequence[float] | Tensor,
    radius_pixels: float = DEFAULT_GAUSSIAN_RADIUS_PIXELS,
) -> Tensor:
    """Create a ``[1, H, W]`` Gaussian heatmap at a floating-point center."""
    if height <= 0 or width <= 0:
        raise ValueError("Heatmap height and width must be positive.")
    if radius_pixels <= 0:
        raise ValueError("radius_pixels must be positive.")
    if len(center) != 2:
        raise ValueError("center must contain [x, y].")

    center_x = float(center[0])
    center_y = float(center[1])
    sigma = float(radius_pixels) / 2.0
    x = torch.arange(width, dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    squared_distance = (
        (x.unsqueeze(0) - center_x).square()
        + (y.unsqueeze(1) - center_y).square()
    )
    return torch.exp(-squared_distance / (2.0 * sigma * sigma)).unsqueeze(0)


def _pil_to_float_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32)
    if not array.flags.writeable:
        array = array.copy()
    tensor = torch.from_numpy(array)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image shape: {tuple(tensor.shape)}.")
    return tensor / 255.0


def _coerce_image_tensor(image: Any) -> Tensor:
    if isinstance(image, Image.Image):
        tensor = _pil_to_float_tensor(image)
    elif isinstance(image, np.ndarray):
        array = image.copy() if not image.flags.writeable else image
        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
    elif torch.is_tensor(image):
        tensor = image
    else:
        raise TypeError(
            "transform must return a PIL image, NumPy array, or torch Tensor."
        )

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected transformed image in [C,H,W], got {tuple(tensor.shape)}."
        )
    tensor = tensor.to(dtype=torch.float32)
    if tensor.numel() and tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.shape[0] != 1:
        raise ValueError(
            f"Expected one grayscale channel, got {tensor.shape[0]} channels."
        )
    return tensor.contiguous()


class RectangleDataset(Dataset):
    """Load generated images and derive heatmaps from ``manifest.jsonl``.

    ``shape`` and ``camera_offset`` are returned for diagnostics only. Model
    code must not use the shape as an input or inference prior. A transform may
    resize the image, in which case the center is scaled accordingly. Crops,
    flips, or rotations are not supported because their geometry cannot be
    inferred generically.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[Image.Image], Any] | None = None,
        split: str | Sequence[str] | None = None,
        manifest_name: str = "manifest.jsonl",
        gaussian_radius_pixels: float = DEFAULT_GAUSSIAN_RADIUS_PIXELS,
        return_heatmap: bool = True,
        downsample_factor: int = 1,
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.root_dir}\n"
                "Please run generate.py first to generate the dataset."
            )
        self.transform = transform
        self.return_heatmap = bool(return_heatmap)
        if (
            isinstance(downsample_factor, bool)
            or int(downsample_factor) != downsample_factor
            or int(downsample_factor) <= 0
        ):
            raise ValueError("downsample_factor must be a positive integer")
        self.downsample_factor = int(downsample_factor)
        self.gaussian_radius_pixels = float(gaussian_radius_pixels)
        if self.gaussian_radius_pixels <= 0:
            raise ValueError("gaussian_radius_pixels must be positive.")
        if self.root_dir.is_file():
            self.manifest_path = self.root_dir
            self.root_dir = self.manifest_path.parent
        else:
            self.manifest_path = self.root_dir / manifest_name
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}\n"
                "The angle-aware dataset requires a JSONL manifest; regenerate "
                "the data with generate.py."
            )

        split_filter = self._normalize_split_filter(split)
        self.records = self._load_manifest(split_filter)
        if not self.records:
            split_description = "all splits" if split is None else repr(split)
            raise ValueError(
                f"No manifest records found for {split_description} in "
                f"{self.manifest_path}."
            )

    @staticmethod
    def _normalize_split_filter(
        split: str | Sequence[str] | None,
    ) -> set[str] | None:
        if split is None:
            return None
        requested = {split} if isinstance(split, str) else set(split)
        allowed = {"train", "val", "test"}
        unknown = requested - allowed
        if unknown:
            raise ValueError(f"Unknown split(s): {sorted(unknown)}.")
        if not requested:
            raise ValueError("split cannot be an empty sequence.")
        return requested

    def _load_manifest(
        self,
        split_filter: set[str] | None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self._validate_record(record, line_number)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid manifest record at "
                        f"{self.manifest_path}:{line_number}: {exc}"
                    ) from exc

                record_id = record["id"]
                if record_id in seen_ids:
                    raise ValueError(
                        f"Duplicate manifest id {record_id!r} at line "
                        f"{line_number}."
                    )
                seen_ids.add(record_id)
                if split_filter is None or record["split"] in split_filter:
                    records.append(record)
        return records

    def _validate_record(
        self,
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        required = {
            "id",
            "image",
            "center",
            "pose",
            "symmetry",
            "valid_angle_bins",
            "angle_label_policy",
            "split",
            "shape",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError("id must be a non-empty string")
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {record['split']!r}")
        if not isinstance(record["shape"], str) or not record["shape"]:
            raise ValueError("shape must be a non-empty string")
        if record["angle_label_policy"] != ANGLE_LABEL_POLICY:
            raise ValueError(
                "angle_label_policy must be "
                f"{ANGLE_LABEL_POLICY!r}; regenerate legacy data with "
                "generate.py"
            )

        center = record["center"]
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not all(isinstance(value, (int, float)) for value in center)
            or not all(math.isfinite(float(value)) for value in center)
        ):
            raise ValueError("center must be two finite numbers [x, y]")
        if not isinstance(record["pose"], (int, float)) or not math.isfinite(
            float(record["pose"])
        ):
            raise ValueError("pose must be a finite number in degrees")
        if (
            not isinstance(record["symmetry"], (int, float))
            or not math.isfinite(float(record["symmetry"]))
            or float(record["symmetry"]) < 0.0
        ):
            raise ValueError("symmetry must be a finite non-negative number")
        if "camera_offset" in record and (
            not isinstance(record["camera_offset"], (int, float))
            or not math.isfinite(float(record["camera_offset"]))
            or float(record["camera_offset"]) <= 0.0
        ):
            raise ValueError("camera_offset must be a finite positive number")

        valid_bins = record["valid_angle_bins"]
        if (
            not isinstance(valid_bins, list)
            or not valid_bins
            or any(type(value) is not int for value in valid_bins)
            or any(not 0 <= value < NUM_ANGLE_BINS for value in valid_bins)
            or len(valid_bins) != len(set(valid_bins))
        ):
            raise ValueError(
                f"valid_angle_bins must contain unique integer bins in "
                f"[0, {NUM_ANGLE_BINS - 1}]"
            )
        symmetry = float(record["symmetry"])
        if symmetry == 0.0:
            expected_valid_count = NUM_ANGLE_BINS
        else:
            symmetry_bins = symmetry / ANGLE_STEP_DEG
            if (
                not symmetry_bins.is_integer()
                or symmetry_bins <= 0
                or NUM_ANGLE_BINS % int(symmetry_bins) != 0
            ):
                raise ValueError(
                    "symmetry must be zero or an integer multiple of the "
                    f"{ANGLE_STEP_DEG}-degree scanner step that divides 360"
                )
            expected_valid_count = NUM_ANGLE_BINS // int(symmetry_bins)
        if len(valid_bins) != expected_valid_count:
            raise ValueError(
                "valid_angle_bins does not contain every symmetry-equivalent "
                f"angle: expected {expected_valid_count}, got {len(valid_bins)}"
            )

        relative_path = Path(record["image"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("image must be a safe path relative to dataset root")
        image_path = self.root_dir / relative_path
        if not image_path.is_file():
            raise ValueError(
                f"image does not exist for line {line_number}: {image_path}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        image_path = self.root_dir / record["image"]
        try:
            with Image.open(image_path) as source:
                image = source.convert("L")
                original_width, original_height = image.size
                if self.downsample_factor > 1:
                    image = image.resize(
                        (
                            max(1, round(original_width / self.downsample_factor)),
                            max(1, round(original_height / self.downsample_factor)),
                        ),
                        resample=Image.Resampling.BILINEAR,
                    )
                transformed = self.transform(image) if self.transform else image
        except Exception as exc:
            raise RuntimeError(
                f"Error loading image for id {record['id']!r}: {exc}"
            ) from exc

        image_tensor = _coerce_image_tensor(transformed)
        height, width = image_tensor.shape[-2:]
        source_x, source_y = (float(value) for value in record["center"])
        center_x = rescale_pixel_coordinate(source_x, original_width, width)
        center_y = rescale_pixel_coordinate(source_y, original_height, height)
        center = torch.tensor([center_x, center_y], dtype=torch.float32)

        valid_angle_mask = torch.zeros(NUM_ANGLE_BINS, dtype=torch.bool)
        valid_angle_mask[record["valid_angle_bins"]] = True

        item: dict[str, Tensor | str] = {
            "image": image_tensor,
            "valid_angle_mask": valid_angle_mask,
            "center": center,
            "pose": torch.tensor(float(record["pose"]), dtype=torch.float32),
            "camera_offset": torch.tensor(
                float(record.get("camera_offset", LEGACY_CAMERA_OFFSET)),
                dtype=torch.float32,
            ),
            "shape": record["shape"],
            "split": record["split"],
            "downsample_factor": torch.tensor(
                self.downsample_factor,
                dtype=torch.float32,
            ),
        }
        if self.return_heatmap:
            item["hole_label"] = create_gaussian_heatmap(
                height,
                width,
                center,
                radius_pixels=(
                    self.gaussian_radius_pixels / self.downsample_factor
                ),
            )
        return item


# A clearer alias for new code while retaining the original public name.
HoleDataset = RectangleDataset


if __name__ == "__main__":
    dataset = RectangleDataset("data")
    first = dataset[0]
    print(
        f"Loaded {len(dataset)} samples; "
        f"image={tuple(first['image'].shape)}, "
        f"valid_angles={int(first['valid_angle_mask'].sum())}, "
        f"step={ANGLE_STEP_DEG} degrees"
    )
