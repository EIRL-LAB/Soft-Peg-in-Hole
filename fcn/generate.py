"""Generate the synthetic hole data set and its JSONL manifest.

Angles in this module use the image convention from the project plan:
positive angles are visual counter-clockwise rotations.  Samples cover
continuous poses, stratified into the same 36 ten-degree bins used by the
rotation scanner.  Corrections are assigned to the nearest scanner bin and
all symmetry-equivalent corrections are retained.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
import trimesh


# Units are meters. Block size is 8.5 cm x 8.5 cm x 1 cm.
BLOCK_SIZE = (0.085, 0.085, 0.01)

CIRCLE_DIAMETER = 0.02
SQUARE_SIDE = 0.02
HEXAGON_SIDE = 0.01
RIGHT_TRIANGLE_LEG = 0.02

DEFAULT_SAMPLES_PER_SHAPE = 1000
# Backward-compatible name used by the original generator.
SAMPLES_PER_SHAPE = DEFAULT_SAMPLES_PER_SHAPE
DEFAULT_SEED = 42
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_MANIFEST_NAME = "manifest.jsonl"

ANGLE_STEP_DEG = 10
NUM_ANGLE_BINS = 360 // ANGLE_STEP_DEG
ANGLE_DEGREES = tuple(range(0, 360, ANGLE_STEP_DEG))
ANGLE_LABEL_POLICY = "all_equivalent_angle_bins_v1"

# The value is the smallest rotation that maps the shape onto itself.  Zero
# denotes continuous rotational symmetry.
SHAPE_SYMMETRY_DEG = {
    "circle": 0,
    "square": 90,
    "hexagon": 60,
    "right_triangle": 360,
}
SHAPE_NAMES = tuple(SHAPE_SYMMETRY_DEG)

# Per-sample camera height above the table.
CAMERA_OFFSET_MIN = 0.20
CAMERA_OFFSET_MAX = 0.30
# Used only when reading/calling legacy code that did not record a height.
LEGACY_CAMERA_OFFSET = 0.35

CAMERA_VFOV = 58.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 3.0

Factory = Callable[[int, float], int]


def setup_simulation(gui: bool = False) -> Tuple[int, float, float, float]:
    """Initialize the PyBullet scene."""
    connection_mode = p.GUI if gui else p.DIRECT
    client = p.connect(connection_mode)
    p.resetSimulation(physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)

    plane_uid = p.loadURDF(
        "plane.urdf",
        useMaximalCoordinates=True,
        physicsClientId=client,
    )
    p.changeVisualShape(
        plane_uid,
        -1,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
        physicsClientId=client,
    )

    table_half_extents = [0.7, 0.3, 0.01]
    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=table_half_extents,
        physicsClientId=client,
    )
    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=table_half_extents,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
        physicsClientId=client,
    )
    table_top_z = 0.75
    p.createMultiBody(
        0.0,
        table_collision,
        table_visual,
        [0.0, 0.0, table_top_z - table_half_extents[2]],
        physicsClientId=client,
    )

    return client, table_half_extents[0], table_half_extents[1], table_top_z


def cutter_height() -> float:
    return BLOCK_SIZE[2] * 1.2


def rotate_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return points @ rotation.T


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def create_polygon_prism(points_2d: np.ndarray, height: float) -> trimesh.Trimesh:
    """Create a watertight convex prism cutter from centered 2D points."""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if len(points_2d) < 3:
        raise ValueError("A prism cutter requires at least three points.")
    if polygon_area(points_2d) < 0:
        points_2d = points_2d[::-1]

    half_height = height / 2.0
    bottom = np.column_stack(
        [points_2d, np.full(len(points_2d), -half_height, dtype=np.float64)]
    )
    top = np.column_stack(
        [points_2d, np.full(len(points_2d), half_height, dtype=np.float64)]
    )
    bottom_center_idx = len(points_2d) * 2
    top_center_idx = bottom_center_idx + 1
    vertices = np.vstack(
        [
            bottom,
            top,
            np.array([[0.0, 0.0, -half_height], [0.0, 0.0, half_height]]),
        ]
    )

    faces: list[list[int]] = []
    n = len(points_2d)
    for i in range(n):
        j = (i + 1) % n
        faces.append([bottom_center_idx, j, i])
        faces.append([top_center_idx, n + i, n + j])
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces),
        process=False,
    )


def apply_z_rotation(mesh: trimesh.Trimesh, angle_rad: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    transform = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
    mesh.apply_transform(transform)
    return mesh


def subtract_hole_from_block(cutter: trimesh.Trimesh) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=BLOCK_SIZE)
    try:
        return box.difference(cutter, engine="manifold")
    except Exception:
        return box.difference(cutter)


def create_block_with_circle_hole(client: int, theta_rad: float = 0.0) -> int:
    """Create a circle hole; ``theta_rad`` is retained as pose metadata."""
    cutter = trimesh.creation.cylinder(
        radius=CIRCLE_DIAMETER / 2.0,
        height=cutter_height(),
        sections=32,
    )
    # Rotating a circle is geometrically a no-op, but accepting theta keeps all
    # factories on the same explicit-pose interface.
    if theta_rad:
        cutter = apply_z_rotation(cutter, theta_rad)
    return create_multibody(client, subtract_hole_from_block(cutter))


def create_block_with_square_hole(client: int, theta_rad: float = 0.0) -> int:
    """Create a square whose pose zero has horizontal and vertical edges."""
    cutter = trimesh.creation.box(
        extents=[SQUARE_SIDE, SQUARE_SIDE, cutter_height()]
    )
    cutter = apply_z_rotation(cutter, theta_rad)
    return create_multibody(client, subtract_hole_from_block(cutter))


def create_regular_hexagon_points(theta_rad: float = 0.0) -> np.ndarray:
    """Return centered hexagon vertices for geometry checks and mesh creation."""
    points = np.array(
        [
            [
                HEXAGON_SIDE * math.cos(2.0 * math.pi * i / 6.0),
                HEXAGON_SIDE * math.sin(2.0 * math.pi * i / 6.0),
            ]
            for i in range(6)
        ],
        dtype=np.float64,
    )
    return rotate_points(points, theta_rad)


def create_block_with_hexagon_hole(client: int, theta_rad: float = 0.0) -> int:
    """Create a regular hexagon.

    At pose zero, vertices point left/right and the top and bottom edges are
    horizontal (parallel to the top and bottom of the image).
    """
    points = create_regular_hexagon_points(theta_rad)
    cutter = create_polygon_prism(points, cutter_height())
    return create_multibody(client, subtract_hole_from_block(cutter))


def create_block_with_right_triangle_hole(
    client: int,
    theta_rad: float = 0.0,
) -> int:
    """Create an isosceles right triangle in the agreed canonical pose.

    At pose zero, the right-angle vertex is at lower left and its two legs
    extend right and up in the rendered image.
    """
    points = np.array(
        [
            [0.0, 0.0],
            [RIGHT_TRIANGLE_LEG, 0.0],
            [0.0, RIGHT_TRIANGLE_LEG],
        ],
        dtype=np.float64,
    )
    points -= points.mean(axis=0)
    points = rotate_points(points, theta_rad)
    cutter = create_polygon_prism(points, cutter_height())
    return create_multibody(client, subtract_hole_from_block(cutter))


def create_multibody(client: int, mesh_result: trimesh.Trimesh) -> int:
    vertices = mesh_result.vertices
    indices = mesh_result.faces.flatten()
    collision = p.createCollisionShape(
        p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        physicsClientId=client,
    )
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=client,
    )
    block_id = p.createMultiBody(
        0.1,
        collision,
        visual,
        physicsClientId=client,
    )
    p.changeDynamics(
        block_id,
        -1,
        lateralFriction=0.9,
        physicsClientId=client,
    )
    return block_id


FACTORIES: dict[str, Factory] = {
    "circle": create_block_with_circle_hole,
    "square": create_block_with_square_hole,
    "hexagon": create_block_with_hexagon_hole,
    "right_triangle": create_block_with_right_triangle_hole,
}


def _validate_camera_offset(camera_offset: float) -> float:
    camera_offset = float(camera_offset)
    if not math.isfinite(camera_offset) or camera_offset <= 0.0:
        raise ValueError("camera_offset must be a finite positive value")
    if camera_offset - BLOCK_SIZE[2] <= CAMERA_NEAR:
        raise ValueError(
            "camera_offset must place the block top beyond CAMERA_NEAR; "
            f"got {camera_offset:.6f} m"
        )
    return camera_offset


def validate_camera_offset_range(
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    minimum = _validate_camera_offset(minimum)
    maximum = _validate_camera_offset(maximum)
    if minimum > maximum:
        raise ValueError(
            "camera_offset_min must be less than or equal to "
            "camera_offset_max"
        )
    return minimum, maximum


def sample_camera_offset(
    rng: random.Random,
    minimum: float = CAMERA_OFFSET_MIN,
    maximum: float = CAMERA_OFFSET_MAX,
) -> float:
    minimum, maximum = validate_camera_offset_range(minimum, maximum)
    return rng.uniform(minimum, maximum)


def random_pose(
    table_half_x: float,
    table_half_y: float,
    table_top_z: float,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    rng: random.Random | None = None,
    camera_offset: float = LEGACY_CAMERA_OFFSET,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Generate a random block position within the actual camera frustum."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    rng = rng or random
    half_z = BLOCK_SIZE[2] / 2

    camera_offset = _validate_camera_offset(camera_offset)
    # The top surface is closest to the camera and therefore has the narrowest
    # usable footprint in world coordinates.
    distance = camera_offset - BLOCK_SIZE[2]
    half_height = distance * math.tan(math.radians(CAMERA_VFOV / 2))
    # computeProjectionMatrixFOV interprets CAMERA_VFOV as vertical FOV.  Derive
    # the horizontal span from the same projection instead of using a second,
    # inconsistent hard-coded FOV.
    half_width = half_height * image_width / image_height

    # The complete 8.5 cm square must remain visible for every in-plane
    # rotation. Its half diagonal plus 5 mm is the required center margin.
    position_margin = math.hypot(BLOCK_SIZE[0], BLOCK_SIZE[1]) / 2.0 + 0.005
    limit_x = min(
        half_width - position_margin,
        table_half_x - position_margin,
    )
    limit_y = min(
        half_height - position_margin,
        table_half_y - position_margin,
    )
    if limit_x <= 0.0 or limit_y <= 0.0:
        raise ValueError(
            "camera_offset is too small to keep the complete block in view: "
            f"{camera_offset:.6f} m"
        )

    x = rng.uniform(-limit_x, limit_x)
    y = rng.uniform(-limit_y, limit_y)
    z = table_top_z + half_z

    return np.array([x, y, z]), p.getQuaternionFromEuler([0, 0, 0])


def settle(client: int, steps: int = 50) -> None:
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client)


def _camera_matrices(
    client: int,
    width: int,
    height: int,
    table_top_z: float,
    camera_offset: float = LEGACY_CAMERA_OFFSET,
) -> tuple[Sequence[float], Sequence[float]]:
    camera_offset = _validate_camera_offset(camera_offset)
    camera_pos = [0.0, 0.0, table_top_z + camera_offset]
    view_matrix = p.computeViewMatrix(
        camera_pos,
        [0, 0, table_top_z],
        [0, 1, 0],
        physicsClientId=client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        CAMERA_VFOV,
        width / height,
        CAMERA_NEAR,
        CAMERA_FAR,
        physicsClientId=client,
    )
    return view_matrix, proj_matrix


def get_hole_center_2d(
    client: int,
    block_id: int,
    width: int,
    height: int,
    table_top_z: float,
    camera_offset: float = LEGACY_CAMERA_OFFSET,
) -> tuple[float, float]:
    """Project the visible hole-opening center to image coordinates.

    ``getBasePositionAndOrientation`` returns the mesh/body origin, which is
    at the middle of the block.  The image, however, observes the hole on the
    block's top surface.  Because the camera is perspective, projecting the
    body center introduces a position-dependent bias.  Project the center of
    that top surface instead.
    """
    pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=client)
    view_matrix, proj_matrix = _camera_matrices(
        client,
        width,
        height,
        table_top_z,
        camera_offset,
    )

    view_matrix_np = np.asarray(view_matrix).reshape(4, 4).T
    proj_matrix_np = np.asarray(proj_matrix).reshape(4, 4).T
    top_surface_z = pos[2] + BLOCK_SIZE[2] / 2.0
    world_pos = np.array([pos[0], pos[1], top_surface_z, 1.0])
    clip_pos = proj_matrix_np @ view_matrix_np @ world_pos

    if abs(float(clip_pos[3])) <= np.finfo(np.float64).eps:
        return (width - 1) / 2.0, (height - 1) / 2.0

    ndc_x = float(clip_pos[0] / clip_pos[3])
    ndc_y = float(clip_pos[1] / clip_pos[3])
    img_x = (ndc_x + 1.0) * 0.5 * (width - 1)
    img_y = (1.0 - ndc_y) * 0.5 * (height - 1)
    return (
        float(np.clip(img_x, 0.0, width - 1.0)),
        float(np.clip(img_y, 0.0, height - 1.0)),
    )


def capture_binary_image(
    client: int,
    width: int,
    height: int,
    table_top_z: float,
    camera_offset: float = LEGACY_CAMERA_OFFSET,
) -> np.ndarray:
    """Capture a binary top-view image."""
    view_matrix, proj_matrix = _camera_matrices(
        client,
        width,
        height,
        table_top_z,
        camera_offset,
    )

    _, _, _, depth, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client,
    )

    depth_raw = np.asarray(depth, dtype=np.float32).reshape(height, width)
    depth_real = CAMERA_FAR * CAMERA_NEAR / (
        CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_raw
    )
    threshold_dist = camera_offset - (BLOCK_SIZE[2] * 0.5)

    binary_image = np.zeros((height, width), dtype=np.uint8)
    binary_image[depth_real < threshold_dist] = 255
    return binary_image


def normalize_pose_deg(pose_deg: float) -> float:
    """Normalize a finite continuous pose to ``[0, 360)`` degrees."""
    pose_deg = float(pose_deg)
    if not math.isfinite(pose_deg):
        raise ValueError(f"pose_deg must be finite, got {pose_deg}.")
    return pose_deg % 360.0


def get_valid_angle_bins(shape: str, pose_deg: float) -> list[int]:
    """Return every symmetry-equivalent scanner correction for a pose."""
    if shape not in SHAPE_SYMMETRY_DEG:
        raise ValueError(f"Unknown shape: {shape!r}.")
    pose_deg = normalize_pose_deg(pose_deg)
    if shape == "circle":
        valid = set(range(NUM_ANGLE_BINS))
    else:
        correction_deg = (-pose_deg) % 360.0
        base_bin = int(
            math.floor(correction_deg / ANGLE_STEP_DEG + 0.5)
        ) % NUM_ANGLE_BINS
        period_bins = SHAPE_SYMMETRY_DEG[shape] // ANGLE_STEP_DEG
        valid = {
            (base_bin + offset) % NUM_ANGLE_BINS
            for offset in range(0, NUM_ANGLE_BINS, period_bins)
        }

    return sorted(valid)


def split_counts(samples_per_shape: int) -> dict[str, int]:
    """Return exact deterministic 70/15/15 counts for one shape."""
    if samples_per_shape <= 0:
        raise ValueError("samples_per_shape must be positive.")
    train_count = int(samples_per_shape * 0.70)
    val_count = int(samples_per_shape * 0.15)
    return {
        "train": train_count,
        "val": val_count,
        "test": samples_per_shape - train_count - val_count,
    }


def _balanced_pose_bins(count: int, rng: random.Random) -> list[int]:
    """Distribute a split over all poses with at most one-sample imbalance."""
    bins = list(range(NUM_ANGLE_BINS))
    rng.shuffle(bins)
    return [bins[index % NUM_ANGLE_BINS] for index in range(count)]


def build_task_list(
    samples_per_shape: int = DEFAULT_SAMPLES_PER_SHAPE,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, object]]:
    """Build a shape- and pose-stratified task list.

    Each shape has exact 70/15/15 split totals.  Within every shape/split pair,
    all 36 pose bins differ in frequency by at most one.
    """
    rng = random.Random(seed)
    counts = split_counts(samples_per_shape)
    tasks: list[dict[str, object]] = []

    for shape in SHAPE_NAMES:
        shape_index = 0
        for split in ("train", "val", "test"):
            for pose_bin in _balanced_pose_bins(counts[split], rng):
                # Continuous coverage within a stratified scanner bin prevents
                # the network from seeing only zero-residual (exact 10-degree)
                # poses. random.uniform can include the upper endpoint, so use
                # random() to guarantee [-5, 5).
                jitter_deg = (
                    rng.random() - 0.5
                ) * ANGLE_STEP_DEG
                pose_deg = (
                    float(ANGLE_DEGREES[pose_bin]) + jitter_deg
                ) % 360.0
                tasks.append(
                    {
                        "id": f"{shape}_{shape_index:06d}",
                        "shape": shape,
                        "factory": FACTORIES[shape],
                        "pose_bin": pose_bin,
                        "pose": pose_deg,
                        "symmetry": float(SHAPE_SYMMETRY_DEG[shape]),
                        "valid_angle_bins": get_valid_angle_bins(shape, pose_deg),
                        "split": split,
                    }
                )
                shape_index += 1

    rng.shuffle(tasks)
    return tasks


def _write_manifest(records: Iterable[dict[str, object]], path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    temp_path.replace(path)


def generate_dataset(
    output_dir: str | Path,
    samples_per_shape: int = DEFAULT_SAMPLES_PER_SHAPE,
    seed: int = DEFAULT_SEED,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    camera_offset_min: float = CAMERA_OFFSET_MIN,
    camera_offset_max: float = CAMERA_OFFSET_MAX,
    gui: bool = False,
) -> Path:
    """Generate images and return the completed manifest path."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    camera_offset_min, camera_offset_max = validate_camera_offset_range(
        camera_offset_min,
        camera_offset_max,
    )
    output_dir = Path(output_dir).resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    tasks = build_task_list(samples_per_shape=samples_per_shape, seed=seed)
    client, table_half_x, table_half_y, table_top_z = setup_simulation(gui=gui)
    records: list[dict[str, object]] = []

    print(
        "Block size: "
        f"{BLOCK_SIZE[0] * 100:.1f}cm x "
        f"{BLOCK_SIZE[1] * 100:.1f}cm x "
        f"{BLOCK_SIZE[2] * 100:.1f}cm"
    )
    print(
        "Camera offset range: "
        f"[{camera_offset_min:.3f}, {camera_offset_max:.3f}]m"
    )
    print(f"Generating {len(tasks)} images in: {output_dir}")

    try:
        for index, task in enumerate(tasks):
            factory = task["factory"]
            if not callable(factory):
                raise TypeError(f"Invalid factory for task {task['id']}.")
            pose_deg = float(task["pose"])
            block_id = factory(client, math.radians(pose_deg))

            try:
                camera_offset = sample_camera_offset(
                    rng,
                    camera_offset_min,
                    camera_offset_max,
                )
                position, orientation = random_pose(
                    table_half_x,
                    table_half_y,
                    table_top_z,
                    camera_offset=camera_offset,
                    image_width=width,
                    image_height=height,
                    rng=rng,
                )
                p.resetBasePositionAndOrientation(
                    block_id,
                    position,
                    orientation,
                    physicsClientId=client,
                )

                binary_img = capture_binary_image(
                    client,
                    width,
                    height,
                    table_top_z,
                    camera_offset=camera_offset,
                )
                hole_x, hole_y = get_hole_center_2d(
                    client,
                    block_id,
                    width,
                    height,
                    table_top_z,
                    camera_offset=camera_offset,
                )

                relative_image_path = Path("images") / f"{task['id']}.png"
                Image.fromarray(binary_img).save(output_dir / relative_image_path)
                records.append(
                    {
                        "id": task["id"],
                        "image": relative_image_path.as_posix(),
                        "center": [hole_x, hole_y],
                        "pose": pose_deg,
                        "camera_offset": camera_offset,
                        "symmetry": task["symmetry"],
                        "valid_angle_bins": task["valid_angle_bins"],
                        "angle_label_policy": ANGLE_LABEL_POLICY,
                        "split": task["split"],
                        # Shape is metadata for labels and stratified evaluation
                        # only; it is never a model input.
                        "shape": task["shape"],
                    }
                )
            finally:
                p.removeBody(block_id, physicsClientId=client)

            if (index + 1) % 50 == 0 or index + 1 == len(tasks):
                print(f"  Generated {index + 1}/{len(tasks)}")
    finally:
        p.disconnect(physicsClientId=client)

    manifest_path = output_dir / manifest_name
    _write_manifest(records, manifest_path)
    print(f"Done. Manifest: {manifest_path}")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the rotation-equivariant FCN synthetic dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--samples-per-shape",
        type=int,
        default=DEFAULT_SAMPLES_PER_SHAPE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument(
        "--camera-offset-min",
        type=float,
        default=CAMERA_OFFSET_MIN,
    )
    parser.add_argument(
        "--camera-offset-max",
        type=float,
        default=CAMERA_OFFSET_MAX,
    )
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_dataset(
        output_dir=args.output_dir,
        samples_per_shape=args.samples_per_shape,
        seed=args.seed,
        width=args.width,
        height=args.height,
        manifest_name=args.manifest_name,
        camera_offset_min=args.camera_offset_min,
        camera_offset_max=args.camera_offset_max,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()
