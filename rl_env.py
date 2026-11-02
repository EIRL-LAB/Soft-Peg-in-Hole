from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import gymnasium as gym
import math3d as m3d
import numpy as np
import serial
from gymnasium import spaces

import black_ratio
from A_tactile_sensor0 import get_force_data
from A_ur5_tilt import Robot


logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("peg_in_hole_env.log", encoding="utf-8"),
        ],
    )


_RL_ROI_CONFIRM_LOCK = threading.Lock()
_RL_ROI_CONFIRM_DONE = False
_RL_ROI_CONFIRMED = False
_RL_ROI_REFERENCE_PATH = "debug_rl_confirmed_hole_roi.png"


@dataclass
class EnvConfig:
    """轴孔装配强化学习环境的运行参数。

    坐标约定：
    - 工具坐标系 +Y 为向下方向。
    - 工具坐标系 X/Z 为水平搜索平面。
    - 观测中的位置使用任务坐标 [x_offset, y_offset, z_offset]；
      真实 TCP 坐标只写入 info 供日志和调试使用。
    """

    robot_ip: str = "192.168.1.101"
    rotation_y_deg: float = -45.0
    reset_pose_path: str = "rl_reset_pose_left.json"
    require_reset_pose: bool = True
    task_frame_path: str = "rl_task_frame_left.json"
    require_task_frame: bool = True

    sensor0_port: str = "/dev/ttyACM0"
    sensor1_port: str = "/dev/ttyACM1"
    sensor_baudrate: int = 921600
    sensor_timeout_s: float = 0.1
    sensor_filter_samples: int = 5
    force_offset_frames: int = 20
    require_sensors: bool = True

    camera_device_id: str = "943222072873"
    require_camera: bool = True
    confirm_roi_on_reset: bool = True

    history_frames: int = 4
    max_steps_per_episode: int = 100

    # reset 初始点以任务原点上方的目标位置为圆心，默认在 0~5mm 半径圆内随机。
    random_reset_min_radius_m: float = 0.0
    random_reset_radius_m: float = 0.005
    # 任务 X/Z 位置归一化范围；越界终止同时使用坐标半径和视觉 ROI。
    lateral_limit_m: float = 0.030
    # 兼容旧配置；当前观测 Y 使用任务坐标偏移，不再使用该字段。
    vertical_norm_range_m: float = 0.060

    max_lateral_step_m: float = 0.005
    max_tilt_step_deg: float = 1.0
    max_abs_tilt_deg: float = 4.0
    action_ema_alpha: float = 1.0

    # reset 目标相对任务原点沿任务 -Y 方向的高度。
    reset_height_above_task_origin_m: float = 0.050
    safe_lift_m: float = 0.050
    surface_force_threshold: float = 5.0
    reset_force_descend_target_y: float = 0.30
    reset_force_descend_speed: float = 0.01
    reset_force_descend_acc: float = 0.1

    weak_down_step_m: float = 0.00200
    # 弱下压使用当前净力中的 f0/f1 X/Y 最大绝对值。
    weak_down_force_threshold: Optional[float] = 10.0
    # 弱下压前要求两个 Z 侧向力误差都低于该阈值，避免未对准时持续下压。
    weak_down_side_force_threshold: Optional[float] = 10.0
    center_progress_reward_scale: float = 0.05
    center_progress_norm_m: float = 0.005
    center_progress_reward_clip: float = 0.05
    # 兼容旧配置；当前中心进展奖励已全程启用，不再按 ROI 黑色占比门控。
    center_progress_enable_until_roi: float = 95.0

    move_acc: float = 0.03
    move_vel: float = 0.01
    tilt_acc: float = 0.03
    tilt_vel: float = 0.03
    settle_s: float = 0.05

    # 工具坐标系下 TCP 到工件末端的偏置，必须按真实夹具测量值调整。
    tip_offset_tool: Tuple[float, float, float] = (0.02, 0.15, 0.26)

    force_norm: float = 30.0
    # 黑色区域占比低于该阈值时判定为越界。
    black_ratio_oob_max: float = 20.0
    black_ratio_success: float = 95.0
    contact_xy_ema_alpha: float = 0.7
    # 兼容旧配置；峰值现按当前回合的完整历史累计，不再使用滑动窗口。
    contact_xy_peak_window_steps: int = 10
    black_delta_reward_clip: float = 0.2
    roi_to_full_frame_bonus: float = 0.0
    force_safety_limit: float = 25.0
    insertion_full_black_ratio_success_min: float = 95.0
    insertion_success_depth_m: float = 0.040
    no_camera_success_require_contact_xy_max: float = 5.0
    success_reward: float = 1.0
    failure_reward: float = -0.5
    camera_tilt_deadband_px: float = 25.0
    camera_zero_direction_deg: float = 270.0
    camera_mirror_x: bool = True
    camera_invert_tilt_direction: bool = True


class SensorReader:
    """单路触觉传感器后台读取线程。"""

    def __init__(self, ser: serial.Serial, samples: int = 5, name: str = "sensor"):
        self.ser = ser
        self.samples = max(1, int(samples))
        self.name = name

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._last_valid = np.zeros(3, dtype=np.float32)
        self._log = logging.getLogger(f"{__name__}.{name}")

        self._thread = threading.Thread(
            target=self._run,
            name=f"SensorReader-{name}",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        buf = []
        while not self._stop_evt.is_set():
            if self._pause_evt.is_set():
                buf = []
                time.sleep(0.01)
                continue

            if not (self.ser and self.ser.is_open):
                time.sleep(0.01)
                continue

            try:
                data = get_force_data(self.ser)
                if data is None:
                    continue
                buf.append(data)
                if len(buf) < self.samples:
                    continue

                filtered = np.median(buf, axis=0).astype(np.float32)
                buf = []
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(filtered)
            except Exception as exc:
                self._log.warning("传感器读取失败，跳过当前帧: %s", exc)
                buf = []
                time.sleep(0.01)

    def get_latest(self) -> np.ndarray:
        try:
            self._last_valid = self._queue.get_nowait()
        except queue.Empty:
            pass
        return self._last_valid.copy()

    def pause(self):
        self._pause_evt.set()
        time.sleep(0.4)

    def resume(self):
        self._pause_evt.clear()

    def stop(self):
        self._stop_evt.set()
        self._pause_evt.clear()
        self._thread.join(timeout=2.0)


class PegInHoleEnv(gym.Env):
    """带水平搜索、倾斜探索和控力下降的轴孔装配环境。

    动作空间，已归一化到 [-1, 1]：
        [dx, dz, dtheta_x, dtheta_z]

    单帧观测，已归一化到 [-1, 1]：
        位置 3D、倾角 2D、触觉状态 4D、当前力 6D、上一步动作 4D、上一步力 6D

    网络最终输入：
        连续 4 帧单帧观测拼接。
    """

    FRAME_DIM = 25

    def __init__(self, config: Optional[EnvConfig] = None, **overrides):
        super().__init__()

        self.cfg = config if config is not None else EnvConfig()
        for key, value in overrides.items():
            if not hasattr(self.cfg, key):
                raise TypeError(f"Unknown EnvConfig option: {key}")
            setattr(self.cfg, key, value)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.FRAME_DIM * self.cfg.history_frames,),
            dtype=np.float32,
        )

        # 兼容现有训练回调，记录 reset 阶段最大随机偏移量。
        self.curriculum_offset = self.cfg.random_reset_radius_m

        logger.info("正在初始化机械臂")
        self.robot_left = Robot(
            ip=self.cfg.robot_ip,
            rotation_y_deg=self.cfg.rotation_y_deg,
        )
        self.task_frame = self._load_task_frame()
        self.task_frame_path = self.task_frame["path"]
        self.task_frame_loaded = bool(self.task_frame["loaded"])
        self.task_frame_json_origin = np.asarray(
            self.task_frame["origin_world"],
            dtype=np.float32,
        )
        self.task_frame_axes = np.asarray(self.task_frame["axes_world"], dtype=np.float32)
        self.task_frame_origin = self.task_frame_json_origin.copy()
        self.task_origin_source = (
            "task_frame_json" if self.task_frame_loaded else "current_pose_fallback"
        )
        self.reset_pose = self._load_reset_pose()
        self.reset_pose_path = self.reset_pose["path"]
        self.reset_pose_loaded = bool(self.reset_pose["loaded"])
        self.reset_pose_matrix = np.asarray(
            self.reset_pose["world_from_tcp"],
            dtype=np.float64,
        )
        self.reset_pose_position = np.asarray(
            self.reset_pose["position_world"],
            dtype=np.float32,
        )
        self.reset_target_matrix = self._build_reset_target_matrix()
        self.reset_target_position = self.reset_target_matrix[:3, 3].astype(np.float32)
        self.reset_target_task_pos = self._world_to_task_delta(
            self.reset_target_position - self.task_frame_origin
        )
        logger.info(
            "已加载锁定坐标系: %s effective_origin=%s origin_source=%s "
            "json_origin=%s reset_pose=%s reset_json_position=%s "
            "reset_target_position=%s reset_target_task_pos=%s x=%s y=%s z=%s",
            self.task_frame_path,
            self.task_frame_origin.tolist(),
            self.task_origin_source,
            self.task_frame_json_origin.tolist(),
            self.reset_pose_path,
            self.reset_pose_position.tolist(),
            self.reset_target_position.tolist(),
            self.reset_target_task_pos.tolist(),
            self.task_frame_axes[0].tolist(),
            self.task_frame_axes[1].tolist(),
            self.task_frame_axes[2].tolist(),
        )

        self.sensor0: Optional[SensorReader] = None
        self.sensor1: Optional[SensorReader] = None
        self._open_sensors()

        self.camera = None
        self._open_camera()

        self.current_step = 0
        self.prev_black_ratio = 0.0
        self.current_black_ratio = 0.0
        self.current_black_ratio_mode = "full_frame_fallback"
        self.current_roi_black_ratio = 0.0
        self.prev_roi_black_ratio = 0.0
        self.current_full_black_ratio = 0.0
        self.prev_black_ratio_mode = "full_frame_fallback"
        self.current_roi_ready = False
        self.current_roi_failed = False
        self.current_roi_source = ""
        self.target_tilt_dir = np.zeros(2, dtype=np.float32)
        self.target_tilt_valid = False
        self.last_black_reward = 0.0
        self.last_black_reward_applied = False
        self.last_black_delta_pp = 0.0
        self.last_black_delta_reward = 0.0
        self.prev_task_radius = 0.0
        self.last_center_progress_m = 0.0
        self.last_center_progress_reward = 0.0
        self.last_center_progress_reward_applied = False
        self.last_roi_to_full_frame_bonus_applied = False
        self.last_position_out_of_bounds = False
        self.last_vision_out_of_bounds = False
        self.last_force_safety_triggered = False
        self.last_force_max_error = 0.0
        self.last_force_abs_errors = np.zeros(6, dtype=np.float32)
        self.depth_from_surface = 0.0
        self.last_descend_task_y_delta = 0.0
        self.contact_xy_ema = 0.0
        self.prev_contact_xy_ema = 0.0
        self.contact_xy_drop_ema = 0.0
        self.contact_xy_raw = 0.0
        self.prev_contact_xy_raw = 0.0
        self.contact_xy_raw_drop = 0.0
        self.contact_xy_historical_peak = 0.0
        # 兼容现有训练日志；该字段现在表示当前回合的历史峰值。
        self.contact_xy_recent_peak = 0.0
        self.contact_xy_peak_drop = 0.0

        self.f0_offset = np.zeros(3, dtype=np.float32)
        self.f1_offset = np.zeros(3, dtype=np.float32)
        self.pre_descend_force_baseline = np.zeros(6, dtype=np.float32)

        self.task_pos = np.zeros(3, dtype=np.float32)
        self.tilt_deg = np.zeros(2, dtype=np.float32)
        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_force = np.zeros(6, dtype=np.float32)

        self._history: deque[np.ndarray] = deque(maxlen=self.cfg.history_frames)
        self._recent_successes: deque[int] = deque(maxlen=20)

    def _load_task_frame(self) -> dict:
        path = os.path.abspath(self.cfg.task_frame_path)
        if not os.path.exists(path):
            if self.cfg.require_task_frame:
                raise RuntimeError(f"未找到锁定坐标系文件: {path}")
            logger.warning("未找到锁定坐标系文件，退回当前 world 位姿作为任务原点: %s", path)
            pose = self.robot_left.get_pose()
            matrix = np.asarray(pose.matrix, dtype=np.float32)
            return {
                "path": path,
                "loaded": False,
                "origin_world": [float(pose.pos.x), float(pose.pos.y), float(pose.pos.z)],
                "axes_world": np.eye(3, dtype=np.float32).tolist(),
                "world_from_task_frame": matrix.tolist(),
            }

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        origin = data.get("origin_world")
        axes = data.get("axes_world")
        matrix = data.get("world_from_task_frame")
        if axes is None:
            x_axis = data.get("x_axis_world")
            y_axis = data.get("y_axis_world")
            z_axis = data.get("z_axis_world")
            if x_axis is not None and y_axis is not None and z_axis is not None:
                axes = [x_axis, y_axis, z_axis]
        if origin is None or axes is None:
            raise RuntimeError(f"锁定坐标系文件格式错误: {path}")

        axes_arr = np.asarray(axes, dtype=np.float32)
        if axes_arr.shape != (3, 3):
            raise RuntimeError(f"锁定坐标系 axes_world 形状错误: {path}")
        if matrix is None:
            matrix_arr = np.eye(4, dtype=np.float32)
            matrix_arr[:3, 3] = np.asarray(origin, dtype=np.float32)
            matrix_arr[:3, :3] = axes_arr.T
        else:
            matrix_arr = np.asarray(matrix, dtype=np.float32)
            if matrix_arr.shape != (4, 4):
                raise RuntimeError(f"锁定坐标系 world_from_task_frame 形状错误: {path}")

        return {
            "path": path,
            "loaded": True,
            "origin_world": origin,
            "axes_world": axes_arr.tolist(),
            "world_from_task_frame": matrix_arr.tolist(),
        }

    def _load_reset_pose(self) -> dict:
        path = os.path.abspath(self.cfg.reset_pose_path)
        if not os.path.exists(path):
            if self.cfg.require_reset_pose:
                raise RuntimeError(f"未找到 reset 初始位姿文件: {path}")
            logger.warning("未找到 reset 初始位姿文件，退回当前 TCP 位姿: %s", path)
            pose = self.robot_left.get_pose()
            matrix = np.asarray(pose.matrix, dtype=np.float32)
            return {
                "path": path,
                "loaded": False,
                "position_world": matrix[:3, 3].astype(float).tolist(),
                "world_from_tcp": matrix.tolist(),
            }

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        matrix = np.asarray(data.get("world_from_tcp"), dtype=np.float64)
        position = np.asarray(data.get("position_world"), dtype=np.float64)
        if matrix.shape != (4, 4):
            raise RuntimeError(f"reset 初始位姿 world_from_tcp 形状错误: {path}")
        if position.shape != (3,):
            raise RuntimeError(f"reset 初始位姿 position_world 形状错误: {path}")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(position)):
            raise RuntimeError(f"reset 初始位姿包含非有限值: {path}")
        if not np.allclose(matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
            raise RuntimeError(f"reset 初始位姿齐次矩阵最后一行错误: {path}")
        if not np.allclose(matrix[:3, 3], position, atol=1e-6):
            raise RuntimeError(f"reset 初始位姿 position_world 与矩阵平移不一致: {path}")

        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise RuntimeError(f"reset 初始位姿旋转矩阵不正交: {path}")
        det = float(np.linalg.det(rotation))
        if abs(det - 1.0) > 1e-3:
            raise RuntimeError(f"reset 初始位姿旋转矩阵 det={det:.6f}，应接近 1: {path}")

        file_robot_ip = data.get("robot_ip")
        if file_robot_ip is not None and str(file_robot_ip) != str(self.cfg.robot_ip):
            logger.warning(
                "reset pose robot_ip=%s 与当前 cfg.robot_ip=%s 不一致",
                file_robot_ip,
                self.cfg.robot_ip,
            )
        file_rotation_y_deg = data.get("rotation_y_deg")
        if file_rotation_y_deg is not None:
            try:
                if (
                    abs(float(file_rotation_y_deg) - float(self.cfg.rotation_y_deg))
                    > 1e-6
                ):
                    logger.warning(
                        "reset pose rotation_y_deg=%s 与当前 cfg.rotation_y_deg=%s 不一致",
                        file_rotation_y_deg,
                        self.cfg.rotation_y_deg,
                    )
            except (TypeError, ValueError):
                logger.warning("reset pose rotation_y_deg 无法解析: %s", file_rotation_y_deg)

        return {
            "path": path,
            "loaded": True,
            "position_world": position.astype(np.float32).tolist(),
            "world_from_tcp": matrix.astype(np.float32).tolist(),
        }

    def _open_sensors(self):
        try:
            ser0 = serial.Serial(
                self.cfg.sensor0_port,
                baudrate=self.cfg.sensor_baudrate,
                timeout=self.cfg.sensor_timeout_s,
            )
            ser1 = serial.Serial(
                self.cfg.sensor1_port,
                baudrate=self.cfg.sensor_baudrate,
                timeout=self.cfg.sensor_timeout_s,
            )
            self.sensor0 = SensorReader(
                ser0,
                samples=self.cfg.sensor_filter_samples,
                name="sensor0",
            )
            self.sensor1 = SensorReader(
                ser1,
                samples=self.cfg.sensor_filter_samples,
                name="sensor1",
            )
            logger.info("触觉传感器后台读取线程已启动")
        except Exception as exc:
            if self.cfg.require_sensors:
                raise RuntimeError("触觉传感器初始化失败") from exc
            logger.warning("触觉传感器初始化失败，将使用零力代替: %s", exc)
            self.sensor0 = None
            self.sensor1 = None

    def _open_camera(self):
        if not self.cfg.require_camera:
            logger.info("require_camera=False，跳过相机初始化")
            self.camera = None
            return

        try:
            self.camera = black_ratio.D435i(device_id=self.cfg.camera_device_id)
            logger.info("D435i 相机已初始化")
        except Exception as exc:
            if self.cfg.require_camera:
                raise RuntimeError("相机初始化失败") from exc
            logger.warning("相机初始化失败，黑色占比将返回 0: %s", exc)
            self.camera = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 每个回合重新开始时，清空时序状态和上一回合残留动作。
        self.current_step = 0
        self.prev_black_ratio = 0.0
        self.current_black_ratio = 0.0
        self.current_black_ratio_mode = "full_frame_fallback"
        self.current_roi_black_ratio = 0.0
        self.prev_roi_black_ratio = 0.0
        self.current_full_black_ratio = 0.0
        self.prev_black_ratio_mode = "full_frame_fallback"
        self.current_roi_ready = False
        self.current_roi_failed = False
        self.current_roi_source = ""
        self.target_tilt_dir[:] = 0.0
        self.target_tilt_valid = False
        self.last_black_reward = 0.0
        self.last_black_reward_applied = False
        self.last_black_delta_pp = 0.0
        self.last_black_delta_reward = 0.0
        self.prev_task_radius = 0.0
        self.last_center_progress_m = 0.0
        self.last_center_progress_reward = 0.0
        self.last_center_progress_reward_applied = False
        self.last_roi_to_full_frame_bonus_applied = False
        self.last_position_out_of_bounds = False
        self.last_vision_out_of_bounds = False
        self.last_force_safety_triggered = False
        self.last_force_max_error = 0.0
        self.last_force_abs_errors[:] = 0.0
        self.depth_from_surface = 0.0
        self.last_descend_task_y_delta = 0.0
        self.contact_xy_ema = 0.0
        self.prev_contact_xy_ema = 0.0
        self.contact_xy_drop_ema = 0.0
        self.contact_xy_raw = 0.0
        self.prev_contact_xy_raw = 0.0
        self.contact_xy_raw_drop = 0.0
        self.contact_xy_historical_peak = 0.0
        self.contact_xy_recent_peak = 0.0
        self.contact_xy_peak_drop = 0.0
        self.smoothed_action[:] = 0.0
        self.last_action[:] = 0.0
        self.last_force[:] = 0.0
        self.task_pos[:] = 0.0
        self.pre_descend_force_baseline[:] = 0.0

        self._safe_lift_before_reset()
        self._move_to_reset_pose()
        self.tilt_deg[:] = 0.0

        # 在 reset 初始位姿周围按配置半径随机落点，模拟粗定位误差。
        offset_x, offset_z = self._sample_reset_offset()
        self._move_reset_offset_xz(offset_x, offset_z)
        logger.info(
            "reset: 随机水平偏移 request x=%+.2f mm z=%+.2f mm; "
            "actual task pos x=%+.2f mm y=%+.2f mm z=%+.2f mm",
            offset_x * 1000,
            offset_z * 1000,
            self.task_pos[0] * 1000,
            self.task_pos[1] * 1000,
            self.task_pos[2] * 1000,
        )

        self._collect_force_offset(self.cfg.force_offset_frames)
        self.pre_descend_force_baseline = self._read_net_force()
        logger.info(
            "reset: 初始下降前净力基线 sensor0=%s sensor1=%s",
            self.pre_descend_force_baseline[:3],
            self.pre_descend_force_baseline[3:],
        )
        surface_force = self._descend_until_surface()
        self._update_task_pos_from_current_pose()
        self.depth_from_surface = 0.0
        self.last_descend_task_y_delta = 0.0

        # reset 结束时记录 offset 后净力和初始黑色占比，作为第一步奖励的基准。
        force = surface_force if surface_force is not None else self._read_net_force()
        self.last_force = force.copy()
        self._initialize_contact_xy_ema(self._contact_xy_force(force))

        self._get_camera_black_ratio_and_tilt_direction(
            confirm_roi=bool(self.cfg.confirm_roi_on_reset)
        )
        self.prev_black_ratio = self.current_black_ratio
        self.prev_black_ratio_mode = self.current_black_ratio_mode
        self.prev_roi_black_ratio = self.current_roi_black_ratio
        self.prev_task_radius = self._lateral_radius()

        frame, raw = self._make_frame(
            current_force=force,
            previous_action=np.zeros(4, dtype=np.float32),
            previous_force=force,
        )
        self._history.clear()
        for _ in range(self.cfg.history_frames):
            self._history.append(frame.copy())

        info = self._build_info(raw, is_success=False, init_reset=True)
        return self._stack_history(), info

    def step(self, action):
        self.current_step += 1

        # SAC 输出的是归一化动作，这里先限幅，再映射为真实位移和角度。
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw_action.shape != self.action_space.shape:
            raise ValueError(
                f"动作维度错误: expected {self.action_space.shape}, got {raw_action.shape}"
            )
        raw_action = np.clip(raw_action, -1.0, 1.0)
        prev_smoothed_action = self.smoothed_action.copy()

        alpha = float(np.clip(self.cfg.action_ema_alpha, 0.0, 1.0))
        self.smoothed_action[:4] = (
            (1.0 - alpha) * self.smoothed_action[:4] + alpha * raw_action[:4]
        ).astype(np.float32)

        dx = float(self.smoothed_action[0] * self.cfg.max_lateral_step_m)
        dz = float(self.smoothed_action[1] * self.cfg.max_lateral_step_m)
        dtheta_x = float(self.smoothed_action[2] * self.cfg.max_tilt_step_deg)
        dtheta_z = float(self.smoothed_action[3] * self.cfg.max_tilt_step_deg)

        prev_force = self.last_force.copy()
        prev_task_radius = float(self.prev_task_radius)
        self._reset_step_reward_components()
        executed_action_norm = self.smoothed_action.copy()
        max_tilt_step = max(float(self.cfg.max_tilt_step_deg), 1e-8)

        actual_dtheta_x, actual_dtheta_z = self._apply_tilt_only(
            dtheta_x,
            dtheta_z,
        )
        executed_action_norm[2] = actual_dtheta_x / max_tilt_step
        executed_action_norm[3] = actual_dtheta_z / max_tilt_step
        self._apply_lateral_only(dx, dz)

        force_before_down = self._read_net_force()
        current_task_radius = self._lateral_radius()
        center_progress_m = float(prev_task_radius - current_task_radius)
        norm_m = max(float(self.cfg.center_progress_norm_m), 1e-9)
        clip_value = abs(float(self.cfg.center_progress_reward_clip))
        center_reward = float(
            np.clip(
                center_progress_m / norm_m
                * float(self.cfg.center_progress_reward_scale),
                -clip_value,
                clip_value,
            )
        )
        center_reward_applied = True
        self.last_center_progress_m = center_progress_m
        self.last_center_progress_reward = center_reward
        self.last_center_progress_reward_applied = bool(center_reward_applied)

        descended, current_force, actual_descend_y = self._force_gated_descend(
            force_before_down
        )

        self.last_descend_task_y_delta = float(actual_descend_y)
        self.depth_from_surface += self.last_descend_task_y_delta
        self.last_force = current_force.copy()
        self.last_action = np.clip(executed_action_norm, -1.0, 1.0).astype(np.float32)

        contact_xy_force = self._contact_xy_force(current_force)
        self._update_contact_xy_ema(contact_xy_force)
        self.last_force_abs_errors = self._force_abs_errors(current_force)
        self.last_force_max_error = self._force_max_error(current_force)
        self.last_force_safety_triggered = (
            self.last_force_max_error > float(self.cfg.force_safety_limit)
        )

        curr_black_ratio, target_tilt_dir, target_tilt_valid = (
            self._get_camera_black_ratio_and_tilt_direction()
        )
        self.current_black_ratio = curr_black_ratio
        self.target_tilt_dir = target_tilt_dir
        self.target_tilt_valid = target_tilt_valid

        frame, raw = self._make_frame(
            current_force=current_force,
            previous_action=self.last_action,
            previous_force=prev_force,
        )
        self._history.append(frame)

        out_of_bounds = self._is_out_of_bounds()

        terminated = False
        truncated = False
        is_success = False

        black_ratio_mode_switched = (
            str(self.current_black_ratio_mode) != str(self.prev_black_ratio_mode)
        )
        if black_ratio_mode_switched:
            black_delta_pp = 0.0
            black_reward = 0.0
        else:
            black_delta_pp = float(curr_black_ratio - self.prev_black_ratio)
            black_reward = float(
                np.clip(
                    black_delta_pp / 100.0,
                    -float(self.cfg.black_delta_reward_clip),
                    float(self.cfg.black_delta_reward_clip),
                )
            )
        reward = float(black_reward + center_reward)
        self.last_black_delta_pp = float(black_delta_pp)
        self.last_black_delta_reward = float(black_reward)
        self.last_black_reward = float(black_reward)
        self.last_roi_to_full_frame_bonus_applied = False
        self.last_black_reward_applied = not black_ratio_mode_switched

        # 终止优先级：成功 > 明确越界失败 > 步数耗尽。
        if self._success_ready():
            reward = float(self.cfg.success_reward)
            terminated = True
            is_success = True
            logger.info(
                "装配成功: active_black_ratio=%.2f%% roi_black_ratio=%.2f%% "
                "full_black_ratio=%.2f%% success_threshold=>%.2f%% "
                "contact_xy_force=%.2f depth=%.3fmm force_max_error=%.2f",
                self.current_black_ratio,
                self.current_roi_black_ratio,
                self.current_full_black_ratio,
                self.cfg.insertion_full_black_ratio_success_min,
                contact_xy_force,
                self.depth_from_surface * 1000.0,
                self.last_force_max_error,
            )
        elif out_of_bounds or self.last_force_safety_triggered:
            reward = float(self.cfg.failure_reward)
            truncated = True
            self.smoothed_action = prev_smoothed_action
            logger.warning(
                "失败终止: position_oob=%s vision_oob=%s force_safety_oob=%s "
                "roi_black_ratio=%.2f%% roi_threshold=%.2f%% "
                "x=%+.2fmm z=%+.2fmm radius=%.2fmm lateral_limit=%.2fmm "
                "force_max_error=%.2f force_safety_limit=%.2f",
                self.last_position_out_of_bounds,
                self.last_vision_out_of_bounds,
                self.last_force_safety_triggered,
                self.current_roi_black_ratio,
                self.cfg.black_ratio_oob_max,
                self.task_pos[0] * 1000,
                self.task_pos[2] * 1000,
                self._lateral_radius() * 1000,
                self.cfg.lateral_limit_m * 1000,
                self.last_force_max_error,
                self.cfg.force_safety_limit,
            )
        elif self.current_step >= self.cfg.max_steps_per_episode:
            truncated = True
            logger.info(
                "回合达到最大步数 %d，截断结束",
                self.cfg.max_steps_per_episode,
            )

        debug_sep = "─" * 60
        force_abs_errors = self.last_force_abs_errors
        side_force_errors_text = np.array2string(
            force_abs_errors[[2, 5]],
            precision=2,
            separator=", ",
            suppress_small=True,
        )
        weak_down_side_threshold_text = (
            " disabled"
            if self.cfg.weak_down_side_force_threshold is None
            else f"{float(self.cfg.weak_down_side_force_threshold):7.2f}"
        )
        print(
            f"{debug_sep}\n"
            f"步骤调试｜第 {self.current_step} 步\n\n"
            "【视觉】\n"
            f"  当前黑色占比: {curr_black_ratio:7.2f}%    "
            f"模式: {self.current_black_ratio_mode}    "
            f"模式切换: {int(black_ratio_mode_switched)}\n"
            f"  ROI 黑色占比 : {self.current_roi_black_ratio:7.2f}%    "
            f"全图黑色占比: {self.current_full_black_ratio:7.2f}%\n"
            f"  成功全图阈值: {self.cfg.insertion_full_black_ratio_success_min:7.2f}%\n"
            f"  相比上一步  : {black_delta_pp:+7.2f} 个百分点\n"
            f"  黑色变化奖励: {self.last_black_delta_reward:+.4f}\n"
            f"  ROI 转全图奖励触发: {int(self.last_roi_to_full_frame_bonus_applied)}\n"
            f"  ROI 转全图奖励值  : {self.cfg.roi_to_full_frame_bonus:+.4f}\n\n"
            "【触觉 / 下降】\n"
            f"  X/Y接触力   : {contact_xy_force:7.2f}\n"
            f"  X/Y接触力EMA: {self.contact_xy_ema:7.2f}\n"
            f"  本步下降    : {int(descended)}\n"
            f"  本步实际下降: {self.last_descend_task_y_delta * 1000.0:7.2f} mm\n"
            f"  累计下探深度: {self.depth_from_surface * 1000.0:7.2f} mm / "
            f"{self.cfg.insertion_success_depth_m * 1000.0:.2f} mm\n\n"
            "【安全】\n"
            f"  force误差 f0: x={force_abs_errors[0]:7.2f} "
            f"y={force_abs_errors[1]:7.2f} z={force_abs_errors[2]:7.2f}\n"
            f"  force误差 f1: x={force_abs_errors[3]:7.2f} "
            f"y={force_abs_errors[4]:7.2f} z={force_abs_errors[5]:7.2f}\n"
            f"  六轴最大力误差: {self.last_force_max_error:7.2f}\n"
            f"  Z侧向力误差  : {side_force_errors_text}\n"
            f"  弱下压Z阈值  : {weak_down_side_threshold_text}\n"
            f"  三轴力安全触发: {int(self.last_force_safety_triggered)}\n"
            f"  任务半径      : {self._lateral_radius() * 1000.0:7.2f} mm\n"
            f"  越界          : {int(out_of_bounds)}\n\n"
            "【奖励】\n"
            f"  中心进展奖励: {self.last_center_progress_reward:+.4f} "
            f"(进展={self.last_center_progress_m * 1000.0:+.2f} mm, "
            f"启用={int(self.last_center_progress_reward_applied)})\n"
            f"  总奖励: {float(reward):+.4f}"
        )

        if terminated or truncated:
            self._record_episode_result(is_success)

        info = self._build_info(
            raw,
            is_success=is_success,
            out_of_bounds=out_of_bounds,
            descended=descended,
            contact_xy_force=contact_xy_force,
        )
        self.prev_black_ratio = curr_black_ratio
        self.prev_black_ratio_mode = self.current_black_ratio_mode
        self.prev_roi_black_ratio = self.current_roi_black_ratio
        self.prev_task_radius = self._lateral_radius()
        return self._stack_history(), float(reward), terminated, truncated, info

    def close(self):
        for label, sensor in (("sensor0", self.sensor0), ("sensor1", self.sensor1)):
            if sensor is None:
                continue
            try:
                sensor.stop()
                if sensor.ser and sensor.ser.is_open:
                    sensor.ser.close()
            except Exception as exc:
                logger.warning("%s 关闭失败: %s", label, exc)

        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception as exc:
                logger.warning("相机关闭失败: %s", exc)

        try:
            self.robot_left.close()
        except Exception as exc:
            logger.warning("机械臂关闭失败: %s", exc)

    def _sample_reset_offset(self) -> Tuple[float, float]:
        min_radius = max(0.0, float(self.cfg.random_reset_min_radius_m))
        max_radius = max(min_radius, float(self.cfg.random_reset_radius_m))
        radius = float(np.sqrt(self.np_random.uniform(min_radius**2, max_radius**2)))
        angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        return radius * float(np.cos(angle)), radius * float(np.sin(angle))

    def _safe_lift_before_reset(self):
        """在 reset 早期沿锁定任务 -Y 方向上抬，避免横向回中心时刮碰。"""
        distance = max(0.0, float(self.cfg.safe_lift_m))
        if distance <= 0.0:
            return

        pose_before = self.robot_left.get_pose()
        self._move_task_xyz(0.0, -distance, 0.0)
        time.sleep(self.cfg.settle_s)
        pose_after = self.robot_left.get_pose()
        self._update_task_pos_from_pose(pose_after)
        tool_delta = pose_before.inverse * pose_after
        logger.info(
            "reset safe lift: requested task -Y %.3f mm; "
            "actual tool delta dx=%+.3f mm dy=%+.3f mm dz=%+.3f mm; "
            "locked task pos x=%+.3f mm y=%+.3f mm z=%+.3f mm",
            distance * 1000.0,
            float(tool_delta.pos.x) * 1000.0,
            float(tool_delta.pos.y) * 1000.0,
            float(tool_delta.pos.z) * 1000.0,
            float(self.task_pos[0]) * 1000.0,
            float(self.task_pos[1]) * 1000.0,
            float(self.task_pos[2]) * 1000.0,
        )

    def _build_reset_target_matrix(self) -> np.ndarray:
        """使用 JSON 姿态和任务原点上方的计算位置构造 reset 目标位姿。"""
        target_matrix = np.asarray(self.reset_pose_matrix, dtype=np.float64).copy()
        reset_height = max(
            0.0,
            float(self.cfg.reset_height_above_task_origin_m),
        )
        target_task_pos = np.array([0.0, -reset_height, 0.0], dtype=np.float32)
        target_world_pos = (
            self.task_frame_origin + self._task_to_world_delta(target_task_pos)
        )
        target_matrix[:3, 3] = np.asarray(target_world_pos, dtype=np.float64)
        return target_matrix

    def _move_to_reset_pose(self):
        """移动到任务原点上方的计算位置，姿态沿用 reset JSON。"""
        pose_before = self.robot_left.get_pose()
        self.reset_target_matrix = self._build_reset_target_matrix()
        self.reset_target_position = self.reset_target_matrix[:3, 3].astype(np.float32)
        self.reset_target_task_pos = self._world_to_task_delta(
            self.reset_target_position - self.task_frame_origin
        )
        target = m3d.Transform(self.reset_target_matrix)
        logger.info(
            "reset: 移动到任务原点上方 %.3f mm，"
            "target world x=%.6f y=%.6f z=%.6f，姿态来源=%s",
            float(self.cfg.reset_height_above_task_origin_m) * 1000.0,
            float(target.pos.x),
            float(target.pos.y),
            float(target.pos.z),
            self.reset_pose_path,
        )
        self.robot_left.set_pose(target, acc=self.cfg.move_acc, vel=self.cfg.move_vel)
        time.sleep(self.cfg.settle_s)

        pose_after = self.robot_left.get_pose()
        self._update_task_pos_from_pose(pose_after)
        actual_tool_delta = pose_before.inverse * pose_after
        actual_matrix = np.asarray(pose_after.matrix, dtype=np.float64)
        target_matrix = self.reset_target_matrix
        position_error_m = float(
            np.linalg.norm(actual_matrix[:3, 3] - target_matrix[:3, 3])
        )
        rotation_delta = target_matrix[:3, :3].T @ actual_matrix[:3, :3]
        cos_angle = float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
        orientation_error_deg = float(np.degrees(np.arccos(cos_angle)))
        logger.info(
            "reset pose actual tool delta: dx=%+.3f mm dy=%+.3f mm dz=%+.3f mm; "
            "pose_error position=%.3f mm orientation=%.4f deg; "
            "locked task pos: x=%+.3f mm y=%+.3f mm z=%+.3f mm",
            float(actual_tool_delta.pos.x) * 1000.0,
            float(actual_tool_delta.pos.y) * 1000.0,
            float(actual_tool_delta.pos.z) * 1000.0,
            position_error_m * 1000.0,
            orientation_error_deg,
            float(self.task_pos[0]) * 1000.0,
            float(self.task_pos[1]) * 1000.0,
            float(self.task_pos[2]) * 1000.0,
        )

    def _move_reset_offset_xz(self, offset_x: float, offset_z: float):
        """在锁定任务 X/Z 平面执行 reset 偏移；任务 Y 不参与该段运动。"""
        pose_before = self.robot_left.get_pose()
        self._move_task_xyz(offset_x, 0.0, offset_z)
        time.sleep(self.cfg.settle_s)
        pose_after = self.robot_left.get_pose()
        self._update_task_pos_from_pose(pose_after)
        tool_delta = pose_before.inverse * pose_after
        logger.info(
            "reset offset actual tool delta: dx=%+.3f mm dy=%+.3f mm dz=%+.3f mm; "
            "locked task pos: x=%+.3f mm y=%+.3f mm z=%+.3f mm",
            float(tool_delta.pos.x) * 1000.0,
            float(tool_delta.pos.y) * 1000.0,
            float(tool_delta.pos.z) * 1000.0,
            float(self.task_pos[0]) * 1000.0,
            float(self.task_pos[1]) * 1000.0,
            float(self.task_pos[2]) * 1000.0,
        )

    def _collect_force_offset(self, n_frames: int):
        logger.info("正在采集触觉传感器去皮偏置，共 %d 帧", n_frames)
        self._pause_sensors()
        buf0 = []
        buf1 = []
        try:
            ser0 = self.sensor0.ser if self.sensor0 else None
            ser1 = self.sensor1.ser if self.sensor1 else None
            max_attempts = max(10, n_frames * 6)

            attempts = 0
            while (len(buf0) < n_frames or len(buf1) < n_frames) and attempts < max_attempts:
                attempts += 1
                if ser0 is not None and len(buf0) < n_frames:
                    data = get_force_data(ser0)
                    if data is not None:
                        buf0.append(data)
                if ser1 is not None and len(buf1) < n_frames:
                    data = get_force_data(ser1)
                    if data is not None:
                        buf1.append(data)

            if len(buf0) < n_frames and self.cfg.require_sensors:
                raise RuntimeError(f"sensor0 去皮采样不足，仅采到 {len(buf0)} / {n_frames} 帧")
            if len(buf0) >= 5:
                self.f0_offset = np.mean(buf0, axis=0).astype(np.float32)
            else:
                self.f0_offset = np.zeros(3, dtype=np.float32)
                logger.warning("sensor0 去皮采样不足，仅采到 %d 帧", len(buf0))

            if len(buf1) < n_frames and self.cfg.require_sensors:
                raise RuntimeError(f"sensor1 去皮采样不足，仅采到 {len(buf1)} / {n_frames} 帧")
            if len(buf1) >= 5:
                self.f1_offset = np.mean(buf1, axis=0).astype(np.float32)
            else:
                self.f1_offset = np.zeros(3, dtype=np.float32)
                logger.warning("sensor1 去皮采样不足，仅采到 %d 帧", len(buf1))
        finally:
            self._resume_sensors()

        logger.info("去皮偏置 sensor0=%s sensor1=%s", self.f0_offset, self.f1_offset)

    def _pause_sensors(self):
        if self.sensor0:
            self.sensor0.pause()
        if self.sensor1:
            self.sensor1.pause()

    def _resume_sensors(self):
        if self.sensor0:
            self.sensor0.resume()
        if self.sensor1:
            self.sensor1.resume()

    def _read_net_force(self) -> np.ndarray:
        """读取后台线程缓存的最新净力。"""
        raw0 = self.sensor0.get_latest() if self.sensor0 else np.zeros(3, dtype=np.float32)
        raw1 = self.sensor1.get_latest() if self.sensor1 else np.zeros(3, dtype=np.float32)
        f0 = raw0.astype(np.float32) - self.f0_offset
        f1 = raw1.astype(np.float32) - self.f1_offset
        return np.concatenate([f0, f1]).astype(np.float32)

    def _read_sensor_direct_or_raise(
        self,
        sensor: Optional[SensorReader],
        label: str,
    ) -> np.ndarray:
        if sensor is None or sensor.ser is None:
            if self.cfg.require_sensors:
                raise RuntimeError(f"{label} 触觉传感器不可用")
            return np.zeros(3, dtype=np.float32)

        data = get_force_data(sensor.ser)
        if data is None:
            if self.cfg.require_sensors:
                raise RuntimeError(f"{label} 触觉传感器读数失败")
            logger.warning("%s 触觉传感器读数失败，使用零力代替", label)
            return np.zeros(3, dtype=np.float32)
        return np.asarray(data, dtype=np.float32)

    def _read_net_force_direct(self) -> np.ndarray:
        """绕过后台缓存直接读串口，适用于控力下降等需要独占串口的阶段。"""
        raw0 = self._read_sensor_direct_or_raise(self.sensor0, "sensor0")
        raw1 = self._read_sensor_direct_or_raise(self.sensor1, "sensor1")
        return np.concatenate([raw0 - self.f0_offset, raw1 - self.f1_offset]).astype(np.float32)

    def _descend_until_surface(self) -> Optional[np.ndarray]:
        logger.info(
            "reset: force-monitored Y descend, surface threshold=%.2f",
            self.cfg.surface_force_threshold,
        )
        _, final_force, _ = self._force_monitored_descend_y(
            distance_m=self.cfg.reset_force_descend_target_y,
            threshold=self.cfg.surface_force_threshold,
            speed=self.cfg.reset_force_descend_speed,
            acc=self.cfg.reset_force_descend_acc,
            label="reset surface descend",
            timeout_buffer_s=3.0,
        )
        return final_force

    def _force_gated_descend(
        self, current_force: np.ndarray
    ) -> Tuple[bool, np.ndarray, float]:
        """Advance Y only while X/Y contact force and Z-side errors are low."""
        if self.cfg.weak_down_force_threshold is None:
            return False, current_force, 0.0

        threshold = float(self.cfg.weak_down_force_threshold)
        contact_xy_force = self._contact_xy_force(current_force)
        if contact_xy_force >= threshold:
            logger.info(
                "weak descend skipped: contact_xy_force=%.2f >= threshold=%.2f",
                contact_xy_force,
                threshold,
            )
            return False, current_force, 0.0

        side_threshold = self.cfg.weak_down_side_force_threshold
        if side_threshold is not None:
            side_threshold = float(side_threshold)
            side_force_abs_errors = self._side_force_abs_errors(current_force)
            side_force_max_error = (
                float(np.max(side_force_abs_errors))
                if side_force_abs_errors.size
                else 0.0
            )
            if side_force_max_error >= side_threshold:
                logger.info(
                    "weak descend skipped: z_side_force_max_error=%.2f >= "
                    "threshold=%.2f errors=%s",
                    side_force_max_error,
                    side_threshold,
                    np.array2string(
                        side_force_abs_errors,
                        precision=2,
                        separator=", ",
                        suppress_small=True,
                    ),
                )
                return False, current_force, 0.0

        return self._force_monitored_descend_y(
            distance_m=float(self.cfg.weak_down_step_m),
            threshold=threshold,
            speed=self.cfg.move_vel,
            acc=self.cfg.move_acc,
            label="weak descend",
            timeout_buffer_s=max(0.2, self.cfg.settle_s),
        )

    def _force_monitored_descend_y(
        self,
        distance_m: float,
        threshold: Optional[float],
        speed: float,
        acc: float,
        label: str,
        timeout_buffer_s: float,
    ) -> Tuple[bool, np.ndarray, float]:
        """Move down in locked task +Y and stop when X/Y contact force reaches threshold."""
        threshold_value = None if threshold is None else float(threshold)
        distance_m = float(distance_m)
        speed = float(speed)
        acc = float(acc)
        timeout_buffer_s = max(0.0, float(timeout_buffer_s))

        pose_before = self.robot_left.get_pose()
        final_force = self._read_net_force()
        command_started = False
        stopped_by_force = False

        self._pause_sensors()
        try:
            final_force = self._read_net_force_direct()
            contact_xy_force = self._contact_xy_force(final_force)
            if threshold_value is not None and contact_xy_force > threshold_value:
                logger.info(
                    "%s skipped: contact_xy_force=%.2f > threshold=%.2f",
                    label,
                    contact_xy_force,
                    threshold_value,
                )
                return False, final_force, 0.0

            self._move_task_xyz_async(0.0, distance_m, 0.0, acc=acc, vel=speed)
            command_started = True

            expected_duration = 1.0
            if abs(speed) > 1e-9:
                expected_duration = abs(distance_m) / abs(speed) + timeout_buffer_s

            start_time = time.monotonic()
            while True:
                final_force = self._read_net_force_direct()
                contact_xy_force = self._contact_xy_force(final_force)

                if threshold_value is not None and contact_xy_force > threshold_value:
                    self.robot_left.stop()
                    stopped_by_force = True
                    logger.info(
                        "%s stopped: contact_xy_force=%.2f > threshold=%.2f",
                        label,
                        contact_xy_force,
                        threshold_value,
                    )
                    break

                if time.monotonic() - start_time >= expected_duration:
                    break

                time.sleep(0.01)
        except Exception:
            if command_started:
                try:
                    self.robot_left.stop()
                    logger.error("%s aborted: force read failed, robot stop issued", label)
                except Exception as stop_exc:
                    logger.error("%s abort stop failed: %s", label, stop_exc)
            raise
        finally:
            self._resume_sensors()

        if command_started:
            time.sleep(self.cfg.settle_s)

        pose_after = self.robot_left.get_pose()
        tool_delta = pose_before.inverse * pose_after
        actual_dx = float(tool_delta.pos.x)
        actual_dy = float(tool_delta.pos.y)
        actual_dz = float(tool_delta.pos.z)
        world_before = np.array(
            [float(pose_before.pos.x), float(pose_before.pos.y), float(pose_before.pos.z)],
            dtype=np.float32,
        )
        world_after = np.array(
            [float(pose_after.pos.x), float(pose_after.pos.y), float(pose_after.pos.z)],
            dtype=np.float32,
        )
        task_delta = self._world_to_task_delta(world_after - world_before)
        actual_task_y_delta = max(0.0, float(task_delta[1]))
        self._update_task_pos_from_pose(pose_after)
        logger.info(
            "%s finished: dx=%+.3f mm dy=%+.3f mm dz=%+.3f mm "
            "task_y_delta=%+.3f mm stopped_by_force=%s contact_xy_force=%.2f",
            label,
            actual_dx * 1000.0,
            actual_dy * 1000.0,
            actual_dz * 1000.0,
            actual_task_y_delta * 1000.0,
            stopped_by_force,
            self._contact_xy_force(final_force),
        )

        return actual_task_y_delta > 1e-7, final_force, actual_task_y_delta

    def _apply_lateral_only(self, dx: float, dz: float):
        """只执行策略给出的锁定任务 X/Z 水平位移。"""
        if dx or dz:
            self._move_task_xyz(dx, 0.0, dz)
            self._update_task_pos_from_current_pose()
            time.sleep(self.cfg.settle_s)

    def _apply_tilt_only(self, dtheta_x: float, dtheta_z: float):
        """只执行绕锁定任务 X/Z 轴的倾斜微调。"""
        dtheta_x = self._clip_tilt_delta(index=0, delta_deg=dtheta_x)
        dtheta_z = self._clip_tilt_delta(index=1, delta_deg=dtheta_z)

        tilt_delta_norm = float(np.hypot(dtheta_x, dtheta_z))
        if tilt_delta_norm > 1e-6:
            axis_task = np.array(
                [
                float(dtheta_x / tilt_delta_norm),
                0.0,
                float(dtheta_z / tilt_delta_norm),
                ],
                dtype=np.float32,
            )
            axis_world = self._task_to_world_delta(axis_task)
            self._rotate_around_tip(
                axis_vec=axis_world,
                angle_deg=tilt_delta_norm,
                frame="world",
            )
            self.tilt_deg[0] += dtheta_x
            self.tilt_deg[1] += dtheta_z
            time.sleep(self.cfg.settle_s)

        return float(dtheta_x), float(dtheta_z)

    def _clip_tilt_delta(self, index: int, delta_deg: float) -> float:
        """根据累计倾角上限裁剪本次倾斜增量。"""
        current = float(self.tilt_deg[index])
        limit = float(self.cfg.max_abs_tilt_deg)
        target = float(np.clip(current + delta_deg, -limit, limit))
        return target - current

    def _rotate_around_tip(self, axis_vec, angle_deg: float, frame: str = "world"):
        """绕工件末端执行 RCM 旋转，降低倾斜时末端扫动。"""
        self.robot_left.rotate_around_tip_about_axis_vector(
            tip_offset_tool=list(self.cfg.tip_offset_tool),
            axis_vec=np.asarray(axis_vec, dtype=float),
            angle_deg=angle_deg,
            frame=frame,
            acc=self.cfg.tilt_acc,
            vel=self.cfg.tilt_vel,
        )

    def _contact_xy_force(self, force6: np.ndarray) -> float:
        """Return max abs current X/Y force used by weak-descend gating."""
        force6 = np.asarray(force6, dtype=np.float32)
        return float(
            max(
                abs(float(force6[0])),
                abs(float(force6[1])),
                abs(float(force6[3])),
                abs(float(force6[4])),
            )
        )

    def _task_to_world_delta(self, task_vec: np.ndarray) -> np.ndarray:
        task_vec = np.asarray(task_vec, dtype=np.float32)
        return (self.task_frame_axes.T @ task_vec).astype(np.float32)

    def _task_axis_world(self, index: int) -> np.ndarray:
        return np.asarray(self.task_frame_axes[index], dtype=np.float32)

    def _world_to_task_delta(self, world_vec: np.ndarray) -> np.ndarray:
        world_vec = np.asarray(world_vec, dtype=np.float32)
        return (self.task_frame_axes @ world_vec).astype(np.float32)

    def _pose_to_task_pos(self, pose) -> np.ndarray:
        world_pos = np.array(
            [float(pose.pos.x), float(pose.pos.y), float(pose.pos.z)],
            dtype=np.float32,
        )
        return self._world_to_task_delta(world_pos - self.task_frame_origin)

    def _update_task_pos_from_pose(self, pose):
        self.task_pos[:] = self._pose_to_task_pos(pose)

    def _update_task_pos_from_current_pose(self):
        self._update_task_pos_from_pose(self.robot_left.get_pose())

    def _move_task_xyz(self, x: float, y: float, z: float):
        world_delta = self._task_to_world_delta(np.array([x, y, z], dtype=np.float32))
        self.robot_left.move_world_xyz(
            x=float(world_delta[0]),
            y=float(world_delta[1]),
            z=float(world_delta[2]),
            acc=self.cfg.move_acc,
            vel=self.cfg.move_vel,
        )

    def _move_task_xyz_async(self, x: float, y: float, z: float, acc: float, vel: float):
        world_delta = self._task_to_world_delta(np.array([x, y, z], dtype=np.float32))
        self.robot_left.move_world_xyz_async(
            x=float(world_delta[0]),
            y=float(world_delta[1]),
            z=float(world_delta[2]),
            acc=acc,
            vel=vel,
        )

    def _is_out_of_bounds(self) -> bool:
        """Return true when task position or visual ROI leaves the safe search area."""
        self.last_position_out_of_bounds = (
            self._lateral_radius() > float(self.cfg.lateral_limit_m)
        )
        self.last_vision_out_of_bounds = (
            self.current_black_ratio_mode == "sam3_roi"
            and float(self.current_roi_black_ratio) < float(self.cfg.black_ratio_oob_max)
        )
        return bool(self.last_position_out_of_bounds or self.last_vision_out_of_bounds)

    def _lateral_radius(self) -> float:
        """返回当前任务坐标 X/Z 半径。"""
        return float(np.hypot(self.task_pos[0], self.task_pos[2]))

    def _reset_step_reward_components(self):
        self.last_black_reward = 0.0
        self.last_black_reward_applied = False
        self.last_black_delta_pp = 0.0
        self.last_black_delta_reward = 0.0
        self.last_center_progress_m = 0.0
        self.last_center_progress_reward = 0.0
        self.last_center_progress_reward_applied = False
        self.last_roi_to_full_frame_bonus_applied = False
        self.last_force_safety_triggered = False
        self.last_force_max_error = 0.0
        self.last_force_abs_errors[:] = 0.0

    def _initialize_contact_xy_ema(self, contact_xy_force: float):
        contact_xy_force = float(contact_xy_force)
        self.prev_contact_xy_raw = contact_xy_force
        self.contact_xy_raw = contact_xy_force
        self.contact_xy_raw_drop = 0.0
        self.prev_contact_xy_ema = contact_xy_force
        self.contact_xy_ema = contact_xy_force
        self.contact_xy_drop_ema = 0.0
        self.contact_xy_historical_peak = self.contact_xy_ema
        self.contact_xy_recent_peak = self.contact_xy_historical_peak
        self.contact_xy_peak_drop = 0.0

    def _update_contact_xy_ema(self, contact_xy_force: float):
        alpha = float(np.clip(self.cfg.contact_xy_ema_alpha, 0.0, 1.0))
        previous = float(self.contact_xy_ema)
        current = float(contact_xy_force)
        self.prev_contact_xy_raw = float(self.contact_xy_raw)
        self.contact_xy_raw = current
        self.contact_xy_raw_drop = float(
            self.prev_contact_xy_raw - self.contact_xy_raw
        )
        self.prev_contact_xy_ema = previous
        self.contact_xy_ema = float(alpha * current + (1.0 - alpha) * previous)
        self.contact_xy_drop_ema = float(previous - self.contact_xy_ema)
        self.contact_xy_historical_peak = max(
            float(self.contact_xy_historical_peak),
            self.contact_xy_ema,
        )
        self.contact_xy_recent_peak = self.contact_xy_historical_peak
        self.contact_xy_peak_drop = float(
            self.contact_xy_historical_peak - self.contact_xy_ema
        )

    def _side_force_abs_errors(self, force6: np.ndarray) -> np.ndarray:
        """Return abs Z errors for [f0_z, f1_z] against reset baseline."""
        force_errors = self._force_abs_errors(force6)
        z_indices = np.array([2, 5], dtype=int)
        return force_errors[z_indices].astype(np.float32)

    def _force_abs_errors(self, force6: np.ndarray) -> np.ndarray:
        """Return abs XYZ errors for both sensors against reset baseline."""
        force6 = np.asarray(force6, dtype=np.float32)
        baseline = np.asarray(self.pre_descend_force_baseline, dtype=np.float32)
        return np.abs(force6 - baseline).astype(np.float32)

    def _force_max_error(self, force6: np.ndarray) -> float:
        """返回两路传感器全部 XYZ 轴分量中的最大绝对力误差。"""
        force_errors = self._force_abs_errors(force6)
        if force_errors.size == 0:
            return 0.0
        return float(np.max(force_errors))

    def _success_ready(self) -> bool:
        if not bool(self.cfg.require_camera):
            return False

        return self.current_full_black_ratio > float(
            self.cfg.insertion_full_black_ratio_success_min
        )

    def _make_frame(
        self,
        current_force: np.ndarray,
        previous_action: np.ndarray,
        previous_force: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """构造单帧 25 维归一化观测，并保留 info 调试数据。"""
        pose = self.robot_left.get_pose()
        raw_tcp = np.array(
            [float(pose.pos.x), float(pose.pos.y), float(pose.pos.z)],
            dtype=np.float32,
        )
        self._update_task_pos_from_pose(pose)

        norm_task_y = np.clip(
            self.task_pos[1]
            / max(float(self.cfg.insertion_success_depth_m), 1e-9),
            0.0,
            1.0,
        )
        norm_pos = np.array(
            [
                np.clip(self.task_pos[0] / self.cfg.lateral_limit_m, -1.0, 1.0),
                norm_task_y,
                np.clip(self.task_pos[2] / self.cfg.lateral_limit_m, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        norm_tilt = np.clip(
            self.tilt_deg / self.cfg.max_abs_tilt_deg,
            -1.0,
            1.0,
        ).astype(np.float32)
        norm_force = np.clip(current_force / self.cfg.force_norm, -1.0, 1.0).astype(np.float32)
        norm_prev_force = np.clip(previous_force / self.cfg.force_norm, -1.0, 1.0).astype(np.float32)
        norm_prev_action = np.clip(previous_action, -1.0, 1.0).astype(np.float32)
        norm_tactile_state = np.array(
            [
                np.clip(self.contact_xy_raw / self.cfg.force_norm, 0.0, 1.0),
                np.clip(self.contact_xy_ema / self.cfg.force_norm, 0.0, 1.0),
                np.clip(
                    self.contact_xy_drop_ema / self.cfg.force_norm,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    self.contact_xy_peak_drop / self.cfg.force_norm,
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

        frame = np.concatenate(
            [
                norm_pos,
                norm_tilt,
                norm_tactile_state,
                norm_force,
                norm_prev_action,
                norm_prev_force,
            ]
        ).astype(np.float32)

        raw = {
            "tcp": raw_tcp,
            "task_pos": self.task_pos.copy(),
            "tilt_deg": self.tilt_deg.copy(),
            "force": current_force.copy(),
            "prev_force": previous_force.copy(),
            "prev_action": previous_action.copy(),
            "task_frame_origin": self.task_frame_origin.copy(),
            "task_frame_axes": self.task_frame_axes.copy(),
        }
        return frame, raw

    def _stack_history(self) -> np.ndarray:
        """将最近 4 帧观测拼接为网络输入。"""
        if not self._history:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        while len(self._history) < self.cfg.history_frames:
            self._history.appendleft(self._history[0].copy())
        return np.concatenate(list(self._history)).astype(np.float32)

    def _update_vision_debug_from_black_ratio(self):
        debug = black_ratio.get_last_ratio_debug()
        self.current_roi_ready = bool(debug.get("roi_ready", False))
        self.current_roi_failed = bool(debug.get("roi_failed", False))
        self.current_roi_source = str(debug.get("roi_source", ""))

    def _make_roi_reference_image(self, frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        roi_mask = np.asarray(roi_mask).astype(bool)
        if roi_mask.shape[:2] != frame.shape[:2]:
            roi_mask = cv2.resize(
                roi_mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        overlay = frame.copy()
        overlay[roi_mask] = (0, 0, 255)
        display = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        contours, _ = cv2.findContours(
            roi_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(display, contours, -1, (0, 255, 255), 2)
        cv2.putText(
            display,
            "SAM3 ROI: y=apply  r=resplit  q=full-frame fallback",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def _ensure_confirmed_hole_roi(self) -> bool:
        """Segment ROI once per process and ask the operator to confirm it."""
        global _RL_ROI_CONFIRM_DONE, _RL_ROI_CONFIRMED

        if self.camera is None:
            return False

        with _RL_ROI_CONFIRM_LOCK:
            if _RL_ROI_CONFIRM_DONE:
                debug = black_ratio.get_last_ratio_debug()
                if not _RL_ROI_CONFIRMED or debug.get("roi_ready", False):
                    return _RL_ROI_CONFIRMED
                _RL_ROI_CONFIRM_DONE = False
                _RL_ROI_CONFIRMED = False

            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    logger.warning("ROI 分割跳过：未获取到相机图像")
                    _RL_ROI_CONFIRM_DONE = True
                    _RL_ROI_CONFIRMED = False
                    return False

                roi = black_ratio.segment_hole_roi_from_frame(frame, force=True)
                if roi is not None:
                    reference = self._make_roi_reference_image(frame, roi)
                    cv2.imwrite(_RL_ROI_REFERENCE_PATH, reference)
                    logger.info(
                        "SAM3 ROI 参考图已保存: %s，请打开图片确认后输入 y/r/q",
                        _RL_ROI_REFERENCE_PATH,
                    )
                    prompt = (
                        f"SAM3 ROI 参考图: {_RL_ROI_REFERENCE_PATH}\n"
                        "打开图片确认后输入 y 应用该 ROI，r 重新分割，q 放弃 ROI 使用全图 fallback: "
                    )
                else:
                    prompt = "SAM3 ROI 分割失败。输入 r 重试，q 放弃 ROI 使用全图 fallback: "

                try:
                    choice = input(prompt).strip().lower()
                except EOFError:
                    choice = "y" if roi is not None else "q"

                if choice in ("y", "yes"):
                    if roi is None:
                        print("[SAM3 ROI] 当前没有可应用的 ROI，请重新分割或放弃。")
                        continue
                    _RL_ROI_CONFIRM_DONE = True
                    _RL_ROI_CONFIRMED = True
                    logger.info("SAM3 ROI 已确认，本进程后续 episode 将复用该 ROI")
                    return True

                if choice in ("r", "retry", "resplit"):
                    logger.info("按用户要求重新执行 SAM3 ROI 分割")
                    continue

                if choice in ("q", "quit", "n", "no"):
                    black_ratio.reset_hole_roi_cache()
                    _RL_ROI_CONFIRM_DONE = True
                    _RL_ROI_CONFIRMED = False
                    logger.warning("用户放弃 SAM3 ROI，本进程将使用全图 fallback")
                    return False

                print("请输入 y、r 或 q。")

    def confirm_hole_roi_before_training(self, require_confirmed: bool = True) -> bool:
        """在训练开始前确认孔洞 ROI，并在当前进程后续 episode 复用该 ROI。"""
        logger.info("训练前开始确认 SAM3 孔洞 ROI")
        confirmed = self._ensure_confirmed_hole_roi()
        if not confirmed:
            message = "训练前 ROI 未确认，停止启动强化学习训练"
            if require_confirmed:
                raise RuntimeError(message)
            logger.warning(message)
            return False

        self._get_camera_black_ratio_and_tilt_direction(confirm_roi=False)
        logger.info(
            "训练前 ROI 已确认并缓存: mode=%s roi_black_ratio=%.2f%% full_black_ratio=%.2f%%",
            self.current_black_ratio_mode,
            self.current_roi_black_ratio,
            self.current_full_black_ratio,
        )
        return True

    def _get_camera_black_ratio_and_tilt_direction(
        self,
        confirm_roi: bool = False,
    ) -> Tuple[float, np.ndarray, bool]:
        """Read one camera frame and update active/ROI/full vision state."""
        if self.camera is None:
            self.current_black_ratio = 0.0
            self.current_black_ratio_mode = "full_frame_fallback"
            self.current_roi_black_ratio = 0.0
            self.current_full_black_ratio = 0.0
            self.current_roi_ready = False
            self.current_roi_failed = False
            self.current_roi_source = ""
            self.target_tilt_dir[:] = 0.0
            self.target_tilt_valid = False
            return 0.0, np.zeros(2, dtype=np.float32), False
        if confirm_roi:
            self._ensure_confirmed_hole_roi()
        frame = self.camera.get_frame()
        if frame is None:
            logger.warning("未获取到相机图像，黑色占比置为 0")
            self.current_black_ratio = 0.0
            self.current_black_ratio_mode = "full_frame_fallback"
            self.current_roi_black_ratio = 0.0
            self.current_full_black_ratio = 0.0
            self._update_vision_debug_from_black_ratio()
            self.target_tilt_dir[:] = 0.0
            self.target_tilt_valid = False
            return 0.0, np.zeros(2, dtype=np.float32), False

        ratio, mask, mode, roi_ratio, full_ratio = black_ratio.select_active_black_mask(frame)
        self.current_black_ratio = float(ratio)
        self.current_black_ratio_mode = str(mode)
        self.current_roi_black_ratio = float(roi_ratio)
        self.current_full_black_ratio = float(full_ratio)
        self._update_vision_debug_from_black_ratio()

        target_tilt_dir = np.zeros(2, dtype=np.float32)
        target_tilt_valid = False
        if mode == "sam3_roi":
            center = black_ratio.find_largest_black_region_center(mask)
            target_tilt_dir, target_tilt_valid = black_ratio.target_tilt_direction_from_black_center(
                frame.shape,
                center,
                deadband_px=self.cfg.camera_tilt_deadband_px,
                camera_zero_direction_deg=self.cfg.camera_zero_direction_deg,
                mirror_x=self.cfg.camera_mirror_x,
                invert_direction=self.cfg.camera_invert_tilt_direction,
            )

        self.target_tilt_dir = np.asarray(target_tilt_dir, dtype=np.float32)
        self.target_tilt_valid = bool(target_tilt_valid)
        return (
            self.current_black_ratio,
            self.target_tilt_dir.copy(),
            self.target_tilt_valid,
        )

    def _get_black_ratio(self) -> float:
        """Backward-compatible black-ratio-only camera read."""
        ratio, _, _ = self._get_camera_black_ratio_and_tilt_direction()
        return ratio

    def _record_episode_result(self, is_success: bool):
        """记录最近若干回合结果，用于日志中的滑动成功率。"""
        self._recent_successes.append(int(is_success))

    def _build_info(
        self,
        raw: dict,
        is_success: bool,
        init_reset: bool = False,
        out_of_bounds: bool = False,
        descended: bool = False,
        contact_xy_force: Optional[float] = None,
    ) -> dict:
        force = raw["force"]
        tcp = raw["tcp"]
        task_pos = raw["task_pos"]
        tilt_deg = raw["tilt_deg"]
        prev_action = raw["prev_action"]
        force_abs_errors = self._force_abs_errors(force)

        return {
            "task_frame_path": self.task_frame_path,
            "reset_pose_path": self.reset_pose_path,
            "reset_pose_loaded": bool(self.reset_pose_loaded),
            "reset_pose_x": float(self.reset_pose_position[0]),
            "reset_pose_y": float(self.reset_pose_position[1]),
            "reset_pose_z": float(self.reset_pose_position[2]),
            "reset_height_above_task_origin_m": float(
                self.cfg.reset_height_above_task_origin_m
            ),
            "reset_target_x": float(self.reset_target_position[0]),
            "reset_target_y": float(self.reset_target_position[1]),
            "reset_target_z": float(self.reset_target_position[2]),
            "reset_target_task_x": float(self.reset_target_task_pos[0]),
            "reset_target_task_y": float(self.reset_target_task_pos[1]),
            "reset_target_task_z": float(self.reset_target_task_pos[2]),
            "task_frame_loaded": bool(self.task_frame_loaded),
            "task_origin_source": str(self.task_origin_source),
            "task_frame_origin_x": float(self.task_frame_origin[0]),
            "task_frame_origin_y": float(self.task_frame_origin[1]),
            "task_frame_origin_z": float(self.task_frame_origin[2]),
            "task_frame_json_origin_x": float(self.task_frame_json_origin[0]),
            "task_frame_json_origin_y": float(self.task_frame_json_origin[1]),
            "task_frame_json_origin_z": float(self.task_frame_json_origin[2]),
            "task_frame_x_axis_x": float(self.task_frame_axes[0, 0]),
            "task_frame_x_axis_y": float(self.task_frame_axes[0, 1]),
            "task_frame_x_axis_z": float(self.task_frame_axes[0, 2]),
            "task_frame_y_axis_x": float(self.task_frame_axes[1, 0]),
            "task_frame_y_axis_y": float(self.task_frame_axes[1, 1]),
            "task_frame_y_axis_z": float(self.task_frame_axes[1, 2]),
            "task_frame_z_axis_x": float(self.task_frame_axes[2, 0]),
            "task_frame_z_axis_y": float(self.task_frame_axes[2, 1]),
            "task_frame_z_axis_z": float(self.task_frame_axes[2, 2]),
            "raw_x": float(tcp[0]),
            "raw_y": float(tcp[1]),
            "raw_z": float(tcp[2]),
            "task_x": float(task_pos[0]),
            "task_y": float(task_pos[1]),
            "task_z": float(task_pos[2]),
            "task_radius": float(np.hypot(task_pos[0], task_pos[2])),
            "prev_task_radius": float(self.prev_task_radius),
            "lateral_limit_m": float(self.cfg.lateral_limit_m),
            "position_out_of_bounds": bool(self.last_position_out_of_bounds),
            "vision_out_of_bounds": bool(self.last_vision_out_of_bounds),
            "theta_x_deg": float(tilt_deg[0]),
            "theta_z_deg": float(tilt_deg[1]),
            "f0_x": float(force[0]),
            "f0_y": float(force[1]),
            "f0_z": float(force[2]),
            "f1_x": float(force[3]),
            "f1_y": float(force[4]),
            "f1_z": float(force[5]),
            # 兼容旧日志字段：f2_* 表示第二路传感器。
            "f2_x": float(force[3]),
            "f2_y": float(force[4]),
            "f2_z": float(force[5]),
            "last_ax": float(prev_action[0]),
            "last_az": float(prev_action[1]),
            "last_atheta_x": float(prev_action[2]),
            "last_atheta_z": float(prev_action[3]),
            "black_ratio": float(self.current_black_ratio),
            "black_ratio_mode": str(self.current_black_ratio_mode),
            "roi_black_ratio": float(self.current_roi_black_ratio),
            "prev_roi_black_ratio": float(self.prev_roi_black_ratio),
            "full_black_ratio": float(self.current_full_black_ratio),
            "roi_ready": bool(self.current_roi_ready),
            "roi_failed": bool(self.current_roi_failed),
            "roi_source": str(self.current_roi_source),
            "target_tilt_x": float(self.target_tilt_dir[0]),
            "target_tilt_z": float(self.target_tilt_dir[1]),
            "black_delta_pp": float(self.last_black_delta_pp),
            "black_delta_reward": float(self.last_black_delta_reward),
            "black_reward": float(self.last_black_reward),
            "black_reward_applied": bool(self.last_black_reward_applied),
            "center_progress_m": float(self.last_center_progress_m),
            "center_progress_reward": float(self.last_center_progress_reward),
            "center_progress_reward_applied": bool(
                self.last_center_progress_reward_applied
            ),
            "center_progress_enable_until_roi": float(
                self.cfg.center_progress_enable_until_roi
            ),
            "roi_to_full_frame_bonus": float(self.cfg.roi_to_full_frame_bonus),
            "roi_to_full_frame_bonus_applied": bool(
                self.last_roi_to_full_frame_bonus_applied
            ),
            "force_abs_errors": [float(v) for v in self.last_force_abs_errors],
            "force_max_error": float(self.last_force_max_error),
            "force_error_f0_x": float(force_abs_errors[0]),
            "force_error_f0_y": float(force_abs_errors[1]),
            "force_error_f0_z": float(force_abs_errors[2]),
            "force_error_f1_x": float(force_abs_errors[3]),
            "force_error_f1_y": float(force_abs_errors[4]),
            "force_error_f1_z": float(force_abs_errors[5]),
            "force_safety_limit": float(self.cfg.force_safety_limit),
            "force_safety_triggered": bool(self.last_force_safety_triggered),
            "weak_down_side_force_threshold": (
                None
                if self.cfg.weak_down_side_force_threshold is None
                else float(self.cfg.weak_down_side_force_threshold)
            ),
            "success_full_black_ratio_min": float(
                self.cfg.insertion_full_black_ratio_success_min
            ),
            "depth_from_surface": float(self.depth_from_surface),
            "last_descend_task_y_delta": float(self.last_descend_task_y_delta),
            "insertion_success_depth_m": float(self.cfg.insertion_success_depth_m),
            "no_camera_success_require_contact_xy_max": float(
                self.cfg.no_camera_success_require_contact_xy_max
            ),
            "depth_success": bool(
                self.depth_from_surface >= float(self.cfg.insertion_success_depth_m)
            ),
            "pre_descend_force_baseline": [
                float(v) for v in self.pre_descend_force_baseline
            ],
            "contact_xy_force": float(
                self._contact_xy_force(force)
                if contact_xy_force is None
                else contact_xy_force
            ),
            "contact_xy_ema": float(self.contact_xy_ema),
            "prev_contact_xy_ema": float(self.prev_contact_xy_ema),
            "contact_xy_drop_ema": float(self.contact_xy_drop_ema),
            "contact_xy_raw": float(self.contact_xy_raw),
            "prev_contact_xy_raw": float(self.prev_contact_xy_raw),
            "contact_xy_raw_drop": float(self.contact_xy_raw_drop),
            "contact_xy_historical_peak": float(self.contact_xy_historical_peak),
            "contact_xy_recent_peak": float(self.contact_xy_recent_peak),
            "contact_xy_peak_drop": float(self.contact_xy_peak_drop),
            "recent_success_rate": float(np.mean(self._recent_successes))
            if self._recent_successes
            else 0.0,
            "is_success": bool(is_success),
            "init_reset": bool(init_reset),
            "out_of_bounds": bool(out_of_bounds),
            "descended": bool(descended),
        }
