"""
A_rl_train.py  ——  轴孔装配单阶段 SAC 训练脚本（视觉奖励 + 触觉安全）

python A_rl_train.py --model logs/models/peg_in_hole/sac_final_model
python A_rl_train.py --model logs/models/peg_in_hole/sac_model_1675_steps
--reset-buffer

1、xyz记录洞口上方坐标
2、分割ROI
3、训练


python A_rl_train.py --model logs/models/peg_in_hole/sac_final_model --play-only


主要特性:
    - SacStatsCallback: 实时打印并写入 TensorBoard 统计指标
        train/success_rate  (成功率，全图黑色占比 > 95%)
        train/fail_rate     (失败/耗尽率)
      以及逐回合的相机黑色占比、累计奖励和逐步奖励分量：
        train/black_ratio   (回合末帧黑色占比 %)
        train/episode_reward (回合累计奖励)
        train/black_delta_reward / train/depth_from_surface / train/force_max_error
    - CurriculumMonitorCallback: 将固定课程最大偏移半径写入 TensorBoard
    - learning_starts=200，先收集约 4 个满回合的样本再更新
    - CheckpointWithBufferCallback: Checkpoint 时同步保存 Replay Buffer 和训练步数
    - 支持从断点完整恢复：模型权重 + Replay Buffer + 训练步数，无需重新随机探索
"""

import argparse
import os
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from A_rl_env import PegInHoleEnv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ENVIRONMENT_CLASSES = {
    'peg_in_hole': PegInHoleEnv,
}


# ================================================================
#  自定义回调 1：成功率统计 + TensorBoard 写入
# ================================================================

class SacStatsCallback(BaseCallback):
    """
    每回合结束时从 info 读取结果，统计并写入 TensorBoard。

    奖励体系说明（对应新版 rl_env.py）:
        成功（is_success=True）: 全图黑色占比 > 95%，reward=1.0, terminated=True
        越界/任一轴力误差超限: reward=-0.5, truncated=True
        步数耗尽: 按当前稠密奖励结束，truncated=True
        中间步骤: 黑色占比变化奖励

    TensorBoard 键名:
        train/success_rate    → 成功率
        train/fail_rate       → 失败/耗尽率
        train/black_ratio     → 回合末帧黑色区域占比（%）
        train/episode_reward  → 回合累计奖励
        train/episode_total   → 累计回合数
        train/black_delta_reward → 当前步黑色占比变化奖励
        train/depth_from_surface → 累计实际下降距离
        train/last_descend_task_y_delta → 当前步实际下降距离
        train/force_max_error → 两路传感器六个轴向分量中的最大力误差
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_success    = 0
        self.ep_fail       = 0
        self.ep_total      = 0
        self._ep_reward    = 0.0   # 当前回合累计奖励（步内累加）

    def _on_step(self) -> bool:
        # 累加当前步奖励（支持单环境，rewards 是列表）
        self._ep_reward += float(self.locals['rewards'][0])
        if self.locals.get('infos'):
            step_info = self.locals['infos'][0]
            self.logger.record(
                "train/black_reward",
                float(step_info.get("black_reward", 0.0)),
            )
            self.logger.record(
                "train/black_delta_pp",
                float(step_info.get("black_delta_pp", 0.0)),
            )
            self.logger.record(
                "train/black_delta_reward",
                float(step_info.get("black_delta_reward", 0.0)),
            )
            self.logger.record(
                "train/center_progress_mm",
                float(step_info.get("center_progress_m", 0.0)) * 1000.0,
            )
            self.logger.record(
                "train/center_progress_reward",
                float(step_info.get("center_progress_reward", 0.0)),
            )
            self.logger.record(
                "train/center_progress_reward_applied",
                float(bool(step_info.get("center_progress_reward_applied", False))),
            )
            self.logger.record(
                "train/roi_to_full_frame_bonus",
                float(step_info.get("roi_to_full_frame_bonus", 0.0)),
            )
            self.logger.record(
                "train/roi_to_full_frame_bonus_applied",
                float(bool(step_info.get("roi_to_full_frame_bonus_applied", False))),
            )
            self.logger.record(
                "train/roi_black_ratio",
                float(step_info.get("roi_black_ratio", 0.0)),
            )
            self.logger.record(
                "train/full_black_ratio",
                float(step_info.get("full_black_ratio", 0.0)),
            )
            self.logger.record(
                "train/contact_xy_force",
                float(step_info.get("contact_xy_force", 0.0)),
            )
            self.logger.record(
                "train/contact_xy_ema",
                float(step_info.get("contact_xy_ema", 0.0)),
            )
            self.logger.record(
                "train/contact_xy_drop_ema",
                float(step_info.get("contact_xy_drop_ema", 0.0)),
            )
            self.logger.record(
                "train/contact_xy_raw",
                float(step_info.get("contact_xy_raw", 0.0)),
            )
            self.logger.record(
                "train/prev_contact_xy_raw",
                float(step_info.get("prev_contact_xy_raw", 0.0)),
            )
            self.logger.record(
                "train/contact_xy_raw_drop",
                float(step_info.get("contact_xy_raw_drop", 0.0)),
            )
            self.logger.record(
                "train/contact_xy_recent_peak",
                float(step_info.get("contact_xy_recent_peak", 0.0)),
            )
            self.logger.record(
                "train/contact_xy_peak_drop",
                float(step_info.get("contact_xy_peak_drop", 0.0)),
            )
            self.logger.record(
                "train/depth_from_surface",
                float(step_info.get("depth_from_surface", 0.0)),
            )
            self.logger.record(
                "train/last_descend_task_y_delta",
                float(step_info.get("last_descend_task_y_delta", 0.0)),
            )
            self.logger.record(
                "train/task_radius",
                float(step_info.get("task_radius", 0.0)),
            )
            self.logger.record(
                "train/force_max_error",
                float(step_info.get("force_max_error", 0.0)),
            )
            self.logger.record(
                "train/force_safety_triggered",
                float(bool(step_info.get("force_safety_triggered", False))),
            )
            self.logger.record(
                "train/out_of_bounds",
                float(bool(step_info.get("out_of_bounds", False))),
            )

        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if not done:
                continue

            self.ep_total += 1

            if info.get("is_success", False):
                self.ep_success += 1
            else:
                self.ep_fail += 1

            success_rate = self.ep_success / self.ep_total
            fail_rate    = self.ep_fail    / self.ep_total
            black_ratio  = float(info.get("black_ratio", 0.0))
            ep_reward    = self._ep_reward

            # ── 写入 TensorBoard ─────────────────────────────────
            self.logger.record("train/success_rate",   success_rate)
            self.logger.record("train/fail_rate",      fail_rate)
            self.logger.record("train/black_ratio",    black_ratio)
            self.logger.record("train/episode_reward", ep_reward)
            self.logger.record("train/episode_total",  self.ep_total)

            # ── 打印到终端 ────────────────────────────────────────
            print(
                f"\n{'─'*60}\n"
                f"  回合统计 (共 {self.ep_total} 回合)\n"
                f"  ✅ 成功:   {self.ep_success:>3d}  ({success_rate:.1%})\n"
                f"  ❌ 失败:   {self.ep_fail:>3d}  ({fail_rate:.1%})\n"
                f"         末帧黑色占比: {black_ratio:.2f}%\n"
                f"         回合累计奖励: {ep_reward:+.4f}\n"
                f"{'─'*60}"
            )

            # 回合结束，重置回合累计奖励
            self._ep_reward = 0.0

        return True


# ================================================================
#  自定义回调 2：课程学习进度写入 TensorBoard
# ================================================================

class CurriculumMonitorCallback(BaseCallback):
    """
    每步把当前课程偏移量写入 TensorBoard。
    新版环境中偏移半径固定为配置值，此回调主要用于确认配置正确。

    TensorBoard 键名:
        train/curriculum_offset_mm → 当前最大偏移半径（毫米）
    """

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env   # Monitor 包装后的环境

    def _on_step(self) -> bool:
        try:
            offset_mm = self.env.unwrapped.curriculum_offset * 1000
            self.logger.record("train/curriculum_offset_mm", offset_mm)
        except AttributeError:
            pass
        return True


# ================================================================
#  自定义回调 3：Checkpoint 时同步保存 Replay Buffer + 训练步数
# ================================================================

class CheckpointWithBufferCallback(CheckpointCallback):
    """
    在 CheckpointCallback 基础上，每次保存模型时额外保存：
        - Replay Buffer（.pkl）
        - 当前训练步数（_steps.txt）

    保存文件命名示例（save_freq=50, 第500步时）:
        sac_model_500_steps.zip          ← 模型权重（原 CheckpointCallback）
        sac_model_500_steps_buffer.pkl   ← Replay Buffer
        sac_model_500_steps_steps.txt    ← 训练步数
    """

    def _on_step(self) -> bool:
        result = super()._on_step()
        # 与父类保存频率对齐：每 save_freq 步触发一次
        if self.n_calls % self.save_freq == 0:
            base_name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            buf_path  = os.path.join(self.save_path, base_name + '_buffer')
            step_path = os.path.join(self.save_path, base_name + '_steps.txt')

            self.model.save_replay_buffer(buf_path)
            with open(step_path, 'w') as f:
                f.write(str(self.num_timesteps))

            print(f"[Checkpoint] Buffer 已保存 → {buf_path}.pkl  "
                  f"(样本数: {self.model.replay_buffer.size()})")
        return result


# ================================================================
#  工具函数：保存模型 + Buffer + 步数
# ================================================================

def save_full_checkpoint(model, path: str):
    """
    一次性保存模型权重、Replay Buffer、训练步数到同一路径前缀。

    生成文件:
        {path}.zip          ← 模型权重
        {path}_buffer.pkl   ← Replay Buffer
        {path}_steps.txt    ← 训练步数
    """
    model.save(path)
    model.save_replay_buffer(path + '_buffer')
    with open(path + '_steps.txt', 'w') as f:
        f.write(str(model.num_timesteps))
    print(f"[Save] 完整断点已保存 → {path}  "
          f"(步数={model.num_timesteps}, "
          f"Buffer样本数={model.replay_buffer.size()})")


# ================================================================
#  环境工厂
# ================================================================

def make_env(
    env_class_name: str,
    task_frame_path: str,
    require_camera: bool = True,
    confirm_roi_on_reset: bool = True,
):
    env_class = ENVIRONMENT_CLASSES[env_class_name]
    return Monitor(
        env_class(
            task_frame_path=task_frame_path,
            require_camera=require_camera,
            confirm_roi_on_reset=confirm_roi_on_reset,
        )
    )


# ================================================================
#  主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Peg-in-Hole SAC Training')
    parser.add_argument('--algorithm',       default='SAC',          type=str)
    parser.add_argument('--model',           default='',             type=str,
                        help='预训练模型路径，留空则从头训练')
    parser.add_argument('--reset-buffer',    default=False,          action='store_true',
                        help='加载模型权重但不恢复旧 Replay Buffer，重新收集新奖励样本')
    parser.add_argument('--play-only',       default=False,          action='store_true',
                        help='仅测试，不训练')
    parser.add_argument('--no-camera',       default=False,          action='store_true',
                        help='无相机部署/测试模式：不初始化相机，跳过 ROI 确认')
    parser.add_argument('--skip-roi-confirm', default=False,         action='store_true',
                        help='跳过训练/测试启动前 ROI 确认；相机仍可初始化用于奖励/日志')
    parser.add_argument('--environment',     default='peg_in_hole',  type=str)
    parser.add_argument('--task-frame',      default='rl_task_frame_left.json',
                        type=str, help='锁定任务坐标系 JSON 路径')

    # ── SAC 超参数 ──────────────────────────────────────────────
    parser.add_argument('--lr',              default=5e-3,    type=float)
    parser.add_argument('--gradient_steps',  default=1,       type=int)
    parser.add_argument('--train_freq',      default=1,       type=int)
    parser.add_argument('--buffer_size',     default=50_000,  type=int)
    parser.add_argument('--learning_starts', default=200,     type=int,
                        help='前期随机探索步数，保证 Buffer 有足够多样本再开始更新'
                             '（从断点恢复时自动设为 0，无需手动修改）')
    parser.add_argument('--gamma',           default=0.95,    type=float)
    parser.add_argument('--batch_size',      default=128,     type=int)
    parser.add_argument('--total_timesteps', default=100_000, type=int)
    parser.add_argument('--save_freq',       default=50,      type=int,
                        help='Checkpoint 保存频率（步）')

    args = parser.parse_args()

    # ── 初始化环境 ─────────────────────────────────────────────
    skip_roi_confirm = bool(args.no_camera or args.skip_roi_confirm)
    env = make_env(
        args.environment,
        args.task_frame,
        require_camera=not args.no_camera,
        confirm_roi_on_reset=not skip_roi_confirm,
    )
    if skip_roi_confirm:
        print("[Train] 已跳过 ROI 确认"
              + ("（无相机模式）" if args.no_camera else ""))
    else:
        try:
            env.unwrapped.confirm_hole_roi_before_training(require_confirmed=True)
        except Exception:
            env.close()
            raise

    if args.no_camera and not args.play_only:
        print("[Train] ⚠️  当前为 --no-camera 训练：视觉奖励/日志将为 0，"
              "通常只建议用于部署测试。")

    drl_algorithm_classes = {'SAC': stable_baselines3.SAC}
    drl_algorithm_class   = drl_algorithm_classes[args.algorithm]

    # ── 初始化或加载模型 ───────────────────────────────────────
    if args.model:
        # ---- 从断点恢复 ----------------------------------------
        model = drl_algorithm_class.load(
            args.model, env=env, learning_rate=args.lr
        )
        print(f"[Train] 已加载模型权重: {args.model}")

        # 恢复 Replay Buffer；奖励函数变化后可用 --reset-buffer 避免旧奖励污染。
        buffer_path = args.model + '_buffer'
        buffer_restored = False
        if args.reset_buffer:
            print("[Train] --reset-buffer 已启用：跳过旧 Replay Buffer，"
                  f"将重新收集 {args.learning_starts} 步新样本")
        elif os.path.exists(buffer_path + '.pkl'):
            model.load_replay_buffer(buffer_path)
            buffer_restored = True
            print(f"[Train] 已恢复 Replay Buffer  "
                  f"(样本数: {model.replay_buffer.size()})")
        else:
            print(f"[Train] ⚠️  未找到 Buffer 文件 ({buffer_path}.pkl)，"
                  f"将重新收集 {args.learning_starts} 步随机样本")

        # 恢复训练步数
        steps_file = args.model + '_steps.txt'
        if os.path.exists(steps_file):
            with open(steps_file) as f:
                restored_steps = int(f.read().strip())
            model.num_timesteps = restored_steps
            model._episode_num  = 0
            if buffer_restored:
                # Buffer 已恢复，无需再随机探索
                model.learning_starts = 0
                print(f"[Train] 已恢复训练步数: {restored_steps}  "
                      f"(learning_starts 已自动设为 0)")
            else:
                model.learning_starts = args.learning_starts
                print(f"[Train] 已恢复训练步数: {restored_steps}  "
                      f"(未恢复 Buffer，learning_starts={model.learning_starts})")
        else:
            print(f"[Train] ⚠️  未找到步数文件 ({steps_file})，"
                  f"步数计数从 0 重新开始")

    else:
        # ---- 全新训练 ------------------------------------------
        algorithm_args = {}
        if args.algorithm == 'SAC':
            algorithm_args.update({
                'buffer_size':     args.buffer_size,
                'gradient_steps':  args.gradient_steps,
                'train_freq':      (args.train_freq, "step"),
                'learning_starts': args.learning_starts,
                'gamma':           args.gamma,
                'batch_size':      args.batch_size,
                'ent_coef':        'auto',   # 自动调节熵系数，平衡探索与利用
            })

        print("[Train] 正在初始化全新的 SAC 模型...")
        model = drl_algorithm_class(
            'MlpPolicy',
            env,
            device='cpu',
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log='logs/tb/peg_in_hole',
            **algorithm_args
        )

    # ── 测试模式 ───────────────────────────────────────────────
    if args.play_only:
        print("=" * 60)
        print("[Train] 进入纯测试模式 (Play Only)")
        print("=" * 60)

        ep_total = ep_success = ep_fail = 0
        ep_reward = 0.0
        obs, _ = env.reset()

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            print(f"  动作: [ΔX={action[0]:.3f}  ΔZ={action[1]:.3f}  "
                  f"ΔθX={action[2]:.3f}  ΔθZ={action[3]:.3f}]  "
                  f"reward={reward:+.4f}  "
                  f"黑色占比={info.get('black_ratio', 0.0):.2f}%")

            if done:
                ep_total += 1
                if info.get("is_success"):
                    ep_success += 1
                else:
                    ep_fail += 1

                print(
                    f"\n  测试统计 (共 {ep_total} 回合) │ "
                    f"进洞={ep_success}  失败={ep_fail}  "
                    f"本回合累计奖励={ep_reward:+.4f}"
                )
                ep_reward = 0.0
                obs, _ = env.reset()

    # ── 训练模式 ───────────────────────────────────────────────
    else:
        print("=" * 60)
        print("[Train] 开始单阶段 SAC 强化学习训练（视觉奖励 + 触觉安全）")
        print(f"        learning_starts={model.learning_starts}  "
              f"buffer_size={args.buffer_size}  "
              f"curriculum_max={env.unwrapped.curriculum_offset * 1000:.1f}mm")
        print("=" * 60)

        save_path = './logs/models/peg_in_hole'
        os.makedirs(save_path, exist_ok=True)

        # CheckpointWithBufferCallback 替代原 CheckpointCallback
        # 每次保存模型时同步保存 Replay Buffer 和训练步数
        checkpoint_cb = CheckpointWithBufferCallback(
            save_freq=args.save_freq,
            save_path=save_path,
            name_prefix='sac_model'
        )
        stats_cb      = SacStatsCallback(verbose=0)
        curriculum_cb = CurriculumMonitorCallback(env=env, verbose=0)

        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=[checkpoint_cb, stats_cb, curriculum_cb],
                log_interval=4,
                reset_num_timesteps=False,   # 断点恢复时保持步数连续
            )
        except KeyboardInterrupt:
            print("\n[Train] 检测到 Ctrl+C，正在保存中断断点...")
            save_full_checkpoint(
                model, os.path.join(save_path, 'interrupted_model')
            )
        finally:
            print("[Train] 训练结束，正在保存最终断点...")
            save_full_checkpoint(
                model, os.path.join(save_path, 'sac_final_model')
            )
            env.close()


if __name__ == '__main__':
    main()
