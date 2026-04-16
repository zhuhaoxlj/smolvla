"""SmolVLA 模型在 Libero 环境中的评估与可视化脚本。"""

import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import preprocess_observation
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors

# 模块级常量
SEED = 100000
MODEL_ID = "HuggingFaceVLA/smolvla_libero"
MAX_EPISODES = 10
MAX_EPISODE_STEPS = 200
OBS_SIZE = 256
ACTION_STEPS = 2


def set_seed(seed: int) -> None:
    """设置全局随机种子以保证可复现性。

    随机种子非常重要，测试发现并非所有 seed/episode 都能成功（成功率约 50%）。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_environment() -> tuple[Any, Any]:
    """配置并创建 Libero 环境。

    Returns:
        包含 (环境配置, 环境实例) 的元组
    """
    print("Preparing environment config...")
    env_cfg = make_env_config("libero", task="libero_object", observation_width=OBS_SIZE, observation_height=OBS_SIZE)
    env_cfg.control_mode = "relative"  # 必须使用 relative
    env_cfg.max_episode_steps = MAX_EPISODE_STEPS

    os.environ["MUJOCO_GL"] = "glfw"
    print("Making environment via official make_env...")
    envs = make_env(env_cfg, n_envs=1)

    # 获取 libero_object 的第一个任务
    env = envs["libero_object"][2]

    return env_cfg, env


def setup_policy(env_cfg: Any, device: torch.device) -> tuple[Any, PreTrainedConfig]:
    """加载并实例化策略模型。

    Args:
        env_cfg: 环境配置对象
        device: 运行设备 (CPU/CUDA)

    Returns:
        包含 (策略实例, 策略配置) 的元组
    """
    print(f"Loading policy config from {MODEL_ID}...")
    policy_cfg = PreTrainedConfig.from_pretrained(MODEL_ID)
    policy_cfg.pretrained_path = Path(MODEL_ID)
    policy_cfg.device = str(device)
    policy_cfg.n_action_steps = ACTION_STEPS  # Chunking 配置

    print("Instantiating policy...")
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    return policy, policy_cfg


def get_action(
    obs: dict[str, Any],
    env: Any,
    policy: Any,
    device: torch.device,
    preprocessor: Callable,
    postprocessor: Callable,
    env_preprocessor: Callable,
    env_postprocessor: Callable
) -> np.ndarray:
    """根据当前观察获取策略动作。

    处理观察字典、调用策略、然后反处理动作格式。

    Args:
        obs: 当前环境观察
        env: 环境实例，用于获取任务描述
        policy: 策略模型实例
        device: 计算设备
        preprocessor: 模型预处理器
        postprocessor: 模型后处理器
        env_preprocessor: 环境预处理器
        env_postprocessor: 环境后处理器

    Returns:
        可直接用于 env.step 的动作数组 (numpy)
    """
    with torch.no_grad():
        obs_dict = preprocess_observation(obs)
        obs_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()
        }

        obs_t = env_preprocessor(obs_dict)

        try:
            obs_t["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            obs_t["task"] = [""] * env.num_envs

        obs_t = preprocessor(obs_t)
        actions = policy.select_action(obs_t)
        actions = postprocessor(actions)

        action_transition = {"action": actions}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

    return action.to("cpu").numpy()


def render_environment(env: Any, ep: int, step_count: int, info: dict[str, Any], success_count: int) -> bool:
    """渲染环境并显示在 OpenCV 窗口。

    Args:
        env: 环境实例
        ep: 当前回合索引
        step_count: 当前步数
        info: 环境步进返回的信息字典
        success_count: 历史成功次数，用于计算成功率

    Returns:
        如果用户按下 'q' 退出，返回 True；否则返回 False
    """
    try:
        img = env.render()
        if isinstance(img, tuple) and len(img) > 0:
            img = img[0]
        if img.shape[0] == 1:
            img = img[0]

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        is_success = info.get("is_success", [False])
        if isinstance(is_success, (list, tuple, np.ndarray)):
            is_success = is_success[0]

        status = "SUCCESS" if is_success else "RUNNING"

        # 字体更小，不加粗
        font_scale = 0.3
        font_thickness = 1

        # 第一行：基础状态
        cv2.putText(
            img_bgr,
            f"Ep: {ep + 1}/{MAX_EPISODES} | Step: {step_count}/{MAX_EPISODE_STEPS}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            font_thickness,
        )

        # 第二行：状态和成功率
        current_sr = (success_count / ep * 100) if ep > 0 else 0.0
        cv2.putText(
            img_bgr,
            f"Status: {status} | SR: {current_sr:.1f}% ({success_count}/{ep})",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            font_thickness,
        )

        cv2.imshow("SmolVLA Libero Visualization", img_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return True
    except cv2.error:
        pass

    return False


def check_termination(terminated: Any, truncated: Any, info: dict[str, Any], ep: int, step_count: int) -> bool:
    """检查回合是否应该结束。

    Args:
        terminated: 环境是否到达终止状态
        truncated: 环境是否被截断
        info: 环境状态信息
        ep: 当前回合索引
        step_count: 当前步数

    Returns:
        如果应该结束当前回合，返回 True
    """
    is_terminated = terminated[0] if isinstance(terminated, (list, tuple, np.ndarray)) else terminated
    is_truncated = truncated[0] if isinstance(truncated, (list, tuple, np.ndarray)) else truncated

    is_success = info.get("is_success", [False])
    if isinstance(is_success, (list, tuple, np.ndarray)):
        is_success = is_success[0]

    if is_terminated or is_truncated or is_success:
        print(f"Episode {ep + 1} finished. Success: {is_success}, Steps: {step_count}")
        time.sleep(1)  # 暂停以便观察结果
        return True

    return False


def run_episode(
    ep: int,
    env: Any,
    env_cfg: Any,
    policy: Any,
    device: torch.device,
    processors: dict[str, Callable],
    success_count: int,
) -> tuple[bool, bool]:
    """运行单个评估回合。

    Args:
        ep: 回合索引
        env: 环境实例
        env_cfg: 环境配置
        policy: 策略实例
        device: 计算设备
        processors: 包含预处理/后处理函数的字典
        success_count: 历史成功次数

    Returns:
        元组 (should_exit_program, is_success)
    """
    print(f"\n--- Starting Episode {ep + 1} ---")
    policy.reset()  # 必须重置策略
    obs, _info = env.reset(seed=SEED + ep)
    step_count = 0
    episode_success = False

    while step_count < env_cfg.max_episode_steps:
        action_numpy = get_action(
            obs, env, policy, device,
            processors["preprocessor"],
            processors["postprocessor"],
            processors["env_preprocessor"],
            processors["env_postprocessor"]
        )

        obs, _reward, terminated, truncated, info = env.step(action_numpy)
        step_count += 1

        is_success_info = info.get("is_success", [False])
        if isinstance(is_success_info, (list, tuple, np.ndarray)):
            is_success_info = is_success_info[0]

        if is_success_info:
            episode_success = True

        if render_environment(env, ep, step_count, info, success_count):
            return True, episode_success  # 提前退出整个程序

        if check_termination(terminated, truncated, info, ep, step_count):
            break

    return False, episode_success


def main() -> None:
    """执行评估循环并渲染可视化窗口。"""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg, env = setup_environment()
    policy, policy_cfg = setup_policy(env_cfg, device)

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=MODEL_ID)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    processors = {
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor
    }

    cv2.namedWindow("SmolVLA Libero Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SmolVLA Libero Visualization", 640, 640)

    success_count = 0

    for ep in range(MAX_EPISODES):
        should_exit, is_success = run_episode(ep, env, env_cfg, policy, device, processors, success_count)
        if is_success:
            success_count += 1

        if should_exit:
            break

    env.close()
    cv2.destroyAllWindows()

    final_sr = (success_count / MAX_EPISODES) * 100
    print(f"\n{'='*40}")
    print(f"Evaluation Complete!")
    print(f"Total Episodes: {MAX_EPISODES}")
    print(f"Successful Episodes: {success_count}")
    print(f"Final Success Rate: {final_sr:.1f}%")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
