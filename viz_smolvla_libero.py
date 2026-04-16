import torch
import numpy as np
import cv2
import os
import time
from pathlib import Path
from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.factory import make_env_pre_post_processors, make_env_config, make_env
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import preprocess_observation


def main():
    # 随机种子非常重要，测试发现并非所有seed/episode都能成功（成功率约50%）
    import random

    seed = 100000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "HuggingFaceVLA/smolvla_libero"

    # 1. 准备环境配置
    print("Preparing environment config...")
    env_cfg = make_env_config("libero", task="libero_object", observation_width=256, observation_height=256)
    env_cfg.control_mode = "relative"  # 必须使用relative
    env_cfg.max_episode_steps = 200

    # 2. 加载模型配置
    print(f"Loading policy config from {model_id}...")
    policy_cfg = PreTrainedConfig.from_pretrained(model_id)
    policy_cfg.pretrained_path = Path(model_id)
    policy_cfg.device = str(device)
    policy_cfg.n_action_steps = 50  # Chunking

    # 3. 加载模型
    print(f"Instantiating policy...")
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # 4. 创建环境
    # 使用官方的make_env以获得完全一致的观察空间处理（包含SyncVectorEnv）
    os.environ["MUJOCO_GL"] = "glfw"
    print("Making environment via official make_env...")
    envs = make_env(env_cfg, n_envs=1)

    # 获取 libero_object 的第一个任务
    env = envs["libero_object"][2]

    # 5. 准备处理器
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=model_id)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    cv2.namedWindow("SmolVLA Libero Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SmolVLA Libero Visualization", 640, 640)

    n_episodes = 10

    for ep in range(n_episodes):
        print(f"\n--- Starting Episode {ep + 1} ---")
        policy.reset()  # 必须重置策略
        obs, info = env.reset(seed=seed + ep)
        done = False
        step_count = 0

        while not done and step_count < env_cfg.max_episode_steps:
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

            action_numpy = action.to("cpu").numpy()
            obs, reward, terminated, truncated, info = env.step(action_numpy)
            step_count += 1

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
                cv2.putText(
                    img_bgr,
                    f"Ep: {ep + 1} | Step: {step_count} | Status: {status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("SmolVLA Libero Visualization", img_bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
            except Exception as _e:
                pass

            is_terminated = terminated[0] if isinstance(terminated, (list, tuple, np.ndarray)) else terminated
            is_truncated = truncated[0] if isinstance(truncated, (list, tuple, np.ndarray)) else truncated
            is_success = info.get("is_success", [False])
            if isinstance(is_success, (list, tuple, np.ndarray)):
                is_success = is_success[0]

            if is_terminated or is_truncated or is_success:
                print(f"Episode {ep + 1} finished. Success: {is_success}, Steps: {step_count}")
                done = True
                time.sleep(1)  # 暂停以便观察结果
                break

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
