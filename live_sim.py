import argparse
import cv2
import gymnasium as gym
import numpy as np
import torch
import time
from pathlib import Path
import contextlib
import os

os.environ["MUJOCO_GL"] = "glfw"

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.types import TransitionKey
from lerobot.utils.device_utils import get_safe_torch_device

@parser.wrap()
def live_eval(cfg: EvalPipelineConfig):
    device = get_safe_torch_device("cuda")
    print(f"Loading policy on {device}...")
    
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
    )
    
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    print("Creating environment...")
    cfg.eval.use_async_envs = False
    
    envs_dict = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
    )
    
    task_group = list(envs_dict.keys())[0]
    task_id = list(envs_dict[task_group].keys())[0]
    env = envs_dict[task_group][task_id]
    
    print(f"Starting live eval for {task_group} / {task_id}")
    
    cv2.namedWindow('Live Sim', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Sim', 800, 800)
    
    policy.reset()
    observation, info = env.reset()
    
    max_steps = env.call("_max_episode_steps")[0]
    
    for step in range(max_steps):
        frames = env.call("render")
        if isinstance(frames, tuple) and len(frames) > 0:
            frames = frames[0]

        if isinstance(frames, list) or isinstance(frames, np.ndarray):
             frame = np.array(frames)
             if len(frame.shape) > 3:
                 frame = frame[0]
        else:
             print(f"unknown frame type {type(frames)}")
             continue

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Live Sim', frame_bgr)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else contextlib.nullcontext():
            # Pass observation wrapped in a dict with OBS_STR
            transition = {TransitionKey.OBSERVATION.value: observation}
            obs_batch = env_preprocessor(transition)
            obs_batch = preprocessor(obs_batch)
            
            action_batch = policy.select_action(obs_batch)
            action_batch = postprocessor(action_batch)
            action_batch = env_postprocessor({ACTION: action_batch})[ACTION]
            
            action_numpy = action_batch.to("cpu").numpy()
        
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        
        done = terminated[0] or truncated[0]
        if done:
            print("Episode finished!")
            break
            
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eval()
