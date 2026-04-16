#!/bin/bash
# eval_smolvla_libero.sh

# Install required dependencies
pip install -e ".[libero,smolvla]"

# Set MuJoCo rendering backend (egl for headless, glfw for with display)
export MUJOCO_GL=egl

# Run evaluation
# We use --policy.n_action_steps=50 because smolvla is trained with chunking (chunk_size=50).
# Although the config.json on the Hub says 1, 50 is common for chunking models.
# You can adjust --eval.n_episodes=10 and --eval.batch_size=1 depending on your GPU.
lerobot-eval \
  --output_dir=outputs/eval/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.n_action_steps=50 \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda
