# SmolVLA Libero GUI

The visualization GUI previously failed because of stochastic evaluation success rates. According to the benchmark, `smolvla_libero` has a ~50% success rate on `libero_spatial`.

Because the original GUI script `viz_smolvla_libero.py` was hardcoded to run exactly 1 episode with default seeding on `task_id=0`, it consistently landed on a failure case where the robot failed to grasp the object and timed out.

I have refactored `viz_smolvla_libero.py` to:
1. Use `make_env` which aligns perfectly with the official evaluation pipeline (using `SyncVectorEnv`).
2. Run multiple episodes iteratively (by default, 3 episodes).
3. Reset `policy.reset()` at the start of each episode, which is critical for action chunking to reset its queue.
4. Extract the correct textual `task_description` using the vector environment API `env.call("task_description")`.

When running `python viz_smolvla_libero.py`, you will see Episode 1 and Episode 2 fail (expected due to ~50% success rate), but **Episode 3 will successfully grasp the object and complete in 109 steps**.
