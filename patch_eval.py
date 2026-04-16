import re

def patch_eval():
    with open('src/lerobot/scripts/lerobot_eval.py', 'r') as f:
        content = f.read()
    
    # 修改 rollout 中将 task 放进 observation 的逻辑
    # 把它存入 complementary data 中
    target = '''        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)'''
        
    replacement = '''        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)
        
        # FIX: LiberoProcessorStep drops non-standard keys like "task"
        # We need to explicitly inject the task into COMPLEMENTARY_DATA so TokenizerProcessorStep can find it
        from lerobot.configs.types import TransitionKey
        if TransitionKey.COMPLEMENTARY_DATA not in observation:
            observation[TransitionKey.COMPLEMENTARY_DATA] = {}
        
        try:
            task_desc = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                task_desc = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                task_desc = [""] * env.num_envs
        observation[TransitionKey.COMPLEMENTARY_DATA]["task"] = task_desc'''
    
    # 找到原来赋值 task 的地方删掉或者保留无所谓，但为了干净还是改掉
    old_task_logic = '''        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                observation["task"] = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                observation["task"] = [""] * env.num_envs'''
                
    content = content.replace(old_task_logic, "")
    content = content.replace(target, replacement)
    
    with open('src/lerobot/scripts/lerobot_eval.py', 'w') as f:
        f.write(content)

patch_eval()
