import gymnasium as gym
import inspect

# 创建环境
env = gym.make("Hopper-v4")

# 查看环境类的位置
print(inspect.getfile(env.unwrapped.__class__))

# 查看 step 方法（奖励在这里计算）
print(inspect.getsource(env.unwrapped.step))


# 在debug时也可以保存模型权重文件
# torch.save(agent.actor), "model.xml")



# # 在 Python 中使用
# import mujoco.viewer

# # 交互式查看器
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         mujoco.mj_step(model, data)
#         viewer.sync()