import numpy as np

from res_mgmt.envs.res_mgmt_env import ResMgmtEnv


num_resource_type = 10
time_size = 5
resource_size = 10


env = ResMgmtEnv(
    num_resource_type=num_resource_type,
    time_size=time_size,
    resource_size=resource_size,
    num_job_slot=10,
    max_num_job=10**3,
)

max_iteration = 10 ** 5
# max_iteration = 10

# render_per_iteration = max_iteration // 10
render_per_iteration = 1000

observation = env.reset()
t = 0
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if t % render_per_iteration == 0:
        print(t, action, reward)
        env.my_render(f"render/{t}.png")
    if done or t > max_iteration:
        print("Episode finished after {} timesteps".format(t))
        break
    t += 1
env.close()
