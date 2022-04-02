from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from res_mgmt.envs.res_mgmt_env import ResMgmtEnv

num_resource_type = 2
time_size = 20
resource_size = 20
num_job_slot = 10
n = 10**2

# Parallel environments
env = make_vec_env(ResMgmtEnv, n_envs=4, env_kwargs={
    "num_resource_type": num_resource_type,
    "time_size": time_size,
    "resource_size": resource_size,
    "num_job_slot": num_job_slot,
    "max_num_job": n,
})

model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    gamma=1,
    tensorboard_log="./a2c_res_mgmt_tensorboard/",
    # device="cpu",
)

model.learn(total_timesteps=10**5)
model.save("a2c_res_mgmt")

del model  # remove to demonstrate saving and loading
