import numpy as np

from res_mgmt.envs.res_mgmt_env import ResMgmtEnv


def packer_scoring(job_id: int, env: ResMgmtEnv) -> int:
    requirements = env.res.meta[job_id].requirements #(num_resource_type, time_size,)
    res_vec = np.zeros(requirements.shape[0], dtype=int)
    for i in range(requirements.shape[0]):
        res_vec[i] = requirements[i, 0]
    avbl_res = env.res.empty_cells_cluster[:, 0]
    return avbl_res.dot(res_vec)