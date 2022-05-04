from res_mgmt.envs.res_mgmt_env import ResMgmtEnv


def sjf_scoring(job_id: int, env: ResMgmtEnv) -> int:
    return 1 / env.res.meta[job_id].duration
