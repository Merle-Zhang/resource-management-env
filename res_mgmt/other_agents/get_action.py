from typing import Callable

from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.res_mgmt_env import ResMgmtEnv


def get_action(env: ResMgmtEnv, scoring: Callable[[int, ResMgmtEnv], float]) -> int:
    result = 0
    result_score = -1
    for index, job_id in enumerate(env.res.job_slots.jobs):
        if job_id == _EMPTY_CELL:
            continue
        pos = env.res.find_pos(job_id)
        if pos != 0: # schedule immediately 
            continue
        tmp_score = scoring(job_id, env)
        if tmp_score > result_score:
            result = index + 1
            result_score = tmp_score
    return result
