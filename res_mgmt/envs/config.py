from typing import Dict

from res_mgmt.envs.generator import get_generator

_EMPTY_CELL: int = -1

Config = Dict[str, int]

_DEFAULT_CONFIG: Config = {
    "num_resource_type": 2,  # d resource types
    "num_job_slot": 3,       # first M jobs
    "time_size": 5,          # column
    "resource_size": 3,      # row
    "max_num_job": 10**3,
    "new_job_rate": 0.7,
    "meta": {},
    "generator": get_generator(2, 20, 20),
}
