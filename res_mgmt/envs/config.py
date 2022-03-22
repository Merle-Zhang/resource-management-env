_EMPTY_CELL: int = -1

Config = dict[str, int]

_DEFAULT_CONFIG: Config = {
    "num_resource_type": 2,  # d resource types
    "num_job_slot": 3,       # first M jobs
    "time_size": 5,          # column
    "resource_size": 3,      # row
    "max_num_job": 10**3,
}
