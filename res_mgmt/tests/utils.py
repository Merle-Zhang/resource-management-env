from res_mgmt.envs.job import Job


def meta_from_durations(durations: dict[int, Job]):
    return {id: Job(duration=duration) for id, duration in durations.items()}
