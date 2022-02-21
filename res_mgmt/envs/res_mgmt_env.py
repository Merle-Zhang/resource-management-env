from typing import Optional
import gym
import numpy as np

# TODO: add type hint
# TODO: merge steps?


class ResMgmtEnv(gym.Env):
    def __init__(
        self,
        num_resource_type: int,  # d resource types
        resource_size: int,  # row
        time_size: int,  # column
        num_job_slot: int,  # first M jobs
        # number of colors, TODO: num_job needed or just len(jobs)?
        num_job: int,
        size_backlog: int,  # TODO: size_backlog needed or just infinite?
        jobs: list  # shape(num_job, num_resource_type)
    ):
        self.num_resource_type = num_resource_type
        self.resource_size = resource_size
        self.time_size = time_size
        self.num_job_slot = num_job_slot
        self.jobs = jobs

        # first M jobs + the null sign
        self.action_space = gym.spaces.Discrete(
            num_job_slot + 1)

        # (number of colors + empty) * (row size * column size)
        cluster_image = gym.spaces.MultiDiscrete(
            [num_job + 1] * (resource_size * time_size))
        job_slot_image = gym.space.MultiBinary(resource_size * time_size)

        self.observation_space = gym.spaces.Dict({
            "clusters": gym.space.Tuple(tuple(
                [cluster_image] * num_resource_type
            )),
            "slots": gym.space.Tuple(tuple(
                [job_slot_image] * num_job_slot
            )),
            "backlog": gym.space.Discrete(size_backlog)
        })

        self.state = None

    # TODO: for now just clear the job slot to all empty. Do we need to remove it from the list? if not we need to check for selecting empty slot?
    def step(self, action: int):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        # assert self.state is not None, "Call reset before using step method." # TODO: why? https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L116

        if action == self.num_job_slot or not self.__action_valid(action):
            self.__time_proceed()
            reward = self.__reward()
        else:
            self.__schedule(action)
            reward = None

        state = np.array(self.state, dtype=np.uintc)
        reward = reward  # TODO: return None if not time proceed for now, but should return float/int
        done = not self.remaining_jobs
        info = {}
        return state, reward, done, info

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.remaining_jobs = self.jobs

        cluster = [0] * (self.resource_size * self.time_size)
        clusters = [cluster] * self.num_resource_type  # 0 means empty

        # TODO: might need pre-process before plug into slots
        slots = self.remaining_jobs[:self.num_job_slot]
        self.remaining_jobs = self.remaining_jobs[self.num_job_slot:]

        backlog = len(self.remaining_jobs)

        self.state = [clusters, slots, backlog]
        return np.array(self.state, dtype=np.uintc)

    # TODO: schedule a job that does “fit”
    def __action_valid(self, action: int) -> bool:
        raise NotImplementedError

    def __time_proceed(self) -> None:
        self.__cluster_shiftup()
        self.__fill_new_jobs()

    def __cluster_shiftup(self) -> None:
        clusters = self.state[0]

        def shiftup(cluster):
            return cluster[:self.resource_size] + ([0] * self.resource_size)
        clusters = [shiftup(cluster) for cluster in clusters]

        self.state[0] = clusters

    def __fill_new_jobs(self) -> None:
        slots = self.state[1]

        for index, slot in enumerate(slots):
            if not self.remaining_jobs:
                break
            if any(cell != 0 for cell in slot):
                continue
            slots[index] = self.remaining_jobs.pop(0)

        self.state[1] = slots

    # TODO: implement
    def __schedule(self, action: int) -> None:
        # TODO: suppose not empty slot for now

        raise NotImplementedError

    # return a position or None
    def __find_fit(self, cluster, job):
        for row in range(self.time_size):
            for col in range(self.resource_size):
                if cluster[row * self.time_size + self.resource_size] != 0:
                    continue
                if self.__try_fit(cluster, job, row, col):
                    return row * self.time_size + self.resource_size
        return None

    def __try_fit(self, cluster, job, row, col) -> bool:
        # TODO: dfs?
        return False

    # TODO: implement
    def __reward(self) -> float:
        raise NotImplementedError
