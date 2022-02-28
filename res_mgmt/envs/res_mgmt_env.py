from typing import Optional
import gym
import numpy as np

# TODO: add type hint
# TODO: add unit test
# TODO: split env from gym env class
# TODO: If each step doesn't have immediately reward, 
#       should it be merged into a single step by selecting from one of the combinations?
#       Or just return zero reward?


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
    #       Update: let's remove it from the list and also update the self.action_space so we can use action_space.sample()
    def step(self, action: int):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        if action == self.num_job_slot or not self.__action_valid(action):
            self.__time_proceed()
            reward = self.__reward()
        else:
            self.__schedule(action)
            reward = 0 # TODO: valid when discount factor is 1. Might need to change it later.

        state = np.array(self.state, dtype=np.uintc)
        reward = reward
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

    # TODO: merge __action_valid with __schedule
    def __schedule(self, action: int) -> None:
        # TODO: suppose not empty slot for now
        job = self.state[1][action]
        for type in range(self.num_resource_type):
            assert not self.__empty(job[type]) # assert not empty for selected action

            location = self.__find_fit(self.state[0][type], job[type])
            if location == None:
                raise ValueError("cannot be scheduled") # TODO: raise error or return false?

            color = max(self.state[0][type]) + 1
            for time in range(self.time_size):
                if job[time * self.time_size + 0] == 0:
                    job_time_size = time
            for resource in range(self.resource_size):
                if job[0 * self.time_size + resource] == 0:
                    job_resource_size = resource
            
            # TODO: need some rewrite, only pass size of rec job



    # return a position or None
    # TODO: assume rectangular for now. Might need to change later, e.g. more resource at the start and less later
    def __find_fit(self, cluster, job):
        for time in range(self.time_size):
            if job[time * self.time_size + 0] == 0:
                job_time_size = time
        for resource in range(self.resource_size):
            if job[0 * self.time_size + resource] == 0:
                job_resource_size = resource
        for time in range(self.time_size):
            for resource in range(self.resource_size):
                if cluster[time * self.time_size + resource] != 0:
                    continue
                if self.__try_fit(cluster, time, resource, job_time_size, job_resource_size):
                    return time * self.time_size + self.resource_size
        return None

    def __try_fit(self, cluster, row, col, job_time_size, job_resource_size) -> bool:
        for time in range(job_time_size):
            if row + time >= self.time_size or self.__cell(cluster, col, row + time) != 0:
                return False
        for resource in range(job_resource_size):
            if col + resource >= self.resource_size or self.__cell(cluster, col + resource, row) != 0:
                return False
        return True

    def __reward(self) -> float:
        return sum(-1 / self.__duration(job) for job in self.state[1] + self.remaining_jobs)

    # TODO: check top left cell for now. Might need to change it later.
    def __empty(self, slot) -> bool:
        return slot[0] == 0

    def __cell(self, block, resource, time):
        return block[time * self.time_size + resource]

    # TODO: implement
    def __duration(self, job) -> int:
        raise NotImplementedError
