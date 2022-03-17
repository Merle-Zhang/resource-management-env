from typing import Optional
import gym
import numpy as np

from res_mgmt.envs.res import Res


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

        # TODO: unused action space before reset
        # first M jobs + the null sign
        self.action_space = gym.spaces.Discrete(
            num_job_slot + 1)

        # (number of colors + empty) * (row size * column size)
        cluster_image = gym.spaces.MultiDiscrete(
            [num_job + 1] * (resource_size * time_size))
        job_slot_image = gym.spaces.MultiBinary(resource_size * time_size)

        # TODO: unused observation space
        self.observation_space = gym.spaces.Dict({
            "clusters": gym.spaces.Tuple(tuple(
                [cluster_image] * num_resource_type
            )),
            "slots": gym.spaces.Tuple(tuple(
                [job_slot_image] * num_job_slot
            )),
            "backlog": gym.spaces.Discrete(size_backlog)
        })

        self.res: Res = None
        self.state = None
        self.actions = None

    def step(self, action: int):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        selected_job = self.actions[action]
        if selected_job != None:
            pos = self.res.find_pos(selected_job)
            if pos == -1:
                self.res.time_proceed()
                reward = self.__reward()
            else:
                self.res.schedule(selected_job, pos)
                # TODO: valid when discount factor is 1. Might need to change it later.
                reward = 0

        self.state = self.res.state()
        state = self.state
        reward = reward
        done = self.res.finish()
        info = {}
        return state, reward, done, info

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        self.res = Res(
            num_resource_type=self.num_resource_type,
            time_size=self.time_size,
            resource_size=self.resource_size,
            num_job_slot=self.num_job_slot,
        )
        self.res.add_jobs(self.jobs)
        self.res.time_proceed()
        self.actions = self.res.actions()
        # TODO: 1. can we change action space 2. Is this action assignment ok, e.g. the none action is at the end of space
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.state = self.res.state()
        # print(concat.shape)
        return self.state

    def __reward(self) -> float:
        return (-1.0 / self.res.durations()).sum()
