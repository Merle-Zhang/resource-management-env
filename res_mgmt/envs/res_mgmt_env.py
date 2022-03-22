from typing import Optional
import gym
import numpy as np
import pygame

from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.render import render
from res_mgmt.envs.res import Res


class ResMgmtEnv(gym.Env):
    def __init__(
        self,
        num_resource_type: int,  # d resource types
        resource_size: int,  # row
        time_size: int,  # column
        num_job_slot: int,  # first M jobs
        # number of colors, TODO: num_job needed or just len(jobs)?
        max_num_job: int,
        size_backlog: int,  # TODO: size_backlog needed or just infinite?
        jobs: list  # shape(num_job, num_resource_type)
    ):
        self.num_resource_type = num_resource_type
        self.resource_size = resource_size
        self.time_size = time_size
        self.num_job_slot = num_job_slot
        self.max_num_job = max_num_job
        self.jobs = jobs

        # TODO: unused action space before reset
        # first M jobs + the null sign
        self.action_space = gym.spaces.Discrete(
            num_job_slot + 1)

        cluster_obs = [max_num_job + 1] * (num_resource_type * time_size * resource_size)
        job_slots_obs = [2] * (num_job_slot * num_resource_type * time_size * resource_size)
        backlog_obs = [max_num_job]
        obs = cluster_obs + job_slots_obs + backlog_obs
        self.observation_space = gym.spaces.MultiDiscrete(obs)

        self.res: Res = None
        self.state = None
        self.screen = None

    def step(self, action: int):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # if step(0) then choose none and step forward
        # else step(N) then choose the job on N-1 (Nth) slot
        action -= 1

        reward = None
        if action != -1:
            job_id = self.res.job_slots.jobs[action]
            if job_id != _EMPTY_CELL:
                pos = self.res.find_pos(job_id)
                if pos != -1:
                    self.res.schedule(job_id, pos)
                    reward = 0
        if reward != 0:
            self.res.time_proceed()
            reward = self.__reward() # TODO: reward before proceed?

        self.state = self.res.state()
        state = self.state
        reward = reward
        done = self.res.finish()
        info = {}
        return state, reward, done, info

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed) # not supported in gym 0.19.0

        self.res = Res(
            num_resource_type=self.num_resource_type,
            time_size=self.time_size,
            resource_size=self.resource_size,
            num_job_slot=self.num_job_slot,
            max_num_job=self.max_num_job,
        )
        self.res.add_jobs(self.jobs)
        self.res.time_proceed()
        # TODO: 1. can we change action space 2. Is this action assignment ok, e.g. the none action is at the end of space
        self.state = self.res.state()
        return self.state

    def __reward(self) -> float:
        return (-1.0 / self.res.durations()).sum()

    def my_render(self, filename: str):
        if self.state is None:
            return
        if self.screen is None:
            pygame.display.init()
            pygame.font.init()
        self.screen = render(self.res)
        pygame.image.save(self.screen, filename)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()
            self.screen = None
