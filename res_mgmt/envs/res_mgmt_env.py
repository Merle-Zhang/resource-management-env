from typing import Optional
import gym
import numpy as np
import pygame

from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.generator import generate_jobs
from res_mgmt.envs.render import render
from res_mgmt.envs.res import Res


class ResMgmtEnv(gym.Env):
    def __init__(
        self,
        num_resource_type: int,  # d resource types
        resource_size: int,  # row
        time_size: int,  # column
        num_job_slot: int,  # first M jobs
        max_num_job: int,
        new_job_rate: float,
    ):
        self.num_resource_type = num_resource_type
        self.resource_size = resource_size
        self.time_size = time_size
        self.num_job_slot = num_job_slot
        self.max_num_job = max_num_job
        self.new_job_rate = new_job_rate

        # first M jobs + the null sign
        self.action_space = gym.spaces.Discrete(
            num_job_slot + 1)

        cluster_obs = (
            [resource_size + 1] *
            (num_resource_type * time_size)
        )
        job_slots_obs = (
            [resource_size + 1] *
            (num_job_slot * num_resource_type * time_size)
        )
        backlog_obs = [max_num_job + 1]
        obs = cluster_obs + job_slots_obs + backlog_obs
        self.observation_space = gym.spaces.MultiDiscrete(obs)

        self.res: Res = None
        self.state = None
        self.screen = None
        self.jobs = None
        self.stepcount = None

    def step(self, action: int):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # print("LOG:", f"step ({self.stepcount})")
        # print("LOG:", f"Chose action ({action})")
        # if step(0) then choose none and step forward
        # else step(N) then choose the job on N-1 (Nth) slot
        action -= 1

        reward = None
        if action != -1:
            job_id = self.res.job_slots.jobs[action]
            # print("LOG:", f"Job ID ({job_id})")
            if job_id != _EMPTY_CELL:
                pos = self.res.find_pos(job_id)
                # print("LOG:", f"Pos ({pos})")
                if pos != -1:
                    self.res.schedule(job_id, pos)
                    reward = 0
                    # print("LOG:", f"reward ({reward})")
        else:
            self.res.time_proceed()
            reward = self.__reward()  # TODO: reward before proceed?
            # print("LOG:", f"Time proceed, reward ({reward})")
        if reward == None:
            # self.res.time_proceed()
            reward = self.__reward() - 10
            # print("LOG:", f"No time proceed, reward ({reward})")

        self.state = self.res.state()
        state = self.state
        reward = reward
        done = self.stepcount >= 50
        info = {}
        # self.my_render(f"render/{self.stepcount}.png")
        # print(self.state)
        # print("======================================")
        self.stepcount += 1
        return state, reward, done, info

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed) # not supported in gym 0.19.0

        self.res = Res(
            num_resource_type=self.num_resource_type,
            time_size=self.time_size,
            resource_size=self.resource_size,
            num_job_slot=self.num_job_slot,
            max_num_job=self.max_num_job,
            new_job_rate=self.new_job_rate,
        )
        self.res.time_proceed()
        self.state = self.res.state()
        self.stepcount = 0
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
