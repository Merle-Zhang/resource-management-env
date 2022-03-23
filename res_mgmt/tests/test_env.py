from collections import deque
import unittest
import numpy as np

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.job import Job
from res_mgmt.envs.res import Res
from res_mgmt.envs.res_mgmt_env import ResMgmtEnv
from res_mgmt.tests.utils import meta_from_durations


class TestEnv(unittest.TestCase):

    def test_normal(self):
        jobs = np.array(
            [[[[ True,  True, False],
               [ True, False,  True],
               [ True, False,  True],
               [False, False,  True],
               [False,  True,  True]],

              [[False, False,  True],
               [ True,  True,  True],
               [False,  True,  True],
               [False, False, False],
               [False,  True, False]]],


             [[[False, False,  True],
               [False,  True, False],
               [ True,  True,  True],
               [False, False,  True],
               [False, False, False]],

              [[False, False,  True],
               [ True, False, False],
               [ True, False,  True],
               [ True,  True, False],
               [ True, False, False]]],


             [[[ True, False, False],
               [False, False,  True],
               [ True, False,  True],
               [ True,  True, False],
               [False, False,  True]],

              [[ True,  True,  True],
               [ True, False,  True],
               [ True,  True,  True],
               [False, False,  True],
               [False, False, False]]],


             [[[False, False,  True],
               [ True,  True, False],
               [False,  True, False],
               [False,  True,  True],
               [ True, False, False]],

              [[False,  True, False],
               [False,  True, False],
               [ True,  True, False],
               [False, False, False],
               [ True, False,  True]]],


             [[[ True, False,  True],
               [False,  True,  True],
               [False,  True,  True],
               [ True, False, False],
               [ True, False,  True]],

              [[ True,  True, False],
               [False,  True,  True],
               [False,  True,  True],
               [ True, False,  True],
               [ True,  True,  True]]]]
        )
        env = ResMgmtEnv(
            num_resource_type=2,
            resource_size=3,
            time_size=5,
            num_job_slot=3,
            max_num_job=10**3,
        )

        # nothing to test at this stage

        env.res = Res(
            num_resource_type=env.num_resource_type,
            time_size=env.time_size,
            resource_size=env.resource_size,
            num_job_slot=env.num_job_slot,
            max_num_job=env.max_num_job,
        )
        env.jobs = jobs
        env.res.add_jobs(env.jobs)
        env.res.time_proceed()
        env.state = env.res.state()

        self.assertTrue((env.res.job_slots.jobs != _EMPTY_CELL).all())
        self.assertEqual(len(env.res.backlog.queue), 2)

        self.assertTrue(env.action_space.contains(0))
        self.assertTrue(env.action_space.contains(1))
        self.assertTrue(env.action_space.contains(2))
        self.assertTrue(env.action_space.contains(3))
        self.assertFalse(env.action_space.contains(4))
        self.assertFalse(env.action_space.contains(-1))

        state, reward, done, info = env.step(2)

        self.assertEqual(env.res.job_slots.jobs[1], _EMPTY_CELL)
        self.assertEqual(reward, 0)
        self.assertFalse(done)


if __name__ == '__main__':
    unittest.main()
