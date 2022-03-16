from collections import deque
import unittest
import numpy as np

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.job import Job
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
            num_job=None,
            size_backlog=None,
            jobs=jobs,
        )

        # nothing to test at this stage

        env.reset()
        
        self.assertTrue((env.res.job_slots.jobs != _EMPTY_CELL).all())
        self.assertEqual(len(env.res.backlog.queue), 2)
        np.testing.assert_allclose(env.actions, [0, 1, 2, None])

        # env.action_space

        state, reward, done, info = env.step(1)

        self.assertEqual(env.res.job_slots.jobs[1], _EMPTY_CELL)
        self.assertEqual(reward, 0)
        self.assertFalse(done)



if __name__ == '__main__':
    unittest.main()
