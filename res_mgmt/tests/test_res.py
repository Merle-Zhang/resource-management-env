import unittest
import numpy as np

from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.job import Job
from res_mgmt.envs.res import Res
from res_mgmt.tests.utils import meta_from_durations


class TestResInit(unittest.TestCase):

    def test_normal(self):
        num_resource_type = 2
        num_job_slot = 3
        time_size = 5
        resource_size = 3
        res = Res(
            num_resource_type=num_resource_type,
            num_job_slot=num_job_slot,
            time_size=time_size,
            resource_size=resource_size,
        )

        self.assertEqual(res.clusters.state.shape,
                         (num_resource_type, time_size, resource_size))
        self.assertEqual(res.job_slots.state.shape, (
            num_job_slot,
            num_resource_type,
            time_size,
            resource_size,
        ))
        self.assertTrue(res.meta is res.clusters.meta)
        self.assertTrue(res.meta is res.job_slots.meta)
        self.assertTrue(res.meta is res.backlog.meta)


class TestResActions(unittest.TestCase):

    def test_normal(self):
        res = Res.fromConfig()
        res.job_slots.jobs = np.array(
            [2, 3, 5, _EMPTY_CELL, 4, 6, _EMPTY_CELL, 8])

        expected = [0, 1, 2, 4, 5, 7, None]
        actural = res.actions()

        self.assertEqual(expected, actural)


class TestResSchedule(unittest.TestCase):

    def test_normal(self):
        res = Res.fromConfig()
        res.schedule(None)


if __name__ == '__main__':
    unittest.main()
