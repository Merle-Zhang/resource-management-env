from hashlib import new
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
        max_num_job = 10**3
        new_job_rate=0.7
        res = Res(
            num_resource_type=num_resource_type,
            num_job_slot=num_job_slot,
            time_size=time_size,
            resource_size=resource_size,
            max_num_job=max_num_job,
            new_job_rate=new_job_rate,
        )

        self.assertEqual(res.clusters.state.shape,
                         (num_resource_type, time_size, resource_size))
        self.assertEqual(res.job_slots.state.shape, (
            num_job_slot,
            num_resource_type,
            time_size,
            resource_size,
        ))
        self.assertEqual(res.max_num_job, max_num_job)
        self.assertTrue(res.meta is res.clusters.meta)
        self.assertTrue(res.meta is res.job_slots.meta)
        self.assertTrue(res.meta is res.backlog.meta)
        np.testing.assert_allclose(res.empty_cells_cluster, [
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
        ])



class TestResActions(unittest.TestCase):

    def test_normal(self):
        res = Res.fromConfig()
        res.job_slots.jobs = np.array(
            [2, 3, 5, _EMPTY_CELL, 4, 6, _EMPTY_CELL, 8])

        expected = [0, 1, 2, 4, 5, 7, None]
        actural = res.actions()

        self.assertEqual(expected, actural)


class TestResFindPos(unittest.TestCase):

    def test_normal(self):
        res = Res.fromConfig()
        res.meta = {2: Job(
            requirements=np.array([
                [2, 2, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ]),
            time_max=2,
        )}
        res.empty_cells_cluster = np.array([
            [0, 0, 1, 3, 3],
            [1, 0, 2, 3, 3],
        ])
        expected = 3
        actural = res.find_pos(2)

        self.assertEqual(expected, actural)

    def test_cannot_find(self):
        res = Res.fromConfig()
        res.meta = {2: Job(
            requirements=np.array([
                [3, 3, 3, 0, 0],
                [1, 1, 1, 0, 0],
            ]),
            time_max=3,
        )}
        res.empty_cells_cluster = np.array([
            [0, 0, 1, 3, 3],
            [1, 0, 2, 3, 3],
        ])
        expected = -1
        actural = res.find_pos(2)

        self.assertEqual(expected, actural)

class TestResSchedule(unittest.TestCase):

    def test_normal(self):
        res = Res.fromConfig()
        res.meta = {2: Job(
            requirements=np.array([
                [2, 2, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ]),
            time_max=2,
        )}
        res.empty_cells_cluster = np.array([
            [0, 0, 1, 3, 3],
            [1, 0, 2, 3, 3],
        ])
        new_empty_cells_cluster = np.array([
            [0, 0, 1, 1, 1],
            [1, 0, 2, 2, 2],
        ])
        res.job_slots.jobs = np.array([0, 2, 3])
        new_jobs = np.array([0, _EMPTY_CELL, 3])
        res.clusters.state = np.array(
            [[[ 1,  1,  1],
              [ 1,  1,  1],
              [ 1,  1, -1],
              [-1, -1, -1],
              [-1, -1, -1]],

             [[ 1,  1, -1],
              [ 1,  1,  1],
              [ 1, -1, -1],
              [-1, -1, -1],
              [-1, -1, -1]]]
        )
        new_state = np.array(
            [[[ 1,  1,  1],
              [ 1,  1,  1],
              [ 1,  1, -1],
              [ 2,  2, -1],
              [ 2,  2, -1]],

             [[ 1,  1, -1],
              [ 1,  1,  1],
              [ 1, -1, -1],
              [ 2, -1, -1],
              [ 2, -1, -1]]]
        )
        expected = True
        actural = res.schedule(2, 3)

        self.assertEqual(expected, actural)
        np.testing.assert_allclose(new_empty_cells_cluster, res.empty_cells_cluster)
        np.testing.assert_allclose(new_jobs, res.job_slots.jobs)
        np.testing.assert_allclose(new_state, res.clusters.state)


class TestResFinish(unittest.TestCase):

    def test_normal(self):
        res = Res.fromConfig()
        self.assertTrue(res.finish())


if __name__ == '__main__':
    unittest.main()
