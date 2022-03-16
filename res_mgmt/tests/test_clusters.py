import unittest
import numpy as np


from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.job import Job
from res_mgmt.tests.utils import meta_from_durations


class TestClustersInit(unittest.TestCase):

    def test_normal(self):
        num_resource_type = 2
        time_size = 5
        resource_size = 3
        meta = {1: Job()}
        clusters = Clusters(num_resource_type, time_size, resource_size, meta)

        self.assertEqual(clusters.state.shape,
                         (num_resource_type, time_size, resource_size))
        np.testing.assert_allclose(clusters.state, _EMPTY_CELL)
        self.assertTrue(clusters.meta is meta)


class TestClustersTimeProceed(unittest.TestCase):

    def test_normal(self):
        clusters = Clusters.fromConfig()

        old_state = np.array(
            [[[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]],

             [[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]],

             [[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]]]
        )

        new_state = np.array(
            [[[3, 4, 5],
              [6, 7, 8],
              [-1, -1, -1]],

             [[3, 4, 5],
              [6, 7, 8],
              [-1, -1, -1]],

             [[3, 4, 5],
              [6, 7, 8],
              [-1, -1, -1]], ]
        )

        clusters.state = old_state
        clusters.time_proceed()

        np.testing.assert_allclose(clusters.state, new_state)


class TestClustersDurations(unittest.TestCase):

    def test_normal(self):
        clusters = Clusters.fromConfig()

        clusters.state = np.array(
            [[[3, 5, 5],
              [6, 7, 7],
              [-1, -1, -1]],

             [[4, 4, 5],
              [8, 7, 8],
              [-1, -1, -1]],

             [[3, 4, 5],
              [6, 7, 8],
              [-1, 3, -1]]]
        )

        clusters.meta = meta_from_durations({
            3: 8,
            4: 6,
            5: 7,
            6: 2,
            7: 1,
            8: 2,
        })

        expected = [8, 6, 7, 2, 1, 2]

        np.testing.assert_allclose(clusters.durations(), expected)


if __name__ == '__main__':
    unittest.main()
