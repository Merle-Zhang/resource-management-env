from collections import deque
import unittest
import numpy as np

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.job_slots import JobSlots
from res_mgmt.tests.utils import meta_from_durations


class TestJobSlotsInit(unittest.TestCase):

    def test_normal(self):
        num_resource_type = 2
        num_job_slot = 3
        time_size = 5
        resource_size = 3
        meta = {1: None}
        job_slots = JobSlots(
            num_job_slot,
            num_resource_type,
            time_size,
            resource_size,
            meta,
        )

        self.assertEqual(job_slots.state.shape, (
            num_job_slot,
            num_resource_type,
            time_size,
            resource_size,
        ))
        np.testing.assert_allclose(job_slots.state, False)
        np.testing.assert_allclose(job_slots.jobs, _EMPTY_CELL)
        self.assertTrue(job_slots.meta is meta)


class TestJobSlotsRefill(unittest.TestCase):

    def test_normal(self):
        job_slots = JobSlots.fromConfig()
        backlog = Backlog(meta=None)
        backlog.state = 1
        backlog.queue = deque([
            (
                4,
                np.array(
                 [[[False,  True, False],
                   [False, False, False],
                   [False,  True, False],
                   [ True, False,  True],
                   [ True, False,  True]],

                  [[ True, False,  True],
                   [ True,  True, False],
                   [ True, False,  True],
                   [False, False, False],
                   [False, False,  True]]],
                )
            )
        ])

        old_jobs = [1, _EMPTY_CELL, 3]
        old_state = np.array(
            [[[[ True, False,  True],
               [ True,  True,  True],
               [ True,  True,  True],
               [ True, False, False],
               [False, False, False]],

              [[ True, False,  True],
               [False, False, False],
               [ True,  True, False],
               [ True,  True, False],
               [False, False,  True]]],


             [[[ True,  True,  True],
               [ True,  True,  True],
               [False,  True, False],
               [False, False, False],
               [False, False,  True]],

              [[False,  True,  True],
               [ True,  True, False],
               [ True,  True, False],
               [ True,  True,  True],
               [ True,  True,  True]]],


             [[[False, False, False],
               [ True,  True,  True],
               [ True,  True,  True],
               [False,  True,  True],
               [ True,  True,  True]],

              [[ True,  True, False],
               [ True,  True,  True],
               [ True, False, False],
               [False, False,  True],
               [ True, False,  True]]]],
        )

        new_jobs = [1, 4, 3]
        new_state = np.array(
            [[[[ True, False,  True],
               [ True,  True,  True],
               [ True,  True,  True],
               [ True, False, False],
               [False, False, False]],

              [[ True, False,  True],
               [False, False, False],
               [ True,  True, False],
               [ True,  True, False],
               [False, False,  True]]],


             [[[False,  True, False],
               [False, False, False],
               [False,  True, False],
               [ True, False,  True],
               [ True, False,  True]],

              [[ True, False,  True],
               [ True,  True, False],
               [ True, False,  True],
               [False, False, False],
               [False, False,  True]]],


             [[[False, False, False],
               [ True,  True,  True],
               [ True,  True,  True],
               [False,  True,  True],
               [ True,  True,  True]],

              [[ True,  True, False],
               [ True,  True,  True],
               [ True, False, False],
               [False, False,  True],
               [ True, False,  True]]]],
        )

        job_slots.state = old_state
        job_slots.jobs = old_jobs

        job_slots.refill(backlog)

        np.testing.assert_allclose(job_slots.state, new_state)
        np.testing.assert_allclose(job_slots.jobs, new_jobs)


class TestJobSlotsDurations(unittest.TestCase):

    def test_normal(self):
        job_slots = JobSlots.fromConfig()

        job_slots.jobs = np.array([1, _EMPTY_CELL, 3, 4, _EMPTY_CELL, 6])

        job_slots.meta = meta_from_durations({
            1: 1,
            2: 4,
            3: 8,
            4: 6,
            5: 7,
            6: 2,
            7: 1,
            8: 2,
        })

        expected = (
            1 +
            8 +
            6 +
            2
        )

        self.assertEqual(job_slots.durations(), expected)


if __name__ == '__main__':
    unittest.main()
