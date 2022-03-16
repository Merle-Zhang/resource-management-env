from collections import deque
import unittest
import numpy as np

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.job import Job
from res_mgmt.tests.utils import meta_from_durations


class TestBacklogInit(unittest.TestCase):

    def test_normal(self):
        meta = {1: Job()}
        backlog = Backlog(meta=meta)
        self.assertEqual(backlog.state, 0)
        self.assertFalse(backlog.queue)
        self.assertTrue(backlog.meta is meta)


class TestBacklogAdd(unittest.TestCase):

    def test_normal(self):
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

        backlog.add(
            Job(id=5), 
            np.array(
                [[[ True, False,  True],
                  [ True,  True,  True],
                  [ True,  True,  True],
                  [ True, False, False],
                  [False, False, False]],

                 [[ True, False,  True],
                  [False, False, False],
                  [ True,  True, False],
                  [ True,  True, False],
                  [False, False,  True]]],
            ),
        )

        new_state = 2
        new_queue = [
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
                ),
            ),
            (
                5,
                np.array(
                    [[[ True, False,  True],
                      [ True,  True,  True],
                      [ True,  True,  True],
                      [ True, False, False],
                      [False, False, False]],

                     [[ True, False,  True],
                      [False, False, False],
                      [ True,  True, False],
                      [ True,  True, False],
                      [False, False,  True]]],
                ),
            ),
        ]

        self.assertEqual(backlog.state, new_state)
        def id(queue):
            return [entry[0] for entry in queue]
        def image(queue):
            return [entry[1] for entry in queue]
        np.testing.assert_allclose(id(backlog.queue), id(new_queue))
        np.testing.assert_allclose(image(backlog.queue), image(new_queue))


class TestBacklogGet(unittest.TestCase):

    def test_normal(self):
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

        expect_id = 4
        expect_image = np.array(
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

        actual_id, actual_image = backlog.get()
        self.assertEqual(actual_id, expect_id)
        np.testing.assert_allclose(actual_image, expect_image)


class TestBacklogDurations(unittest.TestCase):

    def test_normal(self):
        meta = meta_from_durations({
            1: 1,
            2: 4,
            3: 8,
            4: 6,
            5: 7,
            6: 2,
            7: 1,
            8: 2,
        })
        backlog = Backlog(meta=meta)
        backlog.queue = deque([
            (1, None),
            (4, None),
            (3, None),
            (6, None),
            (8, None),
        ])

        expected = [1, 6, 8, 2, 2]

        np.testing.assert_allclose(backlog.durations(), expected)


if __name__ == '__main__':
    unittest.main()
