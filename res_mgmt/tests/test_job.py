import unittest
import numpy as np

from res_mgmt.envs.job import Job


class TestJobFromImage(unittest.TestCase):

    def test_normal(self):
        image = np.array(
            [[[ True,  True,  True],
              [ True, False, False],
              [ True, False,  True],
              [ True, False, False],
              [False, False, False]],

             [[ True, False,  True],
              [ True, False,  True],
              [ True,  True,  True],
              [ True,  True, False],
              [False, False, False]]]
        )
        job = Job.fromImage(2, image)

        self.assertEqual(job.duration, 4)
        np.testing.assert_allclose(job.requirements, np.array(
            [[3, 1, 2, 1, 0],
             [2, 2, 3, 2, 0]]
        ))
        self.assertEqual(job.time_max, 4)


if __name__ == '__main__':
    unittest.main()
