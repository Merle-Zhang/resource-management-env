import numpy as np

from res_mgmt.envs.config import _EMPTY_CELL


class Clusters:
    """The clusters that contains all the scheduled jobs.

    Attributes:
        state: The cluster "image". Numpy array with shape (num_resource_type, time_size, resource_size)
        durations: Map from the job id to its duration.
    """

    def __init__(
        self,
        num_resource_type: int,  # d resource types
        time_size: int,  # column
        resource_size: int,  # row
    ) -> None:
        shape = (num_resource_type, time_size, resource_size)
        self.state = np.fill(shape, _EMPTY_CELL, dtype=int)
        self.durations = {} # job_index -> duration

    def time_proceed(self) -> None:
        """Shift the cluster image up by one row.
        """
        shape = self.state.shape
        shape[1] = 1
        new_empty_row = np.fill(shape, _EMPTY_CELL, dtype=int)
        np.concatenate(
            (self.state[:,1:,:], new_empty_row), 
            axis=1,
            out=self.state,
        )

    def durations(self) -> int:
        """The sum of the durations of all jobs in clusters.

        Returns:
            Sum of the durations of all jobs in cluster.
        """
        # not_empty_cell_indices = np.where(self.state != _EMPTY_CELL)
        # return np.max(not_empty_cell_indices, axis=1)[1] + 1
        jobs_in_cluster = np.unique(self.state)
        return np.vectorize(self.durations.get)(jobs_in_cluster).sum()