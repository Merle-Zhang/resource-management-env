import numpy as np

from res_mgmt.envs.config import _EMPTY_CELL, Config, _DEFAULT_CONFIG


class Clusters:
    """The clusters that contains all the scheduled jobs.

    Attributes:
        state: The cluster "image". Numpy array with shape (num_resource_type, time_size, resource_size)
        duration_map: Map from the job id to its duration.
    """

    def __init__(
        self,
        num_resource_type: int,  # d resource types
        time_size: int,          # column
        resource_size: int,      # row
    ) -> None:
        shape = (num_resource_type, time_size, resource_size)
        self.state = np.full(shape, _EMPTY_CELL, dtype=int)
        self.duration_map = {}  # job_index -> duration

    @classmethod
    def fromConfig(cls, config: Config = _DEFAULT_CONFIG):
        """Create a cluster from config.

        Args:
            config: The config. If not specified, the default config will be used.
        """
        return cls(
            num_resource_type=config["num_resource_type"],
            time_size=config["time_size"],
            resource_size=config["resource_size"],
        )

    def time_proceed(self) -> None:
        """Shift the cluster image up by one row.
        """
        shape = list(self.state.shape)
        shape[1] = 1
        new_empty_row = np.full(shape, _EMPTY_CELL, dtype=int)
        np.concatenate(
            (self.state[:, 1:, :], new_empty_row),
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
        jobs_in_cluster = np.delete(
            jobs_in_cluster, np.where(jobs_in_cluster == _EMPTY_CELL))
        return np.vectorize(self.duration_map.get)(jobs_in_cluster).sum()
