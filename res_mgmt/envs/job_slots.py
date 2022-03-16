import numpy as np

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.config import _EMPTY_CELL, Config, _DEFAULT_CONFIG


class JobSlots:
    """The job slots that contains all the jobs to be scheduled.

    Attributes:
        state: 
            The job slots "image". Numpy array with shape 
            (num_job_slot, num_resource_type, time_size, resource_size)
        jobs: 
            Array of length num_job_slot indicating the job id 
            of the corresponding job in the slots.
        duration_map: 
            Map from the job id to its duration.
    """

    def __init__(
        self,
        num_job_slot: int,       # first M jobs
        num_resource_type: int,  # d resource types
        time_size: int,          # column
        resource_size: int,      # row
    ) -> None:
        shape = (
            num_job_slot,
            num_resource_type,
            time_size,
            resource_size,
        )
        self.state = np.full(shape, False, dtype=np.bool_)
        self.jobs = np.full(num_job_slot, _EMPTY_CELL, dtype=int)
        self.duration_map = {}  # job_index -> duration

    @classmethod
    def fromConfig(cls, config: Config = _DEFAULT_CONFIG):
        """Create a JobSlots from config.

        Args:
            config: The config. If not specified, the default config will be used.
        """
        return cls(
            num_resource_type=config["num_resource_type"],
            num_job_slot=config["num_job_slot"],
            time_size=config["time_size"],
            resource_size=config["resource_size"],
        )

    def refill(self, backlog: Backlog) -> None:
        """Refill the empty slots with top jobs from backlog.
        """
        for index, job in enumerate(self.jobs):
            if not backlog.queue:
                break
            if job != _EMPTY_CELL:
                continue
            id, image = backlog.get()
            self.jobs[index] = id
            self.state[index] = image

    def durations(self) -> int:
        """The sum of the durations of all jobs in job slots.

        Returns:
            Sum of the durations of all jobs in job slots.
        """
        jobs_in_cluster = np.delete(
            self.jobs, np.where(self.jobs == _EMPTY_CELL))
        return np.vectorize(self.duration_map.get)(jobs_in_cluster).sum()
