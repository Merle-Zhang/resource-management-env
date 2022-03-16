import numpy as np
import numpy.typing as npt

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.config import _EMPTY_CELL, Config, _DEFAULT_CONFIG
from res_mgmt.envs.job import Job


class JobSlots:
    """The job slots that contains all the jobs to be scheduled.

    Attributes:
        state: 
            The job slots "image". Numpy array with shape 
            (num_job_slot, num_resource_type, time_size, resource_size)
        jobs: 
            Array of length num_job_slot indicating the job id 
            of the corresponding job in the slots.
        meta: 
            Map from the job id to its metadata in Job.
    """

    def __init__(
        self,
        num_job_slot: int,       # first M jobs
        num_resource_type: int,  # d resource types
        time_size: int,          # column
        resource_size: int,      # row
        meta: dict[int, Job],    # meta data of jobs {job_index -> job_meta}
    ) -> None:
        shape = (
            num_job_slot,
            num_resource_type,
            time_size,
            resource_size,
        )
        self.state = np.full(shape, False, dtype=np.bool_)
        self.jobs = np.full(num_job_slot, _EMPTY_CELL, dtype=int)
        self.meta = meta

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
            meta={},
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

    def durations(self) -> npt.NDArray[np.int_]:
        """The sum of the durations of all jobs in job slots.

        Returns:
            List of the durations of all jobs in job slots.
        """
        jobs_in_cluster = np.delete(
            self.jobs, np.where(self.jobs == _EMPTY_CELL))

        def get_duration(id: int):
            return self.meta[id].duration
        return np.vectorize(get_duration)(jobs_in_cluster)
