from collections import deque
import numpy as np
import numpy.typing as npt
from typing import Dict, Deque, Tuple

from res_mgmt.envs.job import Job
from res_mgmt.envs.config import _EMPTY_CELL, Config, _DEFAULT_CONFIG

JobImage = npt.NDArray[np.bool_]


class Backlog:
    """The backlog containing the ramaining jobs after the first num_job_slot jobs.

    Attributes:
        state: The number of job in backlog.
        queue: The queue of `(id, job)`s.
        meta: Map from the job id to its metadata in Job.
    """

    def __init__(
        self,
        meta: Dict[int, Job],    # meta data of jobs {job_index -> job_meta}
        generator,  # generator for new jobs
        new_job_rate,  # job arrival rate
    ) -> None:
        self.queue: Deque[Tuple[int, JobImage]] = deque()
        self.state = 0
        self.meta = meta
        self.generator = generator
        self.new_job_rate = new_job_rate

    @classmethod
    def fromConfig(cls, config: Config = _DEFAULT_CONFIG):
        """Create a JobSlots from config.

        Args:
            config: The config. If not specified, the default config will be used.
        """
        return cls(
            meta=config["meta"],
            generator=config["generator"],  # generator for new jobs
            new_job_rate=config["new_job_rate"],
        )

    def add(self, job: Job, image: JobImage) -> None:
        """Add a job to the right side of the queue of backlog.

        Args:
            job: 
                The job to be added.
            image: 
                The job image, a numpy bool array with shape 
                (num_resource_type, time_size, resource_size).
        """
        self.queue.append((job.id, image))
        self.state = min(60, len(self.queue))

    def get(self) -> Tuple[int, JobImage]:
        """Get a job from the left side of the queue of backlog.

        The caller should check empty before getting the job, otherwise will raise exception.

        To check empty: 
        ```python
        if backlog.queue:
            job = backlog.get()
        ```

        Returns:
            The integer id, and the first job from the left 
            side of the queue. The job is a numpy bool array with shape 
            (num_resource_type, time_size, resource_size).

        Raises:
            IndexError: An error occurred poping from an empty backlog.
        """
        # if self.queue and self.state > 0:
        result = self.queue.popleft()
        self.state = min(60, len(self.queue))
        return result

    def durations(self) -> npt.NDArray[np.int_]:
        """The sum of the durations of all jobs in backlog.

        Returns:
            List of durations of all jobs in backlog.
        """
        return np.array([self.meta[id].duration for id, _ in self.queue])

    def time_proceed(self) -> None:
        """New job might arrive according to the new job rate.
        """
        if np.random.rand() < self.new_job_rate:
            image = next(self.generator)
            id = np.random.randint(100)
            job = Job.fromImage(id, image)
            self.meta[id] = job
            self.add(job, image)
