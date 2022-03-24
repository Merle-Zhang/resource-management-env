from collections import deque
import numpy as np
import numpy.typing as npt
from typing import Dict, Deque, Tuple

from res_mgmt.envs.job import Job

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
    ) -> None:
        self.queue: Deque[Tuple[int, JobImage]] = deque()
        self.state = 0
        self.meta = meta

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
        self.state += 1

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
        self.state -= 1
        return result

    def durations(self) -> npt.NDArray[np.int_]:
        """The sum of the durations of all jobs in backlog.

        Returns:
            List of durations of all jobs in backlog.
        """
        return np.array([self.meta[id].duration for id, _ in self.queue])
