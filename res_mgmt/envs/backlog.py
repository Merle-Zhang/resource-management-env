from collections import deque
import numpy as np
import numpy.typing as npt

from res_mgmt.envs.job import Job

JobImage = npt.NDArray[np.bool]


class Backlog:
    """The backlog containing the ramaining jobs after the first num_job_slot jobs.

    Attributes:
        state: The number of job in backlog.
        queue: The queue of `(id, job)`s.
    """

    def __init__(self) -> None:
        self.queue = deque[tuple[int, JobImage]]()
        self.state = 0

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

    def get(self) -> tuple[int, JobImage]:
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
