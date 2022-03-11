from typing import Optional

from res_mgmt.envs.job import Job
from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.job_slots import JobSlots
from res_mgmt.envs.backlog import Backlog


class Res:
    """The resources containing clusters, job_slots, and backlog.

    Attributes:
        clusters: The clusters containing the schedules jobs.
        job_slots: The job_slots containing the jobs to be scheduled.
        backlog: Unscheduled jobs that's not in job_slots.
    """

    def __init__(
        self,
        num_resource_type: int,  # d resource types
        resource_size: int,  # row
        time_size: int,  # column
        num_job_slot: int,  # first M jobs
        # number of colors, TODO: num_job needed or just len(jobs)?
        num_job: int,
        size_backlog: int,  # TODO: size_backlog needed or just infinite?
        jobs: list  # shape(num_job, num_resource_type)
    ) -> None:
        self.clusters = Clusters()
        self.job_slots = JobSlots()
        self.backlog = Backlog()

    def actions(self) -> list[Optional[int]]:
        """Get available actions.

        Could be one of the unscheduled job from job slots 
        or an null action which means not choosing anything.

        Returns:
            A list of integer or null.
        """
        raise NotImplementedError

    def time_proceed(self) -> None:
        """Proceed to the next timestep.

        Clusters proceed one timestep, and 
        refill job_slots with the jobs in backlog.
        """
        self.clusters.time_proceed()
        self.job_slots.refill(self.backlog)

    def durations(self) -> list[int]:
        """Durations for all jobs in the systems either scheduled or waiting for service.

        Concatenate durations from clusters, job_slots, and backlog.

        TODO: Might be slow. Potential performance improvement.

        Returns:
            A list of durations as int.
        """
        return self.clusters.durations() + self.job_slots.durations() + self.backlog.durations()

    def schedule(self, job: Job) -> bool:
        """Schedule the job in the job slot `job`.

        Schedule the selected job in the first possible timestep in the cluster, 
        (i.e., the first timestep in which the job’s resource requirements can be 
        fully satisfied till completion)s

        Args:
            job: The selected job to be scheduled.

        Returns:
            False if the job cannot be scheduled, i.e., does not "fit".
        """
        raise NotImplementedError
