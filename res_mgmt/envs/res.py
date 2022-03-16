from typing import Optional

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.config import Config, _DEFAULT_CONFIG
from res_mgmt.envs.job import Job
from res_mgmt.envs.job_slots import JobSlots


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
        time_size: int,  # column
        resource_size: int,  # row
        num_job_slot: int,  # first M jobs
    ) -> None:
        self.clusters = Clusters(
            num_resource_type=num_resource_type,
            time_size=time_size,
            resource_size=resource_size,
        )
        self.job_slots = JobSlots(
            num_resource_type=num_resource_type,
            num_job_slot=num_job_slot,
            time_size=time_size,
            resource_size=resource_size,
        )
        self.backlog = Backlog()
        self.job_slots.refill(self.backlog)

    @classmethod
    def fromConfig(cls, config: Config = _DEFAULT_CONFIG):
        """Create a res from config.

        Args:
            config: The config. If not specified, the default config will be used.
        """
        return cls(
            num_resource_type=config["num_resource_type"],
            num_job_slot=config["num_job_slot"],
            time_size=config["time_size"],
            resource_size=config["resource_size"],
        )

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

    def state(self) -> list:
        """The state (image) of clusters, job slots, and backlog.

        Returns:
            A list of [clusters_state, job_slots_state, backlog_state].
        """
        return [self.clusters.state, self.job_slots.state, self.backlog.state]
