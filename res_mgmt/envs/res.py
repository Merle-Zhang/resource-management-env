from typing import Optional
import numpy as np
import numpy.typing as npt

from res_mgmt.envs.backlog import Backlog
from res_mgmt.envs.clusters import Clusters
from res_mgmt.envs.config import _EMPTY_CELL, Config, _DEFAULT_CONFIG
from res_mgmt.envs.job import Job
from res_mgmt.envs.job_slots import JobSlots


class Res:
    """The resources containing clusters, job_slots, and backlog.

    Attributes:
        clusters: The clusters containing the schedules jobs.
        job_slots: The job_slots containing the jobs to be scheduled.
        backlog: Unscheduled jobs that's not in job_slots.
        meta: Metadata of jobs.
        empty_cells_cluster: Empty cells per timestep (row) per resource type.
    """

    def __init__(
        self,
        num_resource_type: int,  # d resource types
        time_size: int,  # column
        resource_size: int,  # row
        num_job_slot: int,  # first M jobs
    ) -> None:
        self.meta: dict[int, Job] = {}
        self.clusters = Clusters(
            num_resource_type=num_resource_type,
            time_size=time_size,
            resource_size=resource_size,
            meta=self.meta,
        )
        self.job_slots = JobSlots(
            num_resource_type=num_resource_type,
            num_job_slot=num_job_slot,
            time_size=time_size,
            resource_size=resource_size,
            meta=self.meta,
        )
        self.backlog = Backlog(meta=self.meta)
        self.empty_cells_cluster = np.full(
            (num_resource_type, time_size), resource_size, dtype=int)

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
            list[Optional[int]]: A list of indices of jobs and a null.
        """
        empty_slots = np.where(self.job_slots.jobs != _EMPTY_CELL)
        assert(len(empty_slots) == 1)
        return list(empty_slots[0]) + [None]

    def time_proceed(self) -> None:
        """Proceed to the next timestep.

        Clusters proceed one timestep, and 
        refill job_slots with the jobs in backlog.
        """
        self.clusters.time_proceed()
        self.job_slots.refill(self.backlog)

    def durations(self) -> npt.NDArray[np.int_]:
        """Durations for all jobs in the systems either scheduled or waiting for service.

        Concatenate durations from clusters, job_slots, and backlog.

        Returns:
            int: Total durations.
        """
        durations = (self.clusters.durations(),
                     self.job_slots.durations(), self.backlog.durations())
        return np.concatenate(durations)

    def find_pos(self, job_id: int) -> int:
        """Find available position given the job id.

        Find the starting position (row/timestep) that fit the job.

        Args:
            job_id (int): Id of the job.

        Returns:
            int: [0 - time_size] if found available position, otherwise -1.
        """
        job_meta = self.meta[job_id]
        empty_cells_cluster = self.empty_cells_cluster
        time_size = self.clusters.state.shape[1]
        num_resource_type = self.clusters.state.shape[0]
        req = job_meta.requirements
        index_time = 0
        start_index = 0

        def try_current_time(cluster_time, job_time):
            for resource_type in range(num_resource_type):
                if empty_cells_cluster[resource_type, cluster_time] < req[resource_type, job_time]:
                    return False
            return True

        for time in range(time_size):
            if index_time >= job_meta.time_max:
                break
            if try_current_time(time, index_time):
                index_time += 1
            else:
                index_time = 0
                start_index = time + 1

        if index_time < job_meta.time_max:
            return -1
        return start_index

    def schedule(self, job_id: int, start_time_pos: int) -> bool:
        """Schedule the job given job id.

        Schedule the selected job in the first possible timestep in the cluster, 
        (i.e., the first timestep in which the job’s resource requirements can be 
        fully satisfied till completion).

        Args:
            job_id (int): The selected job to be scheduled.

        Raises:
            ValueError: An error occurred poping from wrong result from `find_pos()`.

        Returns:
            bool: False if the job cannot be scheduled, i.e., does not "fit".
        """
        if start_time_pos == -1:
            return False
        job_meta = self.meta[job_id]
        num_resource_type = self.clusters.state.shape[0]
        resource_size = self.clusters.state.shape[2]
        req = job_meta.requirements

        for resource_type in range(num_resource_type):
            for job_time in range(job_meta.time_max):
                cluster_time = start_time_pos + job_time
                job_resource_index = 0
                for resource in range(resource_size):
                    if job_resource_index >= req[resource_type, job_time]:
                        break
                    cluster_pos = (resource_type, cluster_time, resource)
                    if self.clusters.state[cluster_pos] == _EMPTY_CELL:
                        self.clusters.state[cluster_pos] = job_id
                        job_resource_index += 1
                if job_resource_index < req[resource_type, job_time]:
                    msg = f"Wrong start time {start_time_pos}: Cluster time {cluster_time} cannot fit job time {job_time}."
                    raise ValueError(msg)
                self.empty_cells_cluster[resource_type,
                                         cluster_time] -= req[resource_type, job_time]
        self.job_slots.jobs[self.job_slots.jobs == job_id] = _EMPTY_CELL
        return True

    def state(self) -> npt.NDArray[np.int_]:
        """The state (image) of clusters, job slots, and backlog.

        Returns:
            list: A list of [clusters_state, job_slots_state, backlog_state].
        """
        return np.concatenate((
            self.clusters.state, 
            self.job_slots.state, 
            [self.backlog.state],
        ), axis=None)

    def add_jobs(self, jobs: npt.NDArray[np.bool_]) -> None:
        """Add jobs to res.

        Args:
            jobs (npt.NDArray[np.bool_]): 
                list of jobs. List shape (num_job, num_resource_type, time_size, resource_size).
        """
        (num_job,
         num_resource_type,
         time_size,
         resource_size) = jobs.shape
        assert((num_resource_type, time_size, resource_size)
               == self.clusters.state.shape)
        for i in range(num_job):
            job = Job.fromImage(i, jobs[i])
            self.meta[i] = job
            self.backlog.add(job, jobs[i])

    def finish(self) -> bool:
        """Check if finished all jobs.

        Returns:
            bool: If finished all jobs.
        """
        job_slots = (self.job_slots.jobs == _EMPTY_CELL).all()
        backlog = not self.backlog.queue
        return job_slots and backlog
