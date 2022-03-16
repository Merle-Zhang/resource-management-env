class Job:
    """Job containing id and duration.

    Attributes:
        id: The id of the job.
        duration: Time the job will consume.
    """

    def __init__(
        self,
        id: int = None,
        duration: int = None, # TODO: pass in image and evaluate all the properties
        requirements: list[list[int]] = None,
        time_max: int = None,
    ) -> None:
        self.id = id
        self.duration = duration
        self.requirements = requirements
        self.time_max = time_max
