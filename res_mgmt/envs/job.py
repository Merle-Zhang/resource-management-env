class Job:
    """Job containing id and duration.

    Attributes:
        id: The id of the job.
        duration: Time the job will consume.
    """

    def __init__(
        self,
        id: int = None,
        duration: int = None,
    ) -> None:
        self.id = id
        self.duration = duration
