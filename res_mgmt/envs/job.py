import numpy as np
import numpy.typing as npt


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
        requirements: npt.NDArray[np.int_] = None,
        time_max: int = None,
    ) -> None:
        self.id = id
        self.duration = duration
        self.requirements = requirements
        self.time_max = time_max

    @classmethod
    def fromImage(cls, id: int, image: npt.NDArray[np.bool_]):
        duration = Job.duration(image)
        requirements = Job.requirements(image)
        return cls(
            id=id,
            duration=duration,
            requirements=requirements,
            time_max=duration,
        )

    @staticmethod
    def duration(image: npt.NDArray[np.bool_]) -> int:
        return np.max(np.where(image == True), axis=1)[1] + 1

    @staticmethod
    def requirements(image: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
        return image.sum(axis=2)

    def __eq__(self, other):
        if isinstance(other, Job):
            return (
                self.id == other.id and
                self.duration == other.duration and
                (self.requirements == other.requirements).all() and
                self.time_max == other.time_max)
        return False

    def __repr__(self):
        str = """Job(
  id={},
  duration={},
  requirements={},
  time_max={},
)"""
        return str.format(self.id, self.duration, self.requirements, self.time_max)
