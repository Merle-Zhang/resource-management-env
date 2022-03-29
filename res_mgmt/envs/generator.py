from typing import Union
from importlib_metadata import distribution
import numpy as np
import numpy.typing as npt
from numpy.random import random_sample, normal


def generate_jobs(
    num_resource_type: int,
    time_size: int,
    resource_size: int,
    n: int = 1000,
    distribution: str = "normal",
) -> npt.NDArray[np.bool_]:
    """Generate jobs data.

    Args:
        num_resource_type (int): _description_
        time_size (int): _description_
        resource_size (int): _description_
        n (int, optional): _description_. Defaults to 1000.
        distribution (str, optional): one of [normal, union]. Defaults to "normal".

    Returns:
        npt.NDArray[np.bool_]: Generated data.
    """
    random = distri[distribution]
    shape = [
        n,
        num_resource_type,
        time_size,
        resource_size,
    ]
    durations = random(1, time_size, tuple(shape[:2]))
    resources = random(1, resource_size, tuple(shape[:3]))
    jobs = np.full(shape, False, dtype=np.bool_)
    for i in range(n):
        for type in range(num_resource_type):
            for time in range(durations[i, type]):
                jobs[i, type, time, :resources[i, type, time]] = True
    return jobs


def union_random(
    min: int,
    max: int,
    size=None,
) -> Union[int, npt.NDArray[np.int_]]:
    """Random int from uni[min, max)

    Args:
        min (int): Inclusive min.
        max (int): Exclusive max.

    Returns:
        int: random int or int array.
    """
    return np.int_((max - min) * random_sample(size) + min)


def normal_random(
    min: int,
    max: int,
    size=None,
) -> Union[int, npt.NDArray[np.int_]]:
    """Random int from normal[min, max)

    Args:
        min (int): Inclusive min.
        max (int): Exclusive max.

    Returns:
        int: random int or int array.
    """
    mu = min + 0.5 * (max - min)
    sigma = (max - min) / 6
    result = normal(mu, sigma, size)
    return np.clip(np.int_(result), min, max-1)


distri = {
    "union": union_random,
    "normal": normal_random,
}

if __name__ == "__main__":
    print(generate_jobs(2, 5, 3, 5))
