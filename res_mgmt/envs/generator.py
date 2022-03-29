from typing import Union
from importlib_metadata import distribution
import numpy as np
import numpy.typing as npt
from numpy.random import normal, randint, random_sample


def generate_jobs(
    num_resource_type: int,
    time_size: int,
    resource_size: int,
    n: int = 1000,
    distribution: str = "normal",
) -> npt.NDArray[np.bool_]:
    """Generate jobs data.

    Args:
        num_resource_type (int): Number of types of resources.
        time_size (int): Size of the time axis in the image.
        resource_size (int): Size of the resource axis in the image.
        n (int, optional): Number of jobs to be generated. Defaults to 1000.
        distribution (str, optional): One of [normal, union]. Defaults to "normal".

    Returns:
        npt.NDArray[np.bool_]: Generated data.
    """

    short_jobs = 0.8
    long_jobs = 1 - short_jobs

    t = time_size / 20
    r = resource_size

    short_duration = union_random(round(1 * t), round(3 * t), round(short_jobs * n))
    long_duration = union_random(round(10 * t), round(15 * t), round(long_jobs * n))
    duration = np.concatenate([short_duration, long_duration])
    np.random.shuffle(duration)

    dominant_res_index = randint(num_resource_type, size=n)
    demand_dominant = union_random(round(0.25 * r), round(0.5 * r), n)
    demand_other = union_random(round(0.05 * r), round(0.1 * r), n)

    def expend(arr):
        y, x = arr.shape
        pady = time_size - y
        padx = resource_size - x
        return np.pad(arr, ((0, pady), (0, padx)), "constant")

    shape = [
        n,
        num_resource_type,
        time_size,
        resource_size,
    ]
    jobs = np.full(shape, False, dtype=np.bool_)

    for i in range(n):
        for type in range(num_resource_type):
            if type == dominant_res_index[i]:
                demand = demand_dominant[i]
            else:
                demand = demand_other[i]
            job = np.ones((duration[i], demand))
            job = expend(job)
            jobs[i, type] = job
    return jobs

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
