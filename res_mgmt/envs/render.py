import os
import numpy as np
import numpy.typing as npt
import pygame
from pygame.color import THECOLORS as C

from res_mgmt.envs.config import _EMPTY_CELL
from res_mgmt.envs.res import Res

os.environ["SDL_VIDEODRIVER"] = "dummy"

testres = Res.fromConfig()

testres.clusters.state = np.array([
    [
        [0, 1, 1],
        [1, 1, 2],
        [1, 1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ],
    [
        [0, 1, -1],
        [1, 2, 2],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ],
])
testres.job_slots.state = np.array([
    [
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ],
    [
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ],
    [
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ],
], dtype=np.bool_)

testres.backlog.state = 1

colours = [
    "green",
    "red",
    "brown",
    "orange",
    "yellow",
    "purple",
    "pink",
    "blue",
    "white",
    "gray",
]


def render(res: Res):
    (num_job_slot,
     num_resource_type,
     time_size,
     resource_size,
     ) = res.job_slots.state.shape

    unit = 10

    width_unit = (
        resource_size * (num_job_slot + 1) +
        num_job_slot + 4 + 3   # 3 for the text
    )
    height_unit = time_size * num_resource_type + num_resource_type + 1

    width, height = unit * width_unit, unit * height_unit

    screen = pygame.display.set_mode((width, height))
    surf = pygame.Surface((width, height))
    surf.fill(C["white"])

    def scale_tuple(t, scale):
        return tuple([scale * x for x in t])

    def block(x, y):
        x = (x+1) + resource_size*x
        y = (y+1) + time_size*y
        return (x, y, resource_size, time_size)

    blocks = [block(s, t)
              for s in range(num_job_slot + 1)
              for t in range(num_resource_type)
              ]

    def cell(block, x, y, fill):
        (bx, by, _, _) = block
        rect = (bx+x, by+y, 1, 1)
        pygame.draw.rect(surf, fill, scale_tuple(rect, unit))
        pygame.draw.rect(surf, C["black"], scale_tuple(rect, unit), 1)

    for slot in range(num_job_slot):
        for type in range(num_resource_type):
            # slot + 1 to skip the clusters in blocks
            b = blocks[(slot+1) * num_resource_type + type]
            for time in range(time_size):
                for resource in range(resource_size):
                    c = res.job_slots.state[slot, type, time, resource]
                    if c:
                        cell(b, resource, time, C["skyblue"])
                    else:
                        cell(b, resource, time, C["white"])

    for type in range(num_resource_type):
        b = blocks[0 * num_resource_type + type]
        for time in range(time_size):
            for resource in range(resource_size):
                c = res.job_slots.state[slot, type, time, resource]
                c = res.clusters.state[type, time, resource]
                if c != _EMPTY_CELL:
                    if c < len(colours):
                        colour = C[colours[c]]
                    else:
                        colour = list(C.values())[(c - len(colours)) % len(C.values())]
                    cell(b, resource, time, colour)
                else:
                    cell(b, resource, time, C["white"])

    for rect in blocks:
        pygame.draw.rect(surf, C["black"], scale_tuple(rect, unit), 2)

    screen.blit(surf, (0, 0))

    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    textsurface = myfont.render(str(res.backlog.state), False, C["black"])
    text_pos = (
        resource_size * (num_job_slot + 1) + num_job_slot + 2,
        height_unit // 2,
    )
    screen.blit(textsurface, scale_tuple(text_pos, unit))
    return screen


if __name__ == "__main__":
    # pygame.init()
    pygame.display.init()
    pygame.font.init()
    screen = render(testres)
    pygame.image.save(screen, "output.png")
    pygame.display.quit()
    pygame.font.quit()
