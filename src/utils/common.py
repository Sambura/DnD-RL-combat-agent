import numpy as np
import random
import torch
import math
from dice import roll_min, roll_max
from typing import Tuple

from typing import Union

IntPoint2d = Union[tuple[int, int], np.ndarray, list[int]]

def manhattan_distance(point1: IntPoint2d, point2: IntPoint2d) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def to_tuple(coords: any) -> tuple:
    """Convert from list/ndarray coordinates representation to tuple of ints"""
    return tuple(np.array(coords).flatten())

def get_random_coords(max_y: int, max_x: int) -> tuple[int, int]:
    """Get a random coordinate on a board of size (max_y, max_x)"""
    return (random.randrange(max_y), random.randrange(max_x))

def get_random_coords_3d(max_y: int, max_x: int, max_z: int) -> tuple[int, int, int]:
    """Get a random coordinate on a board of size (max_y, max_x)"""
    return (random.randrange(max_y), random.randrange(max_x), random.randrange(max_z))

def bytes_to_human_readable(bytes):
    if bytes < 1024:
        return f'{bytes} bytes'
    elif bytes < 1024**2:
        return f'{bytes / 1024:.2f} KB'
    elif bytes < 1024**3:
        return f'{bytes / (1024**2):.2f} MB'
    elif bytes < 1024**4:
        return f'{bytes / (1024**3):.2f} GB'
    else:
        return f'{bytes / (1024**4):.2f} TB'

def transform_matrix(matrix, func):
    result_matrix = np.empty_like(matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result_matrix[i, j] = func(i, j, matrix[i, j])

    return result_matrix

def seed_everything(seed, deterministic_cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic_cudnn

def roll_avg(string:str) -> int:
  return math.ceil((roll_min(string) + roll_max(string))/2)

def RGB_to_Hex(rgb:Tuple[int, int, int]) -> str:
    # print(f'{rgb=}')
    hex = '#{:02x}{:02x}{:02x}'.format(*rgb)
    # print(f'{hex=}')
    return hex

def Hex_to_RGB(hex:str):
    # print(f'{hex=}')
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    # print(f'{rgb=}')
    return rgb