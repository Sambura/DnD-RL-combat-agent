from ..utils.common import to_tuple
from .units import *

def to_board_size(board_size):
    if not hasattr(board_size, '__len__'): return (board_size, board_size)
    return to_tuple(board_size)

def get_2v2_0_config(board_size: int=5):
    return (
        to_board_size(board_size),
        [
            (GenericSoldier('soldier', attack_damage=25), 1),
            (GenericSoldier('ranger', health=50, attack_damage=25, speed=3, range=4), 1)
        ]
    )

def get_2v2_1_config(board_size: int=8):
    return (
        to_board_size(board_size),
        [
            (GenericSoldier('soldier', attack_damage=25), 1),
            (GenericSoldier('archer', health=50, attack_damage=25, speed=4, range=8), 1)
        ]
    )
