from copy import deepcopy
from itertools import zip_longest
from typing import List, Tuple
import glob
import os
from warnings import warn

from ..utils.common import *
from .game_board import DnDBoard
from .units import Unit
from .load_unit import load_unit, load_renderUnit
from ..gui.adapters import RenderUnit

CCOLORS = {
    "Red": "\033[91m",
    "Green": "\033[92m",
    "Blue": "\033[94m",
    "Purple": "\033[95m",
    "Orange": "\033[93m",
    "Cyan": "\033[96m",
    "Reset": "\033[0m",
}

def print_game(game: DnDBoard, unit_to_color: dict[Unit, str]) -> None:
    num_rows, num_cols = game.board.shape
    number_padding = 2

    print(f'Units alive: {len(game.units)}')
    print(f'Players: {len(game.players_to_units)}:')
    for player_id in game.players_to_units:
        units = game.players_to_units[player_id]
        print(f'\tPlayer #{player_id} ({len(units)} units): ', end='')
        for unit in units:
            color = unit_to_color[unit]
            print(f'`{CCOLORS[color]}{unit.name} ({unit.health} HP){CCOLORS["Reset"]}`', end=', ')
        print("\b\b  ")
    print(f'\t')

    print(" " * (number_padding + 1), end="")  # Offset for y-axis numbering
    for j in range(num_cols):
        print(f"{j:>{number_padding}}", end="")

    print()

    # Print the matrix with numbering on both axes, "empty," and compact grid
    for i in range(num_rows):
        # Print y-axis numbering on the left without colons
        print(f"{i:>{number_padding}} ", end="")

        for j in range(num_cols):
            unit = game.board[i, j]
            if unit is None:
                formatted_value = "  "  # Represents "empty" with two space characters
            else:
                color = unit_to_color[unit]
                formatted_value = f"{CCOLORS[color]}██{CCOLORS['Reset']}"  # Colorize # characters
            print(formatted_value, end="")

        print(f"{i:>{number_padding}} ")

    # Print x-axis numbering at the bottom
    print(" " * (number_padding + 1), end="")  # Offset for y-axis numbering
    for j in range(num_cols):
        print(f"{j:>{number_padding}}", end="")

    print('\n')

    unit, player_id = game.current_unit, game.current_player_id
    color = unit_to_color[unit]
    print(f'Next move is by player #{player_id}: `{CCOLORS[color]}{unit.name}{CCOLORS["Reset"]}`')

def print_move(old_coords, new_coords, move_successful=None):
    old_coords = to_tuple(old_coords)
    new_coords = to_tuple(new_coords)
    success_info = '' if move_successful is None else f' [{"not " if not move_successful else ""}successful]'
    print(f'\tUnit {"moves" if old_coords != new_coords else "does not move"}: {old_coords} -> {new_coords}{success_info};')

def print_action(action, action_successful=None):
    success_info = '' if action_successful is None else f' [{"not " if not action_successful else ""}successful]'
    attibutes = {key: str(value) for key, value in action.kwargs.items()}
    print(f'\tUnit takes aciton `{action.action.name}` with attributes: {attibutes}{success_info};')

def print_turn_info(turn_info):
    for action, kwargs, success in turn_info:
        if action == 'move':
            print_move(kwargs['from'], kwargs['to'], success)
        elif action == 'action':
            print_action(kwargs['action'], success)
        elif action == 'pass':
            print('Unit finishes turn')

def place_unit_randomly_sparse(game: DnDBoard, unit: Unit, player_id: int):
    """Randomly places a unit on the board. Works best for sparse boards"""
    while True:
        coords = get_random_coords(*game.board_shape)
        if game.is_occupied(coords): continue

        game._place_unit(unit, coords, player_id)
        return coords
    
def place_unit_randomly_dense(game: DnDBoard, unit: Unit, player_id: int):
    """
    Randomly places a unit on the board. Works best for dense boards. \
    In practice works faster than sparse version when at least 80% of the cells are occupied
    """
    ys, xs = np.where(game.board == None)
    index = random.randrange(len(xs))
    coords = (ys[index], xs[index])
    game._place_unit(unit, coords, player_id)
    return coords

def get_legal_moves(game: DnDBoard):
    current_unit = game.current_unit

    def is_legal(x, y, unit):
        if unit is not None and unit is not current_unit: return False
        return manhattan_distance(current_unit.pos, (x, y)) <= current_unit.speed
    
    return transform_matrix(game.board, is_legal).astype(bool)

def get_observation_indices(fnames: list[str]):
    if fnames == None: return None

    return [DnDBoard.CHANNEL_NAMES.index(x) for x in fnames]

def constrained_sum_sample_pos(n: int, total:int):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return np.array([a - b for a, b in zip(dividers + [total], [0] + dividers)])

class fieldGenerator:
    def __init__(self, board_size: Tuple[int, int], player_count: int=2) -> None:
        self.game = DnDBoard(board_size)
        self.player_count = player_count
        self.units:List[Tuple[Unit, RenderUnit]] = []
        self.renderUnits:List[RenderUnit] = []

    def load_from_folder(self, json_path: str, verbose=False):
        json_folder = os.path.abspath(json_path)
        if verbose: print(json_folder)
        paths = glob.glob(json_folder + '/*.json')
        if verbose: print(paths)
        for file_path in paths:
            self.loadJSON(file_path)
        return self

    def loadJSON(self, json_path:str):
        self.units.append((load_unit(json_path), load_renderUnit(json_path)))
        return self
    
    def generate_balanced_game(self, targetCR:float = 5, minTypes:int = 1, maxTypes:int = 4, maxUnitsPerType:int = 4):
        self.units.sort(key = lambda x: x[0].CR)
        for player_id in range(self.player_count):
            unitTypeNumber = random.randint(1, min(maxTypes, max(int(targetCR*4), minTypes)))
            CRdistribution = constrained_sum_sample_pos(unitTypeNumber, int(targetCR*4))/4
            # print(f'{CRdistribution=}')
            for groupCR in CRdistribution:
                max_CR_unit = next((unit for unit in self.units if unit[0].CR > groupCR), None)
                # print(f'{max_CR_unit=}')
                if max_CR_unit is None:
                    viableUnits = self.units
                else:
                    viableUnits = self.units[:self.units.index(max_CR_unit)]
                # print(self.units)
                # print(groupCR, len(viableUnits))
                unit = random.choice(viableUnits)
                # print(groupCR/unit[0].CR)
                for _ in range(int(groupCR/unit[0].CR)):
                    pos = place_unit_randomly_sparse(self.game, deepcopy(unit[0]), player_id)
                    self.renderUnits.append(deepcopy(unit[1]))
                    self.renderUnits[-1].pos = pos
        self.game.initialize_game()
        return self.game
    
    def getRenderUnits(self):
        return self.renderUnits
    
    def reset(self):
        self.game = DnDBoard(self.game.board_shape)
        self.renderUnits:List[RenderUnit] = []


def generate_balanced_game(board_size: Tuple[int, int], player_units, player_count=2):
    warn ('generate_balanced_game is now performed through `fieldGenerator` class', DeprecationWarning, stacklevel=2)
    game = DnDBoard(board_size)
    for unit, count in player_units:
        for _ in range(count):
            for player_id in range(player_count):
                place_unit_randomly_sparse(game, deepcopy(unit), player_id)
    
    game.initialize_game()
    return game

def decorate_game(game: DnDBoard, 
                  rename_units: bool=True, 
                  make_colormap: bool=True, 
                  player_names: list[str]=['Ally', 'Enemy'],
                  player_colors: list[list[str]]=[['Green', 'Blue', 'Cyan'], ['Red', 'Purple', 'Orange']]):
    colormap = {}
    color_index = 0
    
    if player_colors is None: player_colors = []
    for units, player_name, colors in zip_longest(game.players_to_units.values(), player_names, player_colors):
        names_to_units = {}
        for unit in units:
            if unit.name not in names_to_units: names_to_units[unit.name] = []
            names_to_units[unit.name].append(unit)
        
        if make_colormap:
            names_to_color = {}
            if colors is not None: color_index = 0
            if colors is None: colors = list(CCOLORS)
            for name in names_to_units:
                names_to_color[name] = colors[color_index]
                color_index = (color_index + 1) % len(colors)
            
            for unit in units: colormap[unit] = names_to_color[unit.name]

        if rename_units:
            for name in names_to_units:
                for index, unit in enumerate(names_to_units[name], 1):
                    unit.name = f'{player_name} {name}'
                    if len(names_to_units[name]) > 1:
                        unit.name += f' {index}'
    
    if make_colormap:
        return game, colormap
    return game

def merge_game_updates(*args):
    units_removed = []
    for update in args:
        if update is not None: units_removed += update['units_removed']

    return { 'units_removed': units_removed }
