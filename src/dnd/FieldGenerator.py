import glob
from copy import deepcopy
import os
from typing import List, Tuple

from ..utils.common import *
from .load_unit import load_unit, load_renderUnit
from .game_board import DnDBoard
from ..gui.RenderUnit import RenderUnit
from .units import Unit
from .game_utils import place_unit_randomly_sparse

def constrained_sum_sample_pos(n: int, total:int):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return np.array([a - b for a, b in zip(dividers + [total], [0] + dividers)])

class FieldGenerator:
    def __init__(self, board_size: Tuple[int, int], player_count: int=2, teams = None) -> None:
        self.game = DnDBoard(board_size)
        self.player_count = player_count
        self.units:List[Tuple[Unit, RenderUnit]] = []
        self.renderUnits:List[RenderUnit] = []
        self.teams = teams

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
    
    def generate_balanced_game(self, targetCR:float = 5, minTypes:int = 1, maxTypes:int = 4, maxUnitsPerType:int = 4, initialize = True, generateUID = False):
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
                    # print(_, 'placing', unit[0].name)
                    unit_copy = deepcopy(unit[0])
                    pos = place_unit_randomly_sparse(self.game, unit_copy, player_id, generateUID=generateUID)
                    self.renderUnits.append(deepcopy(unit[1]))
                    self.renderUnits[-1].pos = pos
                    if generateUID:
                        self.renderUnits[-1].unitUID = unit_copy.get_UID()
                    if self.teams is not None:
                        self.renderUnits[-1].team = self.teams[player_id]


        if initialize:
            self.game.initialize_game()
        return self.game
    
    def getRenderUnits(self):
        return self.renderUnits
    
    def reset(self):
        self.game = DnDBoard(self.game.board_shape)
        self.renderUnits:List[RenderUnit] = []