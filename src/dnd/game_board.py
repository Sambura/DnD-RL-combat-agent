import numpy as np
from enum import IntEnum
import re
from typing import List

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    from src.utils.common import * 
    from src.dnd.units import *
    from src.dnd.actions import ActionInstance
else:
    from ..utils.common import * 
    from .units import *
    from .actions import ActionInstance

class MovementError(Exception):
    'Raised if unit cannot move into selected position'
    pass

class ActionError(Exception):
    'Raised if a selected action cannot be performed'
    pass

class GameState(IntEnum):
    PLAYING = 0
    LOSE = 1
    WIN = 2
    DRAW = 3

class DnDBoard():
    'Number of channels returned by observe_full_board()'
    STATE_CHANNEL_COUNT = 8
    CHANNEL_NAMES = ['Ally units', 'Enemy units', 'Current unit', 'Movement speed', 'Attack range', 'Attack damage', 'Health', 'Turn order']

    def __init__(self, board_dims: tuple[int, int]=(10, 10), reward_head=None):
        self.board_shape = board_dims
        self.board = np.zeros(board_dims, dtype=Unit)
        self.board.fill(None)
        self.players_to_units = {}
        self.units_to_players = {}
        self.units: List[Unit] = []
        self.turn_order = None
        self.current_unit = None
        self.current_player_id = None
        self.reward_head = DnDBoard.calculate_reward_classic if reward_head is None else reward_head

    def get_UIDs(self):
        return np.array([unit.get_UID() for unit in self.units])

    def get_unit_by_UID(self, UID:str) -> Unit:
        return self.units[np.where(self.get_UIDs() == UID)[0][0]]

    def assign_UID(self, unit: Unit):
        if unit is None:
            raise Exception('tried to asign UID to None unit')
        if unit.UID is not None:
            raise Exception('tried to asign UID to unit that already has UID')
        UIDs = self.get_UIDs().tolist() # tolist removes FutureWarning from numpy
        UID = unit.name
        splitted_label = list(filter(None, re.split(r'(\d+)', UID)))
        while UID in UIDs:
            try:
                splitted_label[-1] = str(int(splitted_label[-1])+1)
            except:
                splitted_label.append('0')
            UID = ''.join(splitted_label)
        unit.UID = UID
        
    def place_unit(self, unit: Unit, position: IntPoint2d, player_index: int, replace: bool=False):
        """
        Places the given unit at specified position on the board
        If `replace` is False, raises an error on attempt to replace
        an existing unit
        """
        if unit in self.units_to_players:
            raise RuntimeError('The specified unit is already on the board')
        if not replace and self.is_occupied(position):
            raise RuntimeError('This position is already occupied')
        if self.turn_order is not None:
            raise NotImplementedError('Placing units after game initialization is not supported yet')

        self.assign_UID(unit)
        self._place_unit(unit, position, player_index)
    
    def _place_unit(self, unit: Unit, position: IntPoint2d, player_index: int):
        """Same as place_unit, but no check are performed, and the UID generation is different"""
        self.board[position] = unit    
        if player_index not in self.players_to_units: self.players_to_units[player_index] = []
        self.players_to_units[player_index].append(unit)
        self.units.append(unit)
        unit.pos = to_tuple(position)
        if unit.UID is None: unit.UID = id(unit) # TODO ideally this is not the way
        self.units_to_players[unit] = player_index

    def is_occupied(self, position: IntPoint2d) -> bool:
        """Is the specified cell on the board occupied by a unit?""" 
        return self.board[position] is not None

    def initialize_game(self, check_empty: bool=True):
        #TODO: check UIDs for uniquness
        self.units = self.board[self.board != None].flatten().tolist()
        if check_empty and len(self.units) == 0:
            raise RuntimeError('The board is empty')
        
        # Assign turn order
        self.set_turn_order(random.sample(list(range(len(self.units))), len(self.units)))
        
    def set_turn_order(self, turn_order: list[Unit], current_index: int=0):
        self.turn_order = turn_order
        self.current_turn_index = current_index - 1
        self.advance_turn()

    def _remove_unit(self, unit):
        player_id = self.units_to_players.pop(unit)
        unit_index = self.units.index(unit)
        self.units.remove(unit)
        self.players_to_units[player_id].remove(unit)
        self.board[unit.pos] = None
        unit_turn_index = self.turn_order.index(unit_index)

        if self.current_turn_index >= unit_turn_index:
            self.current_turn_index -= 1

        self.turn_order.remove(unit_index)

        for i in range(len(self.turn_order)):
            if self.turn_order[i] < unit_index: continue
            self.turn_order[i] -= 1

    def update_board(self) -> dict[str, list[tuple[Unit, int]]]:
        """Removes all the dead units from the board"""
        to_remove = []

        for unit in self.units:
            if unit.is_alive(): continue
            to_remove.append((unit, self.units_to_players[unit]))

        for unit, player_id in to_remove: self._remove_unit(unit)

        return { 'units_removed': to_remove }

    def move_unit(self, unit: Unit, new_position: IntPoint2d) -> None:
        """Move the given unit on the board to the specified position"""
        self.check_move_legal(unit, new_position, raise_on_illegal=True)
        self._move_unit(unit, to_tuple(new_position))
    
    def _move_unit(self, unit: Unit, new_position: tuple[int, int]) -> None:
        self.board[unit.pos] = None
        self.board[new_position] = unit
        unit.pos = new_position

    def invoke_action(self, unit: Unit, action: ActionInstance, validate_action: bool=True) -> None:
        """Invoke the given action with the given unit"""
        if action is None: return
        if validate_action: self.check_action_legal(unit, unit.pos, action, raise_on_illegal=True)
        action.invoke(self, skip_illegal=not validate_action)

    def advance_turn(self) -> None:
        """Advance current turn index"""
        self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)
        self.current_unit = self.units[self.turn_order[self.current_turn_index]]
        self.current_player_id = self.units_to_players[self.current_unit]

    # Is moving a unit out of turn order illegal??
    def check_move_legal(self, unit: Unit, new_position: IntPoint2d, raise_on_illegal: bool=False) -> bool:
        target_cell = self.board[new_position]

        if target_cell is not None and target_cell is not unit:
            if raise_on_illegal: raise MovementError('Cell occupied')
            return False

        if manhattan_distance(unit.pos, new_position) > unit.speed:
            if raise_on_illegal: raise MovementError('Too far')
            return False
        
        return True
    
    def check_action_legal(self, 
                           unit: Unit,
                           new_position: IntPoint2d, 
                           action: ActionInstance, 
                           raise_on_illegal: bool=False) -> bool:
        """Check whether the given unit at the given position can perform the given action"""
        if action is None or action.action is None: return True # ???

        if action.action not in unit.actions:
            if raise_on_illegal: raise ActionError('The action does not belong to the selected unit')
            return False
        
        if not action.check_action_legal(self, new_position):
            if raise_on_illegal: raise ActionError('The action is illegal')
            return False
        
        return True

    def get_game_state(self, player_id: int) -> GameState:
        """Get current game state according to `player_id`"""
        if len(self.units) == 0: return GameState.DRAW
        
        player_units = len(self.players_to_units[player_id])

        if player_units == 0: return GameState.LOSE
        if player_units == len(self.units): return GameState.WIN
        return GameState.PLAYING

    def take_turn(self, new_position, action, skip_illegal=False):
        unit, player_id = self.current_unit, self.current_player_id

        move_legal = self.check_move_legal(unit=unit, new_position=new_position, raise_on_illegal=not skip_illegal)
        actual_position = new_position if move_legal else unit.pos
        action_legal = self.check_action_legal(unit, actual_position, action, raise_on_illegal=not skip_illegal)

        if move_legal: self._move_unit(unit=unit, new_position=new_position)
        if action_legal: self.invoke_action(unit, action, validate_action=False) 
        updates = self.update_board()
        self.advance_turn()
        game_state = self.get_game_state(player_id)

        return self.reward_head(self, game_state, unit, player_id, move_legal, action_legal, updates)
    
    def calculate_reward_classic(game, game_state, unit: Unit, player_id: int, move_legal: bool, action_legal: bool, updates: dict):
        units_removed = updates['units_removed']
        # reward = -0.01
        # if not move_legal: reward -= 0.5
        # if not action_legal: reward -= 0.25
        # if action_legal: reward += 0.25
        reward = 0
        game_over = False
        # reward for removing enemy units, 1 for each unit
        reward += len([x for x in units_removed if x[1] != player_id])
        # reward for defeating players
        reward += 5 * len([x for x in units_removed if len(game.players_to_units[x[1]]) == 0 and x[1] != player_id])
        # reward for winning
        if len(game.players_to_units[player_id]) == len(game.units):
            game_over = True
            reward += 10
        # penalty for losing (on your own turn ??)
        if len(game.players_to_units[player_id]) == 0:
            game_over = True
            reward = -10
        
        return reward, game_over

    def passthrough_reward_head(game, *args): return args

    def observe_board(self, player_id=None, indices=None) -> np.ndarray[np.float32]:
        state = self.observe_full_board(player_id)

        if indices is None: return state
        return state[indices]

    def observe_full_board(self, player_id=None) -> np.ndarray[np.float32]:
        player_id = self.current_player_id if player_id is None else player_id

        state = np.zeros((self.STATE_CHANNEL_COUNT, *self.board_shape), dtype=np.float32)
        ally_units = transform_matrix(self.board, lambda x, y, z: (z is not None) and (self.units_to_players[z] == player_id)).astype(bool)
        
        state[0] = ally_units
        state[1] = (self.board != None) ^ ally_units
        state[2, self.current_unit.pos[0], self.current_unit.pos[1]] = 1
        state[3] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.speed)
        state[4] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].range)
        state[5] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].attack_damage)
        state[6] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.health)
        state[7] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else (self.turn_order.index(self.units.index(z)) + 1) / len(self.units))

        return state

    def observe_board_dict(self, player_id=None) -> dict:
        return { key: value for key, value in zip(self.CHANNEL_NAMES, self.observe_board(player_id)) }

if __name__ == '__main__':
    from src.dnd.game_utils import print_game
    p1 = GenericSoldier()
    p2 = GenericSoldier()
    p3 = GenericSoldier()
    p4 = GenericSoldier()
    color_map = {
            p1: "Green",
            p2: "Red",
            p3: "Blue",
            p4: "Purple"
        }
    board = DnDBoard()
    board.place_unit(p1, (3,3), 0)
    board.place_unit(p2, (4,3), 0)
    board.place_unit(p3, (4,4), 0)
    board.initialize_game()
    # print_game(board, color_map)
    # print(board.observe_board_dict())
    print(board.units)
    print(board.board[(4,3)].get_UID())