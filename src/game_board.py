import numpy as np
import re
from typing import List

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    from src.utils import * 
    from src.units import *
    from src.actions import ActionInstance
else:
    from .utils import * 
    from .units import *
    from .actions import ActionInstance

class MovementError(Exception):
    'Raised if unit cannot move into selected position'
    pass

class ActionError(Exception):
    'Raised if a selected action cannot be performed'
    pass

class DnDBoard():
    def __init__(self, board_dims: tuple[int, int]=(10, 10)):
        self.board_shape = board_dims
        self.board = np.zeros(board_dims, dtype=object)
        self.board.fill(None)
        self.players_to_units = {}
        self.units_to_players = {}
        self.units: List[Unit] = []
        self.turn_order = None
    
    def get_UIDs(self):
        return np.array([unit.get_UID() for unit in self.units])

    def get_unit_by_UID(self, UID:str) -> Unit:
        return self.units[np.where(self.get_UIDs() == UID)[0][0]]

    def assign_UID(self, position: IntPoint2d):
        unit:Unit = self.board[position]
        if unit is None:
            raise Exception('tried to asign UID to None unit')
        if unit.UID is not None:
            raise Exception('tried to asign UID to unit that already have UID')
        UIDs = self.get_UIDs().tolist() # tolist removes FutureWarning from numpy
        UID = unit.name
        splitted_label = list(filter(None, re.split(r'(\d+)', UID)))
        while UID in UIDs:
            try:
                splitted_label[-1] = str(int(splitted_label[-1])+1)
            except:
                splitted_label.append('0')
            UID = ''.join(splitted_label)
        self.board[position].UID = UID
        
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
        
        self.board[position] = unit
        self.assign_UID(position)
        if player_index not in self.players_to_units: 
            self.players_to_units[player_index] = []
        self.players_to_units[player_index].append(unit)
        self.units_to_players[unit] = player_index
        self.units.append(unit)

    def is_occupied(self, position: IntPoint2d) -> bool:
        """Is the specified cell on the board occupied by a unit?""" 
        return self.board[position] is not None

    def get_unit_position(self, unit: Unit):
        return np.where(self.board == unit)

    def initialize_game(self, check_empty: bool=True):
        #TODO: check UIDs for uniquness
        self.units = self.board[self.board != None].flatten().tolist()
        if check_empty and len(self.units) == 0:
            raise RuntimeError('The board is empty')
        
        # Assign turn order
        self.turn_order = random.sample(list(range(len(self.units))), len(self.units))
        self.current_turn_index = 0
    
    def get_current_unit(self) -> tuple[Unit, int]:
        """
            Returns tuple of (unit, player_id) for the current unit to take turn
            and the player_id that owns the unit
        """
        if self.turn_order is None: raise RuntimeError('Game was not initialized')
        unit = self.units[self.turn_order[self.current_turn_index]]
        return unit, self.units_to_players[unit]
    
    def get_distance(self, unit1: Unit, unit2: Unit) -> int:
        """Distance between units on the board"""
        pos1 = self.get_unit_position(unit1)
        pos2 = self.get_unit_position(unit2)
        return manhattan_distance(pos1, pos2)
    
    def remove_unit(self, unit):
        unit_pos = self.get_unit_position(unit)
        player_id = self.units_to_players.pop(unit)
        unit_index = self.units.index(unit)
        self.units.remove(unit)
        self.players_to_units[player_id].remove(unit)
        self.board[unit_pos] = None
        unit_turn_value = self.turn_order[unit_index]
        self.turn_order.remove(unit_turn_value)

        for i in range(len(self.turn_order)):
            if self.turn_order[i] < unit_turn_value: continue
            self.turn_order[i] -= 1

        if self.current_turn_index >= unit_turn_value:
            self.current_turn_index -= 1

    def update_board(self) -> dict[str, list[tuple[Unit, int]]]:
        """Removes all the dead units from the board"""
        to_remove = []

        for unit in self.units:
            if unit.is_alive(): continue
            to_remove.append((unit, self.units_to_players[unit]))

        for unit, player_id in to_remove: self.remove_unit(unit)

        return { 'units_removed': to_remove }

    def move_unit(self, unit: Unit, new_position: IntPoint2d, validate_move: bool=True) -> None:
        """Move the given unit on the board to the specified position"""
        if validate_move: self.check_move_legal(unit, new_position, raise_on_illegal=True)

        unit_pos = self.get_unit_position(unit)
        self.board[unit_pos] = None
        self.board[new_position] = unit

    def invoke_action(self, unit: Unit, action: ActionInstance, validate_action: bool=True) -> None:
        """Invoke the given action with the given unit"""
        if action is None: return
        if validate_action: self.check_action_legal(unit, self.get_unit_position(unit), action, raise_on_illegal=True)
        action.invoke(self, skip_illegal=not validate_action)

    def advance_turn(self) -> None:
        """Advance current turn index"""
        self.current_turn_index = (self.current_turn_index + 1) % len(self.units)

    # Is moving a unit out of turn order illegal??
    def check_move_legal(self, unit: Unit, new_position: IntPoint2d, raise_on_illegal: bool=False) -> bool:
        unit_position = self.get_unit_position(unit)
        target_cell = self.board[new_position]

        if target_cell is not None and target_cell is not unit:
            if raise_on_illegal: raise MovementError('Cell occupied')
            return False

        if manhattan_distance(unit_position, new_position) > unit.speed:
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

    def take_turn(self, new_position, action, skip_illegal=False):
        unit, player_id = self.get_current_unit()

        move_legal = self.check_move_legal(unit=unit, new_position=new_position, raise_on_illegal=not skip_illegal)
        action_legal = self.check_action_legal(unit, new_position, action, raise_on_illegal=not skip_illegal)

        if move_legal: self.move_unit(unit=unit, new_position=new_position, validate_move=False)
        if action_legal: self.invoke_action(unit, action, validate_action=False) 
        updates = self.update_board()
        self.advance_turn()

        return self.calculate_reward(player_id, updates['units_removed'])
    
    def calculate_reward(self, player_id: int, units_removed: list[tuple[Unit, int]]):
        reward = 0
        game_over = False
        # reward for removing enemy units, 1 for each unit
        reward += len([x for x in units_removed if x[1] != player_id])
        # reward for defeating players
        reward += 5 * len([x for x in units_removed if len(self.players_to_units[x[1]]) == 0 and x[1] != player_id])
        # reward for winning
        if len(self.players_to_units[player_id]) == len(self.units):
            game_over = True
            reward += 10
        # penalty for losing (on your own turn ??)
        if len(self.players_to_units[player_id]) == 0:
            game_over = True
            reward = -10
        
        return reward, game_over

    def observe_board(self) -> np.ndarray[np.float32]:
        state_channels = 7
        current_unit, player_id = self.get_current_unit()

        state = np.zeros((state_channels, *self.board_shape), dtype=np.float32)
        ally_units = transform_matrix(self.board, lambda x, y, z: (z is not None) and (self.units_to_players[z] == player_id)).astype(bool)
        unit_position = to_tuple(self.get_unit_position(current_unit))
        
        state[0] = ally_units
        state[1] = (self.board != None) ^ ally_units
        state[2, unit_position[0], unit_position[1]] = 1 # [2, *unit_position] seems to cause some problems...
        state[3] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.speed)
        state[4] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].range)
        state[5] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].attack_damage)
        state[6] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.health)

        return state
    
    def get_featuremap_names(self): 
        return ['Ally units', 'Enemy units', 'Current unit', 'Movement speed', 'Attack range', 'Attack damage', 'Health']

    def observe_board_dict(self) -> dict:
        return { key: value for key, value in zip(self.get_featuremap_names(), self.observe_board()) }

if __name__ == '__main__':
    from src.game_utils import print_game
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