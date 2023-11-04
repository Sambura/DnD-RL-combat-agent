import numpy as np
import re
from typing import List

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    from src.utils import * 
    from src.units import *
else:
    from .utils import * 
    from .units import *

def transform_matrix(matrix, func):
    result_matrix = np.empty_like(matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result_matrix[i, j] = func(i, j, matrix[i, j])

    return result_matrix

class DnDBoard():
    def __init__(self, board_dims: tuple[int, int]=(10, 10)):
        self.board_shape = board_dims
        self.board = np.zeros(board_dims, dtype=object)
        self.board.fill(None)
        self.players_to_units = {}
        self.units_to_players = {}
        self.units: List[Unit] = []
    
    def get_UIDs(self):
        return np.array([unit.get_UID() for unit in self.units])

    def get_unit_by_UID(self, UID:str) -> Unit:
        # print(UID, self.get_UIDs())
        return self.units[np.where(self.get_UIDs() == UID)[0][0]]

    def assign_UID(self, position: IntPoint2d):
        unit:Unit = self.board[position]
        if unit is None:
            raise Exception('tried to asign UID to None unit')
        if unit.UID is not None:
            raise Exception('tried to asign UID to unit that already have UID')
        UIDs = self.get_UIDs()
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
        if self.units is None: raise RuntimeError('Game was not initialized')
        unit = self.units[self.turn_order[self.current_turn_index]]
        return unit, self.units_to_players[unit]
    
    def get_distance(self, unit1: Unit, unit2: Unit) -> int:
        """Distance between units on the board"""
        pos1 = self.get_unit_position(unit1)
        pos2 = self.get_unit_position(unit2)
        return manhattan_distance(pos1, pos2)
    
    def check_action_legal(self, new_position, action, verbose=False):
        if action is None: return True

        if not action.check_action_legal(self, new_position):
            if verbose: print('Action illegal')
            return False
        
        unit, player_id = self.get_current_unit()
        if action.action is not None and action.action not in unit.actions:
            if verbose: print('Action invalid')
            return False
        
        return True
    
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

    def update_board(self):
        to_remove = []

        for unit in self.units:
            if unit.is_alive(): continue
            to_remove.append((unit, self.units_to_players[unit]))

        for unit, player_id in to_remove: self.remove_unit(unit)

        return { 'units_removed': to_remove }

    # for the current unit to move in
    def get_legal_positions(self):
        current_unit, player_id = self.get_current_unit()
        pos = self.get_unit_position(current_unit)

        def is_legal(x, y, unit):
            if unit is not None and unit is not current_unit: return False
            return manhattan_distance(pos, (x, y)) <= current_unit.speed
        
        return transform_matrix(self.board, is_legal).astype(bool)

    def move_token(self, unit: Unit, new_position: IntPoint2d, skip_illegal=False) -> None:
        move_legal = self.check_move_legal(unit, new_position)
        
        if not move_legal:
            if not skip_illegal: raise RuntimeError('Illegal move')
        else:
            unit_pos = self.get_unit_position(unit)
            self.board[unit_pos] = None
            self.board[new_position] = unit

    def check_move_legal(self, unit:Unit, new_position: IntPoint2d, verbose=False):
        unit_position = self.get_unit_position(unit)
        target_cell = self.board[new_position]
        if target_cell is not None and target_cell is not unit:
            if verbose: print('Cell occupied')
            return False

        if manhattan_distance(unit_position, new_position) > unit.speed:
            if verbose: print('Too far')
            return False
        
        return True

    def take_turn(self, new_position, action, skip_illegal=False):
        unit, player_id = self.get_current_unit()
        self.move_token(unit=unit, new_position=new_position, skip_illegal=skip_illegal)

        if not self.check_action_legal(new_position, action):
            if not skip_illegal: raise RuntimeError('Illegal action')

            self.current_turn_index = (self.current_turn_index + 1) % len(self.units)
            return 0, False

        if action is not None: action.invoke(self, skip_illegal=skip_illegal)
        update = self.update_board()
        self.current_turn_index = (self.current_turn_index + 1) % len(self.units)
        
        ### Stuff for agent

        reward = 0
        game_over = False
        # reward for removing enemy units, 1 for each unit
        reward += len([x for x in update['units_removed'] if x[1] != player_id])
        # reward for defeating players
        reward += 5 * len([x for x in update['units_removed'] if len(self.players_to_units[x[1]]) == 0 and x[1] != player_id])
        # reward for winning
        if len(self.players_to_units[player_id]) == len(self.units):
            game_over = True
            reward += 10
        # penalty for losing (on your own turn ??)
        if len(self.players_to_units[player_id]) == 0:
            game_over = True
            reward = -10
        
        return reward, game_over
    
    def observe_board(self) -> np.ndarray:
        current_unit, player_id = self.get_current_unit()
        ally_units = transform_matrix(self.board, lambda x, y, z: (z is not None) and (self.units_to_players[z] == player_id)).astype(bool)
        enemy_units = (self.board != None) ^ ally_units
        unit_to_move = np.zeros(ally_units.shape, dtype=bool)
        unit_to_move[self.get_unit_position(current_unit)] = True
        speeds = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.speed)
        attack_ranges = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].range)
        attack_damages = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].attack_damage)
        healths = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.health)

        return np.array([
            ally_units,
            enemy_units,
            unit_to_move,
            speeds,
            attack_ranges,
            attack_damages,
            healths
        ], dtype=np.float32)

    def observe_board_dict(self) -> dict:
        current_unit, player_id = self.get_current_unit()
        ally_units = transform_matrix(self.board, lambda x, y, z: (z is not None) and (self.units_to_players[z] == player_id)).astype(bool)
        enemy_units = (self.board != None) ^ ally_units
        unit_to_move = np.zeros(ally_units.shape, dtype=bool)
        unit_to_move[self.get_unit_position(current_unit)] = True
        speeds = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.speed)
        attack_ranges = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].range)
        attack_damages = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].attack_damage)
        healths = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.health)

        return {
            'ally_units': ally_units,
            'enemy_units': enemy_units,
            'unit_to_move': unit_to_move,
            'speeds': speeds,
            'attack_ranges': attack_ranges,
            'attack_damages': attack_damages,
            'healths': healths
        }

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