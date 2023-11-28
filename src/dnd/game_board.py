import numpy as np
from enum import IntEnum
import re
from typing import List
import itertools

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) ) ) )
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
    CHANNEL_NAMES = ['Ally units', 'Enemy units', 'Current unit', 'Movement speed', 'Attack range', 'Attack damage', 'Health', 'Turn order',
                     'Armor', 'Is melee', 'Can react', 'Movement left', 'Can use action']
    STATE_CHANNEL_COUNT = len(CHANNEL_NAMES)

    def __init__(self, board_dims: tuple[int, int]=(10, 10)):
        self.board_shape = board_dims
        self.board = np.zeros(board_dims, dtype=Unit)
        self.board.fill(None)
        self.players_to_units = {}
        self.units_to_players = {}
        self.units: List[Unit] = []
        self.turn_order = None
        self.current_unit = None
        self.current_player_id = None
        self.current_movement_left = None
        self.reacted_list = []
        self.used_action = False
        self.initialized:bool = False

    def is_initialized(self) -> bool:
        return self.initialized

    def get_UIDs(self) -> List[int]:
        return np.array([unit.get_UID() for unit in self.units])

    def get_unit_by_UID(self, UID:str) -> Unit:
        try:
            return self.units[np.where(self.get_UIDs() == UID)[0][0]]
        except IndexError:
            return None

    def assign_UID(self, unit: Unit) -> None:
        """Assigns UID based on the name of the unit"""
        if unit is None:
            raise Exception('tried to assign UID to None unit')
        if unit.UID is not None:
            raise Exception('tried to assign UID to unit that already has UID')
        # print(self.get_UIDs())
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
        
    def place_unit(self, unit: Unit, position: IntPoint2d, player_index: int, replace: bool=False, generateUID:bool = False):
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

        if generateUID:
            self.assign_UID(unit)
        self._place_unit(unit, position, player_index)
    
    def _place_unit(self, unit: Unit, position: IntPoint2d, player_index: int):
        """Implementation of place_unit(). Only use if you know what you are doing"""
        self.board[position] = unit    
        self.units.append(unit) # for more interactivity in case it is needed
        if player_index not in self.players_to_units: self.players_to_units[player_index] = []
        self.players_to_units[player_index].append(unit)
        unit.pos = to_tuple(position)
        self.units_to_players[unit] = player_index

    def is_occupied(self, position: IntPoint2d) -> bool:
        """Is the specified cell on the board occupied by a unit?""" 
        return self.board[position] is not None

    def initialize_game(self, check_empty: bool=True):
        self.initialized = True
        self.units:List[Unit] = self.board[self.board != None].flatten().tolist()
        if check_empty and len(self.units) == 0:
            raise RuntimeError('The board is empty')
        
        turn_order = sorted(range(len(self.units)), key=lambda i : self.units[i].get_initiative(), reverse=True)
        # turn_order = random.sample(list(range(len(self.units))), len(self.units))
        # print(turn_order)
        self.set_turn_order(turn_order)
        
    def set_turn_order(self, turn_order: list[Unit], current_index: int=0):
        self.turn_order = turn_order
        self.current_turn_index = current_index - 1
        self.finish_turn()

    def remove_unit(self, unit):
        """Removes a unit from the board"""
        player_id = self.units_to_players.pop(unit)
        unit_index = self.units.index(unit)
        del self.units[unit_index]
        self.players_to_units[player_id].remove(unit)
        self.board[unit.pos] = None

        if self.turn_order is None: return
        unit_turn_index = self.turn_order.index(unit_index)

        if self.current_turn_index >= unit_turn_index:
            self.current_turn_index -= 1

        del self.turn_order[unit_turn_index]

        for i in range(len(self.turn_order)):
            if self.turn_order[i] < unit_index: continue
            self.turn_order[i] -= 1

    def update_board(self) -> dict[str, list[tuple[Unit, int]]]:
        """Removes all the dead units from the board"""
        to_remove = []

        for unit in self.units:
            if unit.is_alive(): continue
            to_remove.append((unit, self.units_to_players[unit]))

        for unit, player_id in to_remove: self.remove_unit(unit)

        return { 'units_removed': to_remove }
    
    def get_reaction_list(self):
        """Get the list of melee units that have the current unit in their attack range"""
        reaction = []
        
        for player_id, units in self.players_to_units.items():
            if player_id == self.current_player_id: continue

            for unit in units:
                if unit.melee_attack is None: continue
                if manhattan_distance(self.current_unit.pos, unit.pos) > unit.melee_attack.range: continue
                if unit in self.reacted_list: continue

                reaction.append(unit)

        return reaction
    
    def move(self, new_position: IntPoint2d, raise_on_illegal: bool=True) -> tuple[bool, dict]:
        """ 
        Move the current unit to the specified position. If the move is illegal, it is \
        either not performed, or an error is raised, depending on value of `raise_on_illegal`
        """
        new_position = to_tuple(new_position)
        if not self.check_move_legal(new_position, raise_on_illegal=raise_on_illegal): return False, None
        self.current_movement_left -= manhattan_distance(self.current_unit.pos, new_position)
        
        reaction_list = self.get_reaction_list()
        for unit in reaction_list:
            if manhattan_distance(new_position, unit.pos) <= unit.melee_attack.range: continue
            # TODO: Does reaction attack follow the same rules as a regular attack? Yes
            unit.melee_attack.invoke(self, source_unit=unit, target_unit=self.current_unit)
            self.reacted_list.append(unit)

        self._set_unit_position(self.current_unit, new_position)    
        updates = self.update_board()
        return True, updates

    def _set_unit_position(self, unit: Unit, new_position: tuple[int, int]) -> None:
        """Set unit position on the board. No checks are performed"""
        self.board[unit.pos] = None
        self.board[new_position] = unit
        unit.pos = new_position

    def use_action(self, action: ActionInstance, raise_on_illegal: bool=True) -> tuple[bool, dict]:
        """
        Invoke the given action with a current unit. If the action is illegal, it is \
        either not performed, or an error is raised, depending on value of `raise_on_illegal`
        """
        if not self.check_action_legal(action, raise_on_illegal=raise_on_illegal): return False, None
        self.last_roll_info = action.invoke(self)
        updates = self.update_board()
        self.used_action = True

        return True, updates

    def get_last_roll_info(self):
        return self.last_roll_info

    def finish_turn(self) -> None:
        """Finish the current turn and move on to the next one"""
        self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)
        self.current_unit = self.units[self.turn_order[self.current_turn_index]]
        self.current_player_id = self.units_to_players[self.current_unit]
        self.current_movement_left = self.current_unit.speed
        self.reacted_list.clear()
        self.used_action = False

    def check_move_legal(self, new_position: IntPoint2d, raise_on_illegal: bool=False) -> bool:
        """Check if the current unit can move to the specified position"""
        if not self.current_unit.is_alive(): 
            if raise_on_illegal: raise MovementError('Current unit is dead')
            return False

        target_cell = self.board[new_position]

        # It is possible that `target_cell` == `current_unit`, which techincally should be legal,
        # however as far as I see there is no real benefit in making such a move legal. Also,
        # currenlty training algorithms benefit from this move being illegal
        if target_cell is not None:
            if raise_on_illegal: raise MovementError('Cell occupied')
            return False

        if manhattan_distance(self.current_unit.pos, new_position) > self.current_movement_left:
            if raise_on_illegal: raise MovementError('Too far')
            return False
        
        return True
    
    def check_action_legal(self, action: ActionInstance, raise_on_illegal: bool=False) -> bool:
        """Check whether the current unit can perform the given action"""
        if self.used_action:
            if raise_on_illegal: raise ActionError('Cannot make multiple actions on one turn')
            return False

        if not self.current_unit.is_alive(): 
            if raise_on_illegal: raise ActionError('Current unit is dead')
            return False

        if action.action not in self.current_unit.actions:
            if raise_on_illegal: raise ActionError('The action does not belong to the selected unit')
            return False
        
        if not action.check_action_legal(self):
            if raise_on_illegal: raise ActionError('The action is illegal')
            return False
        
        return True

    def get_game_state(self, player_id: int=0) -> GameState:
        """Get current game state according to `player_id`: Playing, Win, Lose, or Draw"""
        if len(self.units) == 0: return GameState.DRAW
        
        player_units = len(self.players_to_units[player_id])

        if player_units == 0: return GameState.LOSE
        if player_units == len(self.units): return GameState.WIN
        return GameState.PLAYING

    def observe_board(self, player_id=None, indices=None) -> np.ndarray[np.float32]:
        state = self.observe_full_board(player_id)

        if indices is None: return state
        return state[indices]

    # this takes around 3:10 minutes per 500k game iterations on 8x8 board according to my questionable tests
    # 500k should be enough to train the model on a relatively easy board, which would take around 2 hours...
    def observe_full_board(self, player_id=None) -> np.ndarray[np.float32]:
        player_id = self.current_player_id if player_id is None else player_id

        state = np.zeros((self.STATE_CHANNEL_COUNT, *self.board_shape), dtype=np.float32)
        ally_units = transform_matrix(self.board, lambda x, y, z: (z is not None) and (self.units_to_players[z] == player_id)).astype(bool)
        
        # TODO: Surely this can be done faster ?? This is like 10 double nested python for-loops ffs...
        state[0] = ally_units # ally units
        state[1] = (self.board != None) ^ ally_units # enemy units
        state[2, self.current_unit.pos[0], self.current_unit.pos[1]] = 1 # current unit 
        state[3] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.speed) # speed
        state[4] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].range) # attack range
        state[5] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.actions[0].average_damage) # attack damage
        state[6] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.health) # health
        state[7] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else (self.turn_order.index(self.units.index(z)) + 1) / len(self.units)) # turn order
        state[8] = transform_matrix(self.board, lambda x,y,z: 0 if z is None else z.AC) # armor
        state[9] = transform_matrix(self.board, lambda x,y,z: 0 if z is None or next((x for x in z.actions if isinstance(x, MeleeWeaponAttack)), None) is None else 1) # is melee
        state[10] = np.logical_and(state[9], transform_matrix(self.board, lambda x,y,z: 0 if z in self.reacted_list else 1)) # units that can react
        state[11] += self.current_movement_left # movement left in current turn
        state[12] += 1 - int(self.used_action)  # can use action in current turn

        return state

    def observe_board_dict(self, player_id=None) -> dict:
        return { key: value for key, value in zip(self.CHANNEL_NAMES, self.observe_board(player_id)) }
    
    # old agents require this function to exist to be loaded, but they don't actually use it
    def calculate_reward_classic(): raise NotImplementedError()

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