from .game_board import DnDBoard
from .actions import ActionInstance
from .utils import to_tuple, IntPoint2d
from .agent import DnDAgent
from .game_utils import get_legal_moves
import numpy as np
import random

def decode_action(game: DnDBoard, action_vector: tuple[np.ndarray[int], ...]):
    """
    Returns new coordinates along with the action to take, given the board and the \
    action vector returned by the agent.
    """
    current_unit, player_id = game.get_current_unit()
    target_unit = game.board[action_vector[0][1], action_vector[1][1]]
    action = ActionInstance(current_unit.actions[0], source_unit=current_unit, target_unit=target_unit)
    return to_tuple((action_vector[0][0], action_vector[1][0])), action

def get_states(game: DnDBoard, 
               agent: DnDAgent, 
               random_action_resolver: callable=None) -> tuple[np.ndarray, np.ndarray, IntPoint2d, ActionInstance]:
    """
    Convinience function that observes the board and chooses the action based \
    on agent's outputs.

    Parameters:
    random_action_resolver (callable): The function to generate a random agent \
        move. If None, the move random moves are entirely random, hence likely \
        illegal.
    
    Returns: tuple (board_state, action_vector, new_coords, action_instance)
    """
    state = game.observe_board()
    action_vector = agent.choose_action_vector(state, random_action_resolver)
    new_coords, action = decode_action(game, action_vector)

    return state, action_vector, new_coords, action

def get_default_action_resolver(game):
    def random_action_resolver(state):
        legal_moves = get_legal_moves(game)
        new_pos = random.choice(list(zip(*np.where(legal_moves))))
        target_pos = random.choice(list(zip(*np.where(state[1]))))
        return (new_pos[0], target_pos[0]), (new_pos[1], target_pos[1])
    
    return random_action_resolver
