from ..dnd.game_board import DnDBoard
from ..dnd.actions import ActionInstance
from ..utils.common import to_tuple, IntPoint2d
from ..dnd.game_utils import get_legal_moves, print_game, take_turn
from .agent import DnDAgent
from IPython.display import clear_output
import numpy as np
import random
import time

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
               random_action_resolver: callable=None,
               state_indices: list[int]=None) -> tuple[np.ndarray, np.ndarray, IntPoint2d, ActionInstance]:
    """
    Convinience function that observes the board and chooses the action based \
    on agent's outputs.

    Parameters:
    random_action_resolver (callable): The function to generate a random agent \
        move. If None, the move random moves are entirely random, hence likely \
        illegal.
    
    Returns: tuple (board_state, action_vector, new_coords, action_instance)
    """
    state = game.observe_board(state_indices)
    action_vector = agent.choose_action_vector(state, random_action_resolver)
    new_coords, action = decode_action(game, action_vector)

    return state, action_vector, new_coords, action

def get_default_action_resolver(game: DnDBoard):
    def random_action_resolver(state):
        legal_moves = get_legal_moves(game)
        new_pos = random.choice(list(zip(*np.where(legal_moves))))
        target_pos = random.choice(list(zip(*np.where(state[1]))))
        return (new_pos[0], target_pos[0]), (new_pos[1], target_pos[1])
    
    return random_action_resolver

def self_play_loop(agent: DnDAgent, 
                   game: DnDBoard, 
                   color_map: dict, 
                   reset_epsilon: bool=True, 
                   random_action_resolver=None, 
                   manual_input: bool=False, 
                   delay: float=0.5) -> int:
    game_over = False
    iter_count = 0
    if reset_epsilon:
        epsilon = agent.epsilon
        agent.epsilon = 0

    print_game(game, color_map)
    try:
        while not game_over:
            try:
                iter_count += 1

                if manual_input:
                    command = input()
                    if command == 'stop':
                        raise KeyboardInterrupt()
                    elif command == 'continue':
                        manual_input = False
                else:
                    time.sleep(delay)
                
                clear_output(wait=True)
                print(f'Iteration: {iter_count}')

                _, _, new_coords, action = get_states(game, agent, random_action_resolver)
                _, game_over = take_turn(game, new_coords, action, color_map, True)
            except KeyboardInterrupt:
                print(f'\nGame interrupted after {iter_count} iterations')
                return None
    finally:
        if reset_epsilon: agent.epsilon = epsilon

    print(f'\nGame over in {iter_count} iterations')

    return iter_count
