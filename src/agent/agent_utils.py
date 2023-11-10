from ..dnd.game_board import DnDBoard
from ..dnd.actions import ActionInstance
from ..utils.common import to_tuple, IntPoint2d, manhattan_distance, transform_matrix
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
    current_unit = game.current_unit
    target_unit = game.board[action_vector[0][1], action_vector[1][1]]
    action = ActionInstance(current_unit.actions[0], source_unit=current_unit, target_unit=target_unit)
    return to_tuple((action_vector[0][0], action_vector[1][0])), action

def get_states(game: DnDBoard, 
               agent: DnDAgent, 
               state_indices: list[int]=None) -> tuple[np.ndarray, np.ndarray, IntPoint2d, ActionInstance]:
    """
    Convinience function that observes the board and chooses the action based \
    on agent's outputs.
    
    Returns: tuple (board_state, action_vector, new_coords, action_instance)
    """
    state = game.observe_board(indices=state_indices)
    action_vector = agent.choose_action_vector(state)
    new_coords, action = decode_action(game, action_vector)

    return state, action_vector, new_coords, action

def get_legal_action_resolver(board_size):
    yi, xi = np.meshgrid(np.arange(board_size[0]), np.arange(board_size[1]), indexing='ij')

    def resolver(state: np.ndarray[np.float32]):
        current_unit_pos = np.where(state[2] != 0)
        y, x = current_unit_pos[0][0], current_unit_pos[1][0]
        current_unit_speed = state[3, y, x]

        occupied = np.logical_or(state[0], state[1])
        occupied[y, x] = 0

        distance = np.abs(yi - y) + np.abs(xi - x)
        possible_positions = np.where(np.logical_and(distance <= current_unit_speed, occupied == 0))
        possible_targets = np.where(state[1])

        pos_index = random.randrange(len(possible_positions[0]))
        target_index = random.randrange(len(possible_targets[0]))

        return [
            [possible_positions[0][pos_index], possible_targets[0][target_index]],
            [possible_positions[1][pos_index], possible_targets[1][target_index]]
        ]
    
    return resolver

def self_play_loop(agent: DnDAgent, 
                   game: DnDBoard, 
                   color_map: dict, 
                   reset_epsilon: bool=True, 
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

                _, _, new_coords, action = get_states(game, agent)
                _, game_over = take_turn(game, new_coords, action, color_map, True)
            except KeyboardInterrupt:
                print(f'\nGame interrupted after {iter_count} iterations')
                return None
    finally:
        if reset_epsilon: agent.epsilon = epsilon

    print(f'\nGame over in {iter_count} iterations')

    return iter_count
