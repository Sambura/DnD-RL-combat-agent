from ..dnd.game_board import DnDBoard, GameState
from ..dnd.actions import ActionInstance
from ..utils.common import to_tuple, IntPoint2d, manhattan_distance, transform_matrix
from ..dnd.game_utils import get_legal_moves, print_game, print_move, print_action
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

def agents_play_loop(agent1: DnDAgent, 
                     agent2: DnDAgent, 
                     game: DnDBoard, 
                     color_map: dict,
                     manual_input: bool=False, 
                     delay: float=0.5,
                     reset_epsilon: bool=True) -> int:
    game_over = False
    iter_count = 0
    if reset_epsilon:
        epsilon1 = agent1.epsilon
        agent1.epsilon = 0
        epsilon2 = agent2.epsilon
        agent2.epsilon = 0

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
                
                clear_output(wait=False)
                print(f'Iteration: {iter_count}')

                agent = agent1 if game.current_player_id == 0 else agent2
                _, _, new_coords, action = get_states(game, agent)
                old_coords = game.current_unit.pos
                move_legal, _ = game.move(new_coords, raise_on_illegal=False)
                action_legal, _ = game.use_action(action, raise_on_illegal=False)
                game.finish_turn()
                game_over = game.get_game_state() != GameState.PLAYING

                print_move(old_coords, new_coords, move_legal)
                print_action(action, action_legal)
                print_game(game, color_map)

            except KeyboardInterrupt:
                print(f'\nGame interrupted after {iter_count} iterations')
                return None
    finally:
        if reset_epsilon: 
            agent1.epsilon = epsilon1
            agent2.epsilon = epsilon2

    winner = 0 if len(game.players_to_units[1]) == 0 else 1
    print(f'\nGame over in {iter_count} iterations. Winner: player #{winner + 1}')

    return iter_count, winner
