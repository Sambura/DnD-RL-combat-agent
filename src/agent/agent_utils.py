from ..dnd.game_board import DnDBoard, GameState
from ..dnd.actions import ActionInstance
from ..utils.common import to_tuple, IntPoint2d, manhattan_distance, transform_matrix, get_random_coords
from ..dnd.game_utils import get_legal_moves, print_game, print_move, print_action, print_turn_info
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
    target_unit = game.board[action_vector[0][1], action_vector[1][1]]
    action = game.current_unit.actions[0].instantiate(game.current_unit, target_unit)
    return to_tuple((action_vector[0][0], action_vector[1][0])), action

def decode_action_seq(game: DnDBoard, action_vector: tuple[int, int, int]):
    """decode_action, but for sequential decision agents"""
    selected_action, target_position = action_vector[0], (action_vector[1], action_vector[2])

    if selected_action == 0: # move to the target_position
        return target_position, None
    elif selected_action == 1: # attack the unit in the target_position
        target = game.board[target_position]
        return None, game.current_unit.actions[0].instantiate(game.current_unit, target)
    elif selected_action == 2: # pass
        return None, None
    
    raise RuntimeError('Unknown action')

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

def get_states_seq(game: DnDBoard, 
                   agent: DnDAgent, 
                   state_indices: list[int]=None) -> tuple[np.ndarray, np.ndarray, IntPoint2d, ActionInstance]:
    """get_states, but for sequential decision agents"""
    state = game.observe_board(indices=state_indices)
    action_vector = agent.choose_single_action(state)
    new_coords, action = decode_action_seq(game, action_vector)

    return state, action_vector, new_coords, action

def get_legal_action_resolver(board_size, sequential_actions=False):
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
    
    PASS_CHANCE = 0.1
    ACTION_CHANCE = 0.5
    def sequential_resolver(state: np.ndarray[np.float32]):
        remaining_speed = state[11, 0, 0]
        can_move = remaining_speed > 0
        can_act = state[12, 0, 0] > 0

        # pass if random said so OR if we can't do anything else
        if not (random.random() < PASS_CHANCE or (not can_move and not can_act)):
            current_unit_pos = np.where(state[2] != 0)
            y, x = current_unit_pos[0][0], current_unit_pos[1][0]        
            distance = np.abs(yi - y) + np.abs(xi - x)

            # try to use action if random said so or if we can't move:
            if (can_act and random.random() < ACTION_CHANCE) or not can_move: # move
                attack_range = state[4, y, x]
                possible_targets = np.where(np.logical_and(state[1], distance <= attack_range))
                # if len(possible_targets[0]) == 0, it means there are no legal actions. Fall through to either move or pass
                if len(possible_targets[0]) > 0:
                    target_index = random.randrange(len(possible_targets[0]))
                    return (1, possible_targets[0][target_index], possible_targets[1][target_index])

            if can_move:
                occupied = np.logical_or(state[0], state[1])
                possible_positions = np.where(np.logical_and(distance <= remaining_speed, occupied == 0))
                if len(possible_positions[0]) > 0:
                    pos_index = random.randrange(len(possible_positions[0]))
                    return (0, possible_positions[0][pos_index], possible_positions[1][pos_index])

        return (2, *get_random_coords(board_size[0], board_size[1]))

    return sequential_resolver if sequential_actions else resolver

def agent_take_turn_seq(game: DnDBoard, agent: DnDAgent, action_to_string: bool=False):
    actions = []

    while game.current_movement_left > 0 or not game.used_action: # while unit is still able to do something
        _, _, new_coords, action = get_states_seq(game, agent)

        if new_coords is not None: # move to new_coords
            unit_position = game.current_unit.pos
            move_legal, updates = game.move(new_coords, raise_on_illegal=False)
            finish_turn = not move_legal
            actions.append(('move', {'from': unit_position, 'to': new_coords}, move_legal))
        elif action is not None: # invoke the action
            action_legal, updates = game.use_action(action, raise_on_illegal=False)
            finish_turn = not action_legal

            actions.append(('action', {'action': str(action) if action_to_string else action}, action_legal))
        else:
            actions.append(('pass', {}, True))
            finish_turn = True

        if finish_turn: break
    
    game.finish_turn()
    return actions

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
                if agent.sequential_actions:
                    turn_info = agent_take_turn_seq(game, agent)
                    print_turn_info(turn_info)

                else:
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
