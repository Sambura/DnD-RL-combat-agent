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
    
    PASS_CHANCE = 0.02
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

def agent_take_turn(game: DnDBoard, agent: DnDAgent, state_indices: list[int]=None, get_turn_info: bool=False):
    actions = []

    if agent.sequential_actions:
        while game.current_movement_left > 0 or not game.used_action: # while unit is still able to do something
            _, _, new_coords, action = get_states_seq(game, agent, state_indices=state_indices)

            if new_coords is not None: # move to new_coords
                unit_position = game.current_unit.pos
                move_legal, updates = game.move(new_coords, raise_on_illegal=False)
                finish_turn = not move_legal
                if get_turn_info: actions.append(('move', {'from': unit_position, 'to': new_coords}, move_legal))
            elif action is not None: # invoke the action
                action_legal, updates = game.use_action(action, raise_on_illegal=False)
                finish_turn = not action_legal
                if get_turn_info: actions.append(('action', {'action': action}, action_legal))
            else:
                finish_turn = True
                if get_turn_info: actions.append(('pass', {}, True))

            if finish_turn: break
    else:
        _, _, new_coords, action = get_states(game, agent, state_indices=state_indices)
        unit_position = game.current_unit.pos
        move_legal, _ = game.move(new_coords, raise_on_illegal=False)
        action_legal, _ = game.use_action(action, raise_on_illegal=False)
        if get_turn_info: 
            actions.append(('move', {'from': unit_position, 'to': new_coords}, move_legal))
            actions.append(('action', {'action': action}, action_legal))
    
    game.finish_turn()
    return actions

def agents_play_loop_bare(game: DnDBoard, agents: list[DnDAgent], state_indices: list[list[int]], iter_limit=1000) -> tuple[int, int]:
    """Make agents play the game against each other. Returns iteration count and index of the winner"""
    for iter_count in range(iter_limit):
        player_id = game.current_player_id

        agent_take_turn(game, agents[player_id], state_indices[player_id])
        if game.get_game_state() != GameState.PLAYING:
            winner = 0 if len(game.players_to_units[1]) == 0 else 1
            return iter_count + 1, winner

    return iter_limit, -1

def agents_play_loop(agents: list[DnDAgent], 
                     game: DnDBoard, 
                     color_map: dict,
                     manual_input: bool=False, 
                     delay: float=0.5,
                     reset_epsilon: bool=True,
                     state_indices: list[int]=None) -> int:
    game_over = False
    iter_count = 0
    if reset_epsilon:
        epsilons = [agent.epsilon for agent in agents]
        for agent in agents: agent.epsilon = 0
    if state_indices is None: state_indices = [None] * len(agents)

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

                turn_info = agent_take_turn(game, agents[game.current_player_id], state_indices[game.current_player_id], True)
                print_turn_info(turn_info)
                game_over = game.get_game_state() != GameState.PLAYING
                print_game(game, color_map)

            except KeyboardInterrupt:
                print(f'\nGame interrupted after {iter_count} iterations')
                return None
    finally:
        if reset_epsilon: 
            for agent, eps in zip(agents, epsilons): agent.epsilon = eps

    winner = 0 if len(game.players_to_units[1]) == 0 else 1
    print(f'\nGame over in {iter_count} iterations. Winner: player #{winner + 1}')

    return iter_count, winner
