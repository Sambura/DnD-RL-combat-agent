from ..dnd.game_board import DnDBoard, GameState
from ..dnd.game_utils import merge_game_updates
from .agent import DnDAgent
from .agent_utils import get_states, get_states_seq
from ..dnd.units import Unit
from typing import Optional

def train_loop_trivial(agent: DnDAgent, 
                       game: DnDBoard,
                       reward_fn: callable,
                       iter_limit: int=10000,
                       do_learn: bool=True,
                       memorize_fn: callable=None,
                       raise_on_limit: bool=True) -> int:
    """
    The simplest training loop for DnDAgent. As the new state, agent remembers the state of the \
    board right after it made a move. The reward is not transformed.

    Parameters:
    agent (DnDAgent): The agent to train
    game (DnDBoard): The board on which the train loop should be run
    iter_limit (int): The maximum allowed length of the game. If the game is longer, an error is raised

    Returns:
    (int) The number of iterations it took to finish the game
    """
    if agent.sequential_actions:
        raise RuntimeWarning('Provided agent is incompatible with this train loop')
    if memorize_fn is None: memorize_fn = agent.memorize

    for iter_count in range(iter_limit):
        unit, player_id = game.current_unit, game.current_player_id
        
        state, action_vector, new_coords, action = get_states(game, agent)
        move_legal, updates1 = game.move(new_coords, raise_on_illegal=False)
        action_legal, updates2 = game.use_action(action, raise_on_illegal=False)
        updates = merge_game_updates(updates1, updates2)
        game.finish_turn()
        game_state = game.get_game_state(player_id)
        game_over = game_state != GameState.PLAYING
        reward = reward_fn(game, game_state, unit, player_id, move_legal, action_legal, updates)
        new_state = game.observe_board(player_id)

        memorize_fn(state, action_vector, reward, new_state, game_over)
        if do_learn: agent.random_learn()

        if game_over: return iter_count + 1
    
    if raise_on_limit: raise RuntimeError('Iteration limit exceeded')
    return iter_limit

def train_loop_full(agent: DnDAgent,
                    game: DnDBoard, 
                    reward_fn: callable,
                    iter_limit: int=10000,
                    do_learn: bool=True,
                    memorize_fn: callable=None,
                    raise_on_limit: bool=True) -> int:
    if agent.sequential_actions:
        raise RuntimeWarning('Provided agent is incompatible with this train loop')
    if memorize_fn is None: memorize_fn = agent.memorize
    last_state, last_action, last_turn_info = None, None, None
    
    for iter_count in range(iter_limit):
        unit, player_id = game.current_unit, game.current_player_id

        state, action_vector, new_coords, action = get_states(game, agent)
        move_legal, updates1 = game.move(new_coords, raise_on_illegal=False)
        action_legal, updates2 = game.use_action(action, raise_on_illegal=False)
        updates = merge_game_updates(updates1, updates2)
        game.finish_turn()
        game_state = game.get_game_state(player_id)
        game_over = game_state != GameState.PLAYING
        turn_info = (game_state, unit, player_id, move_legal, action_legal, updates)
        new_state = game.observe_board()

        next_turn_ours = game.current_player_id == player_id

        if next_turn_ours or game_over: # if the next move is ours again, memorize current transition
            reward = reward_fn(game, turn_info, None)
            memorize_fn(state, action_vector, reward, new_state, game_over)
            if do_learn: agent.random_learn()

        if not next_turn_ours or game_over:
            if last_state is not None:
                reward = reward_fn(game, last_turn_info, turn_info)
                memorize_fn(last_state, last_action, reward, new_state, game_over)
                if do_learn: agent.random_learn()

            last_state = state
            last_action = action_vector
            last_turn_info = turn_info
        
        if game_over:
            return iter_count + 1

    if raise_on_limit: raise RuntimeError('Iteration limit exceeded')
    return iter_limit

def train_loop_sequential_V1(agent: DnDAgent, 
                             game: DnDBoard,
                             reward_fn: callable,
                             iter_limit: int=10000,
                             do_learn: bool=True,
                             memorize_fn: callable=None,
                             raise_on_limit: bool=False) -> int:
    if not agent.sequential_actions:
        raise RuntimeWarning('Provided agent is incompatible with this train loop')
    if memorize_fn is None: memorize_fn = agent.memorize

    for iter_count in range(iter_limit):
        unit, player_id = game.current_unit, game.current_player_id

        # we repeatedly show board to agent, asking what it wants to do
        # to avoid infinite loop, we stop iteration as soon as unit cannot move
        # or use action. Additionally, we terminate current turn as soon as agent
        # tries to make an illegal move. Moving into the same board cell is also
        # considered illegal for these purposes
        while game.current_movement_left > 0 or not game.used_action: # while unit is still able to do something
            state, action_vector, new_coords, action = get_states_seq(game, agent)
            action_legal, move_legal = None, None

            if new_coords is not None: # move to new_coords
                move_legal, updates = game.move(new_coords, raise_on_illegal=False)
                finish_turn = not move_legal
            elif action is not None: # invoke the action
                action_legal, updates = game.use_action(action, raise_on_illegal=False)
                finish_turn = not action_legal
            else:
                finish_turn = True
                updates = None

            game_state = game.get_game_state(player_id)
            reward = reward_fn(game, game_state, unit, player_id, move_legal, action_legal, updates)
            new_state = game.observe_board(player_id)
            game_over = game_state != GameState.PLAYING
            memorize_fn(state, action_vector, reward, new_state, game_over)
            if do_learn: agent.random_learn()

            if game_over: return iter_count + 1

            if finish_turn: break

        game.finish_turn()
    
    if raise_on_limit: raise RuntimeError('Iteration limit exceeded')
    return iter_limit

def calculate_reward_classic(game, game_state, unit: Unit, player_id: int, move_legal: bool, action_legal: bool, updates: dict):
    units_removed = updates['units_removed']
    reward = 0
    # reward for removing enemy units, 1 for each unit
    reward += len([x for x in units_removed if x[1] != player_id])
    # reward for defeating players
    reward += 5 * len([x for x in units_removed if len(game.players_to_units[x[1]]) == 0 and x[1] != player_id])
    # reward for winning
    if len(game.players_to_units[player_id]) == len(game.units):
        reward += 10
    # penalty for losing (on your own turn ??)
    # apparently penalizing -100 for losing makes model diverge rapidly
    if len(game.players_to_units[player_id]) == 0:
        reward = -10
    
    return reward

def calculate_reward_classic_seq(game, game_state, unit: Unit, player_id: int, move_legal: bool, action_legal: bool, updates: dict):
    reward = 0

    if move_legal is not None: # agent (tried) to move
        if not move_legal: return reward

        units_removed = updates['units_removed'] # only possible if move_legal == True

        # penalty for killing yourself
        if len(game.players_to_units[player_id]) == 0:
            reward -= 1

        return reward
    elif action_legal is not None: # agent (tried) to invoke action
        if not action_legal: return reward

        units_removed = updates['units_removed'] # only possible if action_legal == True

        # reward for removing enemy units
        reward += len([x for x in units_removed if x[1] != player_id])

        # reward for defeating players
        reward += 5 * len([x for x in units_removed if len(game.players_to_units[x[1]]) == 0 and x[1] != player_id])

        # reward for winning
        if len(game.players_to_units[player_id]) == len(game.units):
            reward += 10

        return reward
    
    # agent passed
    return reward

def reward_full_v1(game, data_agent, data_enemy):
    game_state, unit, player_id, move_legal, action_legal, updates = data_agent

    units_removed = updates['units_removed']

    units_left = len(game.players_to_units[player_id])
    if units_left == len(game.units):
        return 15
    elif units_left == 0:
        return -15
    
    return len([x for x in units_removed if x[1] != player_id]) * 1