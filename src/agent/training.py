from ..dnd.game_board import DnDBoard, GameState
from ..dnd.game_utils import merge_game_updates
from .agent import DnDAgent
from .agent_utils import get_states, get_states_seq
from ..dnd.units import Unit
from typing import Optional

def train_loop_trivial(agent: DnDAgent, 
                       game: DnDBoard,
                       reward_fn: callable,
                       iter_limit: int=10000) -> int:
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

    for iter_count in range(iter_limit):
        state, action_vector, new_coords, action = get_states(game, agent)
        unit, player_id = game.current_unit, game.current_player_id
        
        move_legal, updates1 = game.move(new_coords, raise_on_illegal=False)
        action_legal, updates2 = game.use_action(action, raise_on_illegal=False)
        game.finish_turn()
        updates = merge_game_updates(updates1, updates2)
        game_state = game.get_game_state(player_id)
        reward = reward_fn(game, game_state, unit, player_id, move_legal, action_legal, updates)
        new_state = game.observe_board(player_id)
        game_over = game_state != GameState.PLAYING

        agent.memorize(state, action_vector, reward, new_state, game_over)
        agent.learn()

        if game_over: return iter_count + 1
    
    raise RuntimeError('Iteration limit exceeded')

def train_loop_sequential_V1(agent: DnDAgent, 
                             game: DnDBoard,
                             reward_fn: callable,
                             iter_limit: int=10000,
                             do_learn: bool=True) -> int:
    if not agent.sequential_actions:
        raise RuntimeWarning('Provided agent is incompatible with this train loop')

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
            agent.memorize(state, action_vector, reward, new_state, game_over)
            if do_learn: agent.learn()

            if game_over: return iter_count + 1

            if finish_turn: break

        game.finish_turn()
    
    raise RuntimeError('Iteration limit exceeded')

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
    #if len(game.players_to_units[player_id]) == 0:
    #    reward = -100
    #    pass
    
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
