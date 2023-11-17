from ..dnd.game_board import DnDBoard, GameState
from .agent import DnDAgent
from .agent_utils import get_states
from ..dnd.units import Unit
from typing import Optional

def train_loop_trivial(agent: DnDAgent, 
                       game: DnDBoard, 
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
    for iter_count in range(iter_limit):
        state, action_vector, new_coords, action = get_states(game, agent)
        player_id = game.current_player_id

        reward, game_over = game.take_turn(new_coords, action, skip_illegal=True)
        new_state = game.observe_board(player_id)

        agent.memorize(state, action_vector, reward, new_state, game_over)
        agent.learn()

        if game_over: return iter_count + 1
    
    raise RuntimeError('Iteration limit exceeded')

def train_loop_delayed(agent: DnDAgent, game: DnDBoard) -> int:
    game_over = False
    iter_count = 0
    
    last_state, last_reward, last_action = None, None, None
    while not game_over:
        iter_count += 1
    
        state, action_vector, new_coords, action = get_states(game, agent)
        reward, game_over = game.take_turn(new_coords, action, skip_illegal=True)
        new_state = game.observe_board()

        ## THIS IS WRONG, FIX REWARD
        if last_state is not None:
            total_reward = last_reward - reward
            agent.memorize(last_state, last_action, total_reward, new_state, game_over)
            
        if game_over:
            agent.memorize(state, action_vector, reward, new_state, game_over)
        
        agent.learn()
        last_state = state
        last_action = action_vector
        last_reward = reward

    return iter_count

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
    if len(game.players_to_units[player_id]) == 0:
        reward = -100
    
    return reward, game_state != GameState.PLAYING
