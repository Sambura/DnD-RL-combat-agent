from ..dnd.game_board import DnDBoard, GameState
from ..dnd.game_utils import merge_game_updates
from .agent import DnDAgent
from .agent_utils import get_states
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
    
    return reward
