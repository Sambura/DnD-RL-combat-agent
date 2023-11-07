from .agent import DnDAgent
from .game_board import DnDBoard
from .agent_utils import get_states
from typing import Optional

def train_loop_trivial(agent: DnDAgent, 
                       game: DnDBoard, 
                       random_action_resolver: Optional[callable]=None, 
                       iter_limit: int=10000) -> int:
    """
    The simplest training loop for DnDAgent. As the new state, agent remembers the state of the \
    board right after it made a move. The reward is not transformed.

    Parameters:
    agent (DnDAgent): The agent to train
    game (DnDBoard): The board on which the train loop should be run
    random_action_resolver (callable): The function to generate agent's random actions
    iter_limit (int): The maximum allowed length of the game. If the game is longer, an error is raised

    Returns:
    (int) The number of iterations it took to finish the game
    """
    for iter_count in range(iter_limit):
        state, action_vector, new_coords, action = get_states(game, agent, random_action_resolver)

        reward, game_over = game.take_turn(new_coords, action, skip_illegal=True)
        new_state = game.observe_board()

        agent.memorize(state, action_vector, reward, new_state, game_over)
        agent.learn()

        if game_over: return iter_count + 1
    
    raise RuntimeError('Iteration limit exceeded')
