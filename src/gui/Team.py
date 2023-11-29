from src.agent.agent import DnDAgent
from src.dnd.game_utils import get_observation_indices
from src.dnd.game_board import DnDBoard
from typing import Tuple
import numpy as np

class Team:
    def __init__(self, name:str, color:Tuple[int, int, int] = None, agent_path = None) -> None:
        self.agent_path = agent_path
        self.loaded_agent = None
        self.state_indices = None
        self.name = name
        if color == None:
            self.color = tuple(np.random.random_integers(low=0, high=255, size=3))
        else:
            self.color = color
        
    def initialize_agent(self):
        if self.agent_path is not None:
            self.loaded_agent = DnDAgent.load_agent(self.agent_path, strip=True, epsilon=0)
            self.state_indices = get_observation_indices(DnDBoard.CHANNEL_NAMES[:self.loaded_agent.in_channels])

    def is_controlled_by_player(self):
        return self.agent_path is None
    
    def get_color(self):
        return self.color
    
    def get_name(self):
        return self.name