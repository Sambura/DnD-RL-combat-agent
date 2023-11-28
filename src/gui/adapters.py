from PIL import Image
from typing import Tuple
import numpy as np

class Team:
    def __init__(self, name:str, color:Tuple[int, int, int] = None, model_path:str = None) -> None:
        self.model_path = model_path
        self.name = name
        if color == None:
            self.color = tuple(np.random.random_integers(low=0, high=255, size=3))
        else:
            self.color = color
        
    def is_controlled_by_player(self):
        return self.model_path is None
    
    def get_color(self):
        return self.color
    
    def get_name(self):
        return self.name
    
class RenderUnit:
    def __init__(self, unitUID: int, pos: Tuple[int, int], token: Image, team: Team = None) -> None:
        self.unitUID = unitUID
        self.team = team
        self.pos = np.ndarray[int, int]
        if pos is None:
            pos = np.array([0, 0], dtype=int)
        else:
            self.pos = np.array(pos, dtype = int)
        self.token = token

    def getToken(self, size = None) -> Image: 
        if size is None:
            return self.token.copy()
        else:
            return self.token.copy().resize((size, size))
    
    def getUID(self) -> str:
        return self.unitUID
    
    def setPos(self, pos) -> None:
        self.pos = np.array(pos, dtype = int)

    def getPos(self) -> np.ndarray:
        return self.pos
    
    def getTeamColor(self) -> Tuple[int, int, int]:
        if self.team is None:
            return (0,0,0)
        return self.team.get_color()
