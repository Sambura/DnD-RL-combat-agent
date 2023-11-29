from PIL import Image
from typing import Tuple
import numpy as np

from .Team import Team

class RenderUnit:
    def __init__(self, unitUID: int, pos: Tuple[int, int], token: Image, team: Team = None) -> None:
        self.unitUID = unitUID
        self.team = team
        self.pos = np.ndarray[int, int]
        self.render = True
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
    
    def die(self) -> None:
        self.render = False

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
