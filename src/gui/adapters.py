from PIL import Image
import numpy as np

class RenderUnit:
    def __init__(self, unitUID: int, pos: tuple, token: Image) -> None:
        self.unitUID = unitUID
        self.pos = np.array(pos, dtype = int)
        self.token = token

    def getToken(self, size = None) -> Image: 
        if size is None:
            return self.token.copy()
        else:
            return self.token.copy().resize((size, size))
    
    def getUID(self) -> str:
        return self.unitUID
    
    def getPos(self) -> np.array:
        return self.pos
