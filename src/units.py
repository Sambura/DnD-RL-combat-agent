from .actions import SwordAttack
from PIL import Image
import numpy as np

class Unit:
    def __init__(self, name, health, speed, UID) -> None:
        self.UID = UID
        self.health = health
        self.name = name
        self.speed = speed
        self.actions = []
    
    def get_UID(self) -> int:
        return self.UID

    def take_damage(self, damage):
        self.health -= damage

    def is_alive(self): return self.health > 0

    def __str__(self): return self.name
    
    def __hash__(self) -> int:
        return hash(self.UID)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Unit):
            return self.UID == __value.UID
        return NotImplemented

class GenericSoldier(Unit):
    def __init__(self, 
                 name: str='Generic soldier', 
                 speed: int=3, 
                 health: int=100, 
                 attack_damage: int=10,
                 range: int=1,
                 name_postfix: str="",
                 UID=None) -> None:
        super().__init__(name + name_postfix, health, speed, UID)
        self.actions.append(SwordAttack(attack_damage, range=range))

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
    
    def getLabel(self) -> str:
        return self.unit.name
    
    def getPos(self) -> np.array:
        return self.pos

