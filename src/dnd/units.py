from .actions import SwordAttack

class Unit:
    def __init__(self, name, health, speed, UID) -> None:
        self.UID = UID
        self.health = health
        self.name = name
        self.speed = speed
        self.actions = []
        self.pos = None
    
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
