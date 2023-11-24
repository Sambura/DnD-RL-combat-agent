from typing import List
from .actions import MeleeWeaponAttack, Action

class Unit:
    def __init__(self, name, health, speed, AC, UID, CR = 1) -> None:
        self.UID = UID
        self.health = health
        self.name = name
        self.speed = speed
        self.actions: List[Action] = []
        self.pos = None
        self.AC = AC # TODO: rename?
        self.melee_attack = None
        self.CR = CR

    def get_UID(self) -> int:
        return self.UID

    def take_damage(self, damage):
        self.health -= damage

    def add_action(self, action: Action):
        # TODO: handle multiple different MeleeWeaponAttack's ??
        if isinstance(action, MeleeWeaponAttack): 
            self.melee_attack = action
        self.actions.append(action)

    def is_alive(self): return self.health > 0

    def __str__(self): return self.name

class GenericSoldier(Unit):
    def __init__(self, 
                 name: str='Generic soldier', 
                 speed: int=3, 
                 health: int=100, 
                 attack_damage: int=10,
                 range: int=1,
                 name_postfix: str="",
                 AC:int=10,
                 UID:int=None) -> None:
        super().__init__(name + name_postfix, health, speed, AC, UID)
        self.add_action(MeleeWeaponAttack(-1, attack_damage=attack_damage, range=range))
