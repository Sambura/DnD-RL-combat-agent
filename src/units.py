from .actions import SwordAttack

class Unit:
    def __init__(self, name, health, speed):
        self.health = health
        self.name = name
        self.speed = speed
        self.actions = []
    
    def take_damage(self, damage):
        self.health -= damage

    def is_alive(self): return self.health > 0

    def __str__(self): return self.name

class GenericSoldier(Unit):
    def __init__(self, 
                 name: str='Generic soldier', 
                 speed: int=3, 
                 health: int=100, 
                 attack_damage: int=10,
                 range: int=1,
                 name_postfix: str=""):
        super().__init__(name + name_postfix, health, speed)
        
        self.actions.append(SwordAttack(attack_damage, range=range))
