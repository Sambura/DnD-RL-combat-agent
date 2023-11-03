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
    def __init__(self, name='Generic soldier', speed=3, health=100, attack_damage=10, name_postfix=""):
        super().__init__(name + name_postfix, health, speed)
        
        self.actions.append(SwordAttack(attack_damage))
