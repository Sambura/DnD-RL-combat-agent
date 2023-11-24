from ..utils.common import manhattan_distance, roll_avg
from dice import roll
# from .game_board import DnDBoard
# from .units import Unit

class Action:
    """
    Abstract base class for all actions
    """
    def __init__(self, name: str):
        self.name = name

    def invoke(self, *args):
        """Invoke the action on the board"""
        raise NotImplementedError()

class Attack(Action):
    def __init__(self, hit: int, attack_damage: int, range: int, name: str):
        super().__init__(name)
        self.hit = hit
        self.attack_damage = attack_damage
        self.average_damage = roll_avg(str(attack_damage))
        self.range = range
        
    def invoke(self, game, source_unit, target_unit):
        target_unit.take_damage(roll(str(self.attack_damage))) #TODO include AC in damage calculation

    def check_action_legal(self, game, source_unit, target_unit):
        return (target_unit is not None) and (manhattan_distance(source_unit.pos, target_unit.pos) <= self.range)
    
    def instantiate(self, source_unit, target_unit):
        return ActionInstance(self, source_unit=source_unit, target_unit=target_unit)

    
class MeleeWeaponAttack(Attack):
    def __init__(self, hit: int, attack_damage: int, range: int=1, name: str='Sword attack'):
        super().__init__(hit, attack_damage, range, name)
        
class RangedWeaponAttack(Attack):
    def __init__(self, hit: int, attack_damage: int, range: int=15, name: str='Bow attack'):
        super().__init__(hit, attack_damage, range, name)

class MeleeSpellAttack(Attack):
    def __init__(self, hit: int, attack_damage: int, range: int=1, name: str='Shocking Grasp attack'):
        super().__init__(hit, attack_damage, range, name)
    
class RangedSpellAttack(Attack):
    def __init__(self, hit: int, attack_damage: int, range: int=15, name: str='Firebolt attack'):
        super().__init__(hit, attack_damage, range, name)

class ActionInstance:
    """
    Basically a container that holds an action and a dict of parameters \
    required to invoke the action. i.e. source_unit and target_unit for attacks
    """
    def __init__(self, action: Action, **kwargs):
        self.action = action
        self.kwargs = kwargs

    def check_action_legal(self, game): 
        return self.action.check_action_legal(game, **self.kwargs)
    
    def invoke(self, game):
        return self.action.invoke(game, **self.kwargs)

    def __str__(self):
        return f'{self.action.name}: {({x:str(y) for x, y in self.kwargs.items()})}'
