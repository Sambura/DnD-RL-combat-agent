from ..utils.common import manhattan_distance
from dice import parse_expression
from dice.utilities import single
from dice.constants import DiceExtreme
from copy import deepcopy, copy
# from .game_board import DnDBoard # Causes circular import
# from .units import Unit # Causes circular import

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
        self.range = range
        
        self.parsed_hit =  parse_expression('d20+'+str(hit))
        self.parsed_damage = parse_expression(str(attack_damage))
        self.average_damage = (single([element.evaluate(force_extreme=DiceExtreme.EXTREME_MAX) for element in self.parsed_damage])+
                               single([element.evaluate(force_extreme=DiceExtreme.EXTREME_MIN) for element in self.parsed_damage]))/2
        
    def invoke(self, game, source_unit, target_unit, roll:bool = False):
        if roll:
            attack_roll = single([element.evaluate() for element in self.parsed_hit]) 
            damage = 0
            if attack_roll >= target_unit.AC: # hits if attack roll >= target AC
                damage = single([element.evaluate() for element in self.parsed_damage])
                target_unit.take_damage(damage) 
            return (attack_roll, damage)
        else:
            hit_chance = (target_unit.AC - self.hit + 1) / 20
            target_unit.take_damage(hit_chance * self.average_damage)
            return (999, hit_chance * self.average_damage)

    def check_action_legal(self, game, source_unit, target_unit, roll=None):
        return (target_unit is not None) and (manhattan_distance(source_unit.pos, target_unit.pos) <= self.range)
    
    def instantiate(self, source_unit, target_unit, **kwargs):
        return ActionInstance(self, source_unit=source_unit, target_unit=target_unit, **kwargs)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'parsed_hit' or 'parsed_damage':
                # print(k, v)
                setattr(result, k, copy(v))
            else:
                setattr(result, k, copy(v))
                # setattr(result, k, deepcopy(v, memo))
        return result

    
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
