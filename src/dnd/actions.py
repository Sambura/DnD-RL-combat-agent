from ..utils.common import manhattan_distance
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

class SwordAttack(Action):
    def __init__(self, attack_damage: int, range: int=1, name: str='Sword attack'):
        super().__init__(name)
        self.attack_damage = attack_damage
        self.range = range

    def invoke(self, game, source_unit, target_unit, skip_illegal: bool):
        if not self.check_action_legal(game, source_unit.pos, source_unit, target_unit):
            if not skip_illegal: raise RuntimeError('Too far to attack')
            return None
        
        target_unit.take_damage(self.attack_damage)

    def check_action_legal(self, game, new_position, source_unit, target_unit):
        if source_unit is target_unit: return False
        if target_unit is None: return False
        
        if manhattan_distance(target_unit.pos, new_position) > self.range:
             return False
        return True
    
class ActionInstance:
    """
    An action along with the required parameters. Used to make a move
    """
    def __init__(self, action = None, **kwargs):
        self.action = action
        self.kwargs = kwargs

    def check_action_legal(self, game, new_position): 
        if self.action is None: return True
        return self.action.check_action_legal(game, new_position, **self.kwargs)
    
    def invoke(self, game, skip_illegal):
        if self.action is None: return None
        return self.action.invoke(game, **self.kwargs, skip_illegal=skip_illegal)
