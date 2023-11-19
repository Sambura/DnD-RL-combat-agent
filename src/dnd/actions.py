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

    def invoke(self, game, source_unit, target_unit):
        target_unit.take_damage(self.attack_damage)

    def check_action_legal(self, game, source_unit, target_unit):
        return (target_unit is not None) and (manhattan_distance(source_unit.pos, target_unit.pos) <= self.range)
    
class ActionInstance:
    """
    An action along with the required parameters. Used to make a move
    """
    def __init__(self, action = None, **kwargs):
        self.action = action
        self.kwargs = kwargs

    def check_action_legal(self, game): 
        if self.action is None: return True
        return self.action.check_action_legal(game, **self.kwargs)
    
    def invoke(self, game):
        if self.action is None: return None
        return self.action.invoke(game, **self.kwargs)
