import json
import math
from typing import Tuple
from dice import roll, roll_min, roll_max
from os import listdir
from PIL import Image
from os import path


if __name__ == '__main__':
    import sys
    sys.path.append( path.dirname( path.dirname( path.dirname( path.abspath(__file__)))))
    from src.gui.adapters import RenderUnit 
    from src.dnd.units import Unit
    from src.dnd.actions import *
    from src.utils.common import roll_avg
else: 
    from ..gui.adapters import RenderUnit
    from ..utils.common import roll_avg
    from .units import Unit
    from .actions import *

def parse_json(json_path:str):
 with open(json_path) as f:
  return json.load(f)

def load_unit(json_path:str, rollHP=False) -> Unit:
  data = parse_json(json_path)
  battleStats = data['battleStats']
  CR = battleStats['CR']
  AC = battleStats['AC']
  HP = roll(battleStats['HP']) if rollHP else roll_avg(battleStats['HP'])
  cellSpeed = battleStats['speed']//5 #5 feet per cell
  attacks = data['battleStats']['attacks']
  unit = Unit(name=getTokenName(json_path), health=HP, speed=cellSpeed, AC=AC, UID=None, CR = CR)
  
  for attack in attacks:
    if attack['type'] == 'meleeWeaponAttack':
      unit.add_action(MeleeWeaponAttack(hit=attack['hit'],
                                        attack_damage=attack['damage'],
                                        range=attack['range']//5, 
                                        name=attack['name']))
    elif attack['type'] == 'rangedWeaponAttack':
      unit.add_action(RangedWeaponAttack(hit=attack['hit'],
                                         attack_damage=attack['damage'],
                                         range=attack['range']//5, 
                                         name=attack['name']))
    elif attack['type'] == 'meleeSpellAttack':
      unit.add_action(MeleeSpellAttack(hit=attack['hit'],
                                       attack_damage=attack['damage'],
                                       range=attack['range']//5, 
                                       name=attack['name']))
    elif attack['type'] == 'rangedSpellAttack':
      unit.add_action(RangedSpellAttack(hit=attack['hit'],
                                        attack_damage=attack['damage'],
                                        range=attack['range']//5, 
                                        name=attack['name']))
    else:
      raise KeyError("Tried importing unknown attack type") 
  return unit

def getTokenImagePath(json_path:str, gradio = False) -> Image:
  data = parse_json(json_path)
  if gradio:
    json_folder = 'Tokens'
  else:
    json_folder = path.dirname(path.abspath(json_path))
  return json_folder+'\\'+data['tokenImage']

def getTokenName(json_path:str) -> str:
  data = parse_json(json_path)
  return data['tokenName']

def load_renderUnit(json_path:str, pos: Tuple[int, int] = None, gradio = False) -> RenderUnit:
  token = Image.open(getTokenImagePath(json_path, gradio=gradio))
  renderUnit = RenderUnit(pos=pos, token=token, unitUID=None)
  return renderUnit

if __name__ == '__main__':
  # print(listdir('Tokens'))
  unit:Unit = load_unit('Tokens/Zombie.json')
  renderUnit:RenderUnit = load_renderUnit('Tokens/Zombie.json', pos=(0,0), gradio=False)
  print(unit.actions)
  # renderUnit.token.show()

  # data = parse_json()
  # print(data, end='\n\n')
  # print(f"{data['tokenName']=}", end='\n\n')
  # print(f"{data['battleStats']['attacks']=}", end='\n\n')
  # print(roll_min(data['battleStats']['attacks'][0]['damage'], end='\n\n'))
  # print(roll(data['battleStats']['attacks'][0]['damage'], end='\n\n'))
  # print(roll_max(data['battleStats']['attacks'][0]['damage'], end='\n\n'))
  # print(data.keys())
