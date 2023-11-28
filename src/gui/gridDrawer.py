from typing import List, Tuple
from .adapters import RenderUnit
from PIL import Image, ImageDraw
import numpy as np
from ..dnd.units import Unit

def generate_grid(grid_size = 5, space = 10) -> Image:
    square_size = space + 1
    image_size = grid_size * square_size + 1
    new_image = Image.new('RGB', (image_size, image_size), color = 'white')
    for i in range(0, image_size, square_size):
        hline = Image.new('RGB', (image_size, 1), color = 'black')
        new_image.paste(hline, (0, i))
    for i in range(0, image_size, square_size):
        vline = Image.new('RGB', (1, image_size), color = 'black')
        new_image.paste(vline, (i, 0))
    return new_image

def draw_field(renderUnits: List[RenderUnit], gridScale, board_size, selectedToken = None, target=None) -> Image:
  field = generate_grid(board_size, gridScale)
  for unit in renderUnits:
    token = unit.getToken(gridScale)
    position = (1 + unit.pos[1]*(gridScale + 1), 1 + unit.pos[0]*(gridScale + 1))
    field.paste(token, position, token)
  field = highlight_tokens(field, renderUnits, gridScale, selectedToken)
  if target is not None:
    field = highlight_target(field, target, gridScale)
  return field

def highlight_tokens(field: Image, renderUnits: List[RenderUnit], gridScale, selectedToken):
  draw = ImageDraw.Draw(field)
  for renderUnit in renderUnits:
    y, x = renderUnit.getPos()
    token_team_color = renderUnit.getTeamColor()
    position = [x*(gridScale + 1), 
                y*(gridScale + 1), 
                (x+1)*(gridScale + 1), 
                (y+1)*(gridScale + 1)]
    if selectedToken is not None and renderUnit == renderUnits[selectedToken]:
      draw.rounded_rectangle(position, radius=gridScale/10, outline=token_team_color, width=gridScale//16)
    else:
      draw.rounded_rectangle(position, radius=gridScale/10, outline=token_team_color, width=gridScale//30)       
  return field

def highlight_target(field: Image, target: Tuple[int, int], gridScale):
  x, y = target
  draw = ImageDraw.Draw(field)
  vertical = [(x+0.5)*(gridScale + 1)-gridScale//40, 
              (y+0.3)*(gridScale + 1), 
              (x+0.5)*(gridScale + 1)+gridScale//40,
              (y+0.7)*(gridScale + 1)]
  horizontal = [(x+0.3)*(gridScale + 1), 
                (y+0.5)*(gridScale + 1)-gridScale//40, 
                (x+0.7)*(gridScale + 1), 
                (y+0.5)*(gridScale + 1)+gridScale//40]
  draw.rectangle(vertical, fill='#000000')
  draw.rectangle(horizontal, fill='#000000')
  return field