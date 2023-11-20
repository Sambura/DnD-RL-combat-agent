from typing import List
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

def draw_field(renderUnits: List[RenderUnit], selectedToken, gridScale = None, gridSize = None) -> Image:
  field = generate_grid(gridSize, gridScale)
  for unit in renderUnits:
    token = unit.getToken(gridScale)
    position = (1 + unit.pos[1]*(gridScale + 1), 1 + unit.pos[0]*(gridScale + 1))
    field.paste(token, position, token)
  if selectedToken is not None and selectedToken != '':
    field = highlight_selected_token(renderUnits, field, gridScale, selectedToken)
  return field

def highlight_selected_token(renderUnits: List[RenderUnit], field: Image, gridScale, selection):
  token_position = renderUnits[selection].getPos()[::-1]
  draw = ImageDraw.Draw(field)
  position = np.array([1 + token_position*(gridScale + 1), 1 + (token_position+1)*(gridScale + 1)]).ravel()
  draw.rounded_rectangle(position, radius=gridScale/10, outline=(255, 0, 0), width=3)
  return field