from src.dnd.units import Unit, GenericSoldier
from src.utils.common import RGB_to_Hex, Hex_to_RGB
from src.dnd.game_board import DnDBoard, MovementError
from src.gui.gridDrawer import generate_grid, draw_field
from src.gui.RenderUnit import RenderUnit
from src.gui.Team import Team
from src.dnd.load_unit import load_unit, load_renderUnit, getTokenImagePath, getTokenName
from src.dnd.FieldGenerator import FieldGenerator
from src.agent.agent import DnDAgent
from src.agent.agent_utils import agent_take_turn

import os
from PIL import ImageColor
from typing import List
import gradio as gr
import itertools
import glob
import re
import numpy as np


board_size:int = None
board:DnDBoard = None
render_units: List[RenderUnit] = []
selectedToken = None
selectedCell = None #TODO
teams:List[Team] = [Team('Player',(0,0,255)),
                    Team('Agent', (255,0,0), agent_path='agents/gen16-11.3i-80.0k')]
fieldGenerator:FieldGenerator = None

def get_render_unit_by_UID(UID):
  return next((render_unit for render_unit in render_units if render_unit.getUID() == UID))

def generate_board(gridScale, new_board_size):
  global board_size, board, render_units, fieldGenerator
  board_size = new_board_size
  board = DnDBoard((board_size, board_size))
  render_units = []
  field = generate_grid(board_size, gridScale)
  fieldGenerator = FieldGenerator((board_size, board_size), 2, teams)
  return field

def add_token(image, y:int, x:int, team, jsonDescriptor = None):
  global board_size
  # print(jsonDescriptor.name)
  if jsonDescriptor is None:
    board.place_unit(GenericSoldier(), (y,x), team, generateUID=True)
    render_units.append(RenderUnit(board.board[(y,x)].get_UID(), (y,x), image, team=teams[team]))
  else:
    board.place_unit(load_unit(jsonDescriptor.name), (y,x), team, generateUID=True)
    r_unit = load_renderUnit(jsonDescriptor.name, (y,x), gradio=True)
    r_unit.unitUID = board.board[(y,x)].get_UID()
    r_unit.team = teams[team]
    render_units.append(r_unit)
  if y < board_size:
    y += 1
  else:
    y = 0
    x += 1
  return y, x, update_UID_list()

def move_token(index, x:int, y:int):
  unit:Unit = board.get_unit_by_UID(render_units[index].getUID())
  try:
    board._set_unit_position(unit=unit, new_position=(y,x))
    render_units[index].pos = np.array([y, x], dtype = int)
  except MovementError as e:
    print("Error moving into chosen cell:", e)

def set_selected_token(tokenID):
  global selectedToken
  # print('set selectedToken to:', selectedToken)
  if type(tokenID) == int:
    selectedToken = tokenID
  else:
    selectedToken = None

def render_field(gridScale, target_x, target_y):
  if board.is_initialized():
    return draw_field(renderUnits=render_units, gridScale=gridScale, board_size=board_size, selectedToken=selectedToken, target=(target_x, target_y))
  else:
    return draw_field(renderUnits=render_units, gridScale=gridScale, board_size=board_size, selectedToken=selectedToken)

def update_UID_list():
  new_choices = [x.getUID() for x in render_units]
  new_choices = list(zip(new_choices, itertools.count()))
  return gr.Dropdown.update(choices=new_choices, interactive=True, label='tokenID')

def on_board_click(boardImg, gridScale, evt: gr.SelectData):
  clickedCell = (evt.index[1]//(gridScale+1), evt.index[0]//(gridScale+1))
  index = None
  if not board.is_initialized():
    try:
      index = [tuple(x.getPos()) for x in render_units if x.render].index(clickedCell)
    except ValueError:
      pass
  else:
    index = selectedToken
  return index, clickedCell[0], clickedCell[1]

def team_selection(team):
  if type(team) is not int: #returned raw string <- user assigned new team
    teams.append(Team(team))
    team = len(teams)-1
    global fieldGenerator
    if board_size is not None:
      fieldGenerator = FieldGenerator((board_size, board_size), len(teams), teams)
  team_names = [team.get_name() for team in teams]
  new_choices = list(zip(team_names, itertools.count()))
  return (gr.Dropdown.update(choices=new_choices, interactive=True), 
          RGB_to_Hex(teams[team].get_color()),
          teams[team].agent_path,
          teams[team].agent_path)

def team_selection2(team):
  if type(team) is not int: #returned raw string <- user assigned new team
    teams.append(Team(team))
    team = len(teams)-1
    global fieldGenerator
    if board_size is not None:
      fieldGenerator = FieldGenerator((board_size, board_size), len(teams), teams)
  team_names = [team.get_name() for team in teams]
  new_choices = list(zip(team_names, itertools.count()))
  return gr.Dropdown.update(choices=new_choices, interactive=True)

def team_set_color(team_color, team):
  if type(team) is not int: #returned raw string <- user assigned new team
    team = len(teams)-1
  # print('setting team color to Hex', team_color, Hex_to_RGB(team_color))
  teams[team].color = Hex_to_RGB(team_color)

def team_set_agent(team_agent_path, team):
  if type(team) is not int: #returned raw string <- user assigned new team
    team = len(teams)-1
  if team_agent_path == '':
    teams[team].agent_path = None
  else:
    if type(team_agent_path) is str: 
      if os.path.isfile(team_agent_path+'/agent.pkl') and os.path.isfile(team_agent_path+'/eval_model.pt'):
        teams[team].agent_path = team_agent_path #TODO
      else:
        raise FileExistsError("can't find model data in designated folder")
    else:
      raise TypeError("unexpected type of model_path")
  return teams[team].agent_path

def generate_game(targetCR):
  global board, render_units, fieldGenerator
  fieldGenerator.reset()
  board = fieldGenerator.load_from_folder(json_path='./Tokens', verbose=True).generate_balanced_game(targetCR=targetCR, initialize=False, generateUID=True)
  render_units = fieldGenerator.getRenderUnits()

def initialize_game():
  global board
  board.initialize_game()
  for i in range(len(teams)):
    teams[i].initialize_agent()
  return (gr.Button.update(visible=False),
      gr.Slider.update(visible=False),
      gr.Button.update(visible=False),
      gr.Dataframe.update(visible=True),
      gr.Dropdown.update(visible=True),
      gr.Number.update(visible=True),
      gr.Number.update(visible=True),
      gr.Number.update(visible=True),
      gr.Button.update(visible=True),
      gr.Number.update(visible=True),
      gr.Button.update(visible=True),
      gr.Button.update(visible=True),
      gr.Button.update(visible=True))

def end_turn():
  global board
  board.finish_turn()

def update_turn_queue():
  global board
  df_data = [[next((u.team.get_name() for u in render_units if u.getUID() == board.units[i].get_UID()), 'Error'),
            board.units[i].get_UID(),
            board.units[i].get_initiative(),
            f'{board.units[i].health}/{board.units[i].maxHealth}'] 
            for i in board.turn_order]
  return gr.DataFrame.update(np.roll(df_data, -board.current_turn_index, axis=0)),\
         [x.getUID() for x in render_units].index(board.current_unit.get_UID())

def update_dead_units():
  for i in range(len(render_units)):
    boardUnit:Unit = board.get_unit_by_UID(render_units[i].getUID())
    if boardUnit is None:
      render_units[i].die()
    else:
      render_units[i].setPos(boardUnit.pos)


def update_action_list():
  attacks = [action.name for action in board.current_unit.actions]
  return gr.Dropdown.update(choices=list(zip(attacks, itertools.count())),
                            value=0)

def attack_click(target_x, target_y, selected_action):
  global board
  print(board.current_unit.actions[selected_action].name)
  source_unit = board.current_unit
  target_unit = board.board[target_y, target_x]
  print(target_unit.get_UID())
  action = source_unit.actions[selected_action].instantiate(source_unit=source_unit, target_unit=target_unit, roll=True)
  attacked, updates = board.use_action(action)
  print(updates)
  for dead_unit in updates['units_removed']:
    get_render_unit_by_UID(dead_unit.get_UID()).die()
  print('(attack, damage) =', board.get_last_roll_info())
  
def move_click(target_x, target_y):
  global board
  moved, updates = board.move((target_y, target_x), raise_on_illegal=False)
  if moved:
    get_render_unit_by_UID(board.current_unit.get_UID()).setPos((target_y, target_x)) 
  # print_game(*decorate_game(board))
  return(board.current_movement_left)

def actors_act_once():
  acting_team = get_render_unit_by_UID(board.current_unit.get_UID()).team
  if not acting_team.is_controlled_by_player():
    turn_info = agent_take_turn(board, acting_team.loaded_agent, acting_team.state_indices, True)

def actors_act(number_actors_act):
  turns_made = 0
  while turns_made < number_actors_act or number_actors_act == -1:
    acting_team = get_render_unit_by_UID(board.current_unit.get_UID()).team
    if acting_team.is_controlled_by_player() or board.is_game_over():
      return
    actors_act_once()
    turns_made+=1
  update_turn_queue() 


with gr.Blocks() as demo:
  #Grid
  with gr.Row():
    new_board_size = gr.Slider(label="Board Size", value=10, minimum=2, maximum=100, step=1)
    makeBoard = gr.Button(value="Make Board")
  gridScale = gr.Slider(label="gridScale", value=64, minimum=16, maximum=128, step=1)
  im_canvas = gr.Image(interactive=False, height=600)

  with gr.Tabs(visible=True) as tabs:
    #Team Setup
    with gr.TabItem("Team Setup") as tab0:
      team_name1 = gr.Dropdown(value=0, label="Team Name", allow_custom_value=True,
                               info="Enter new team name to create new team",
                               choices=list(zip([team.get_name() for team in teams], itertools.count())))
      with gr.Row():
        team_color = gr.ColorPicker(value=RGB_to_Hex(teams[0].get_color()), label="Team color")
        cached_agent_path = gr.Textbox(label="Cached Agent folder")
      team_agent_path = gr.Textbox(label="Team Agent folder")
      assign_agent = gr.Button(value="assign Agent")
      agent_folders = ['agents/'+folder for folder in next(os.walk('./agents/'))[1]] + ['']
      agent_examples = gr.Examples(examples=agent_folders, inputs=[team_agent_path])
      team_name1.input(team_selection, inputs=[team_name1], outputs=[team_name1, team_color, team_agent_path, cached_agent_path])
      team_color.input(team_set_color, inputs=[team_color, team_name1])
      assign_agent.click(team_set_agent, inputs=[team_agent_path, team_name1], outputs=[cached_agent_path])
    
  #Add Token  
    with gr.TabItem("Add Token") as tab1:
      with gr.Row():
        with gr.Column(min_width=10):
          y = gr.Number(label="x pos", precision=0)
          x = gr.Number(label="y pos", precision=0)
          label = gr.Text(label="Label")
          team_name2 = gr.Dropdown(value=0, label="Team", choices=list(zip([team.get_name() for team in teams], itertools.count())), allow_custom_value=True, type='value')
          team_name2.input(team_selection2, inputs=[team_name2], outputs=[team_name2])
          jsonDescriptor = gr.File()
        im_in = gr.Image(image_mode='RGBA', type='pil')
        examples_paths = glob.glob('./Tokens/*.json')
        token_data = [[getTokenImagePath(path), getTokenName(path), path] for path in examples_paths]
        examples = gr.Examples(examples=token_data, inputs=[im_in, label, jsonDescriptor], examples_per_page=4)
      btn_add = gr.Button(value="Add token")

    #Move Token
    with gr.TabItem("Move Token") as tab2:
      with gr.Row():
          tokenID = gr.Dropdown(['t1', 't2'], label='tokenID', allow_custom_value=True)
          x_move = gr.Number(label="x pos", precision=0)
          y_move = gr.Number(label="y pos", precision=0)
      btn_move = gr.Button(value="Move token")
    
    with gr.TabItem("Play the game") as tab3:
      with gr.Row():
        game_generate = gr.Button(value="Generate random board")
        targetCR = gr.Slider(value=1, step=0.25, minimum=0.25, maximum=10)
      game_start = gr.Button(value="Initialize game")

      with gr.Row():
        with gr.Column():
          turn_order = gr.Dataframe( #TODO: team name colorization
            headers=["Team", "UnitUID", "Init", "HP"],
            datatype=["str", "str", "number", "str"],
            value=[['team1', 'unitUID1', 10, '1/1'],
                   ['team2', 'unitUID2', 9, '1/1'],
                   ['team3', 'unitUID3', 8, '1/1']],
            interactive = False, visible=False,
            label="Turn Order",
          )
        with gr.Column():
          with gr.Row():
            btn_actor_act = gr.Button(value = 'actors act', visible=False)
            number_actors_act = gr.Number(value=0, precision=0, visible=False, show_label=False)
          selected_action = gr.Dropdown(choices=['sample1', 'sample2'], label='Selected Action', visible=False, interactive=True)
          game_movement_left = gr.Number(value=6, label='Movement left', interactive=False, visible=False)
          with gr.Row():
            target_x = gr.Number(value=0, label="target x", precision=0, visible=False)
            target_y = gr.Number(value=0, label="target y", precision=0, visible=False)
          attack_btn = gr.Button(value = 'Attack', visible=False)
          move_btn = gr.Button(value = 'Move', visible=False)
          btn_end_turn = gr.Button(value = 'end_turn', visible=False)

    makeBoard.click(generate_board, inputs = [gridScale, new_board_size], outputs=[im_canvas])
    gridScale.change(render_field, inputs = [gridScale, target_x, target_y], outputs=[im_canvas])
    im_canvas.select(on_board_click, inputs=[im_canvas, gridScale], outputs=[tokenID, target_y, target_x])
    
    tab2.select(update_UID_list, outputs=tokenID)
    btn_add.click(add_token, inputs=[im_in, x, y, team_name2, jsonDescriptor], outputs=[x, y, tokenID])\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    
    tokenID.change(set_selected_token, inputs=[tokenID])\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    btn_move.click(move_token, inputs=[tokenID, x_move, y_move])\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    
    game_generate.click(generate_game, targetCR)\
      .then(update_UID_list, outputs=tokenID)\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    game_start.click(initialize_game, 
                     outputs=[game_generate, targetCR, game_start, 
                              turn_order, selected_action, 
                              game_movement_left, target_x, target_y, 
                              btn_actor_act, number_actors_act, 
                              attack_btn, move_btn, btn_end_turn])\
      .then(update_turn_queue, outputs=[turn_order, tokenID])\
      .then(update_dead_units)\
      .then(update_action_list, outputs=[selected_action])\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    btn_end_turn.click(end_turn)\
      .then(update_turn_queue, outputs=[turn_order, tokenID])\
      .then(update_dead_units)\
      .then(update_action_list, outputs=[selected_action])
    target_y.change(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    target_x.change(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    attack_btn.click(attack_click, inputs=[target_x, target_y, selected_action])\
      .then(update_turn_queue, outputs=[turn_order, tokenID])\
      .then(update_dead_units)\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    move_btn.click(move_click, inputs=[target_x, target_y], outputs=[game_movement_left])\
      .then(update_turn_queue, outputs=[turn_order, tokenID])\
      .then(update_dead_units)\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)
    btn_actor_act.click(actors_act, inputs=[number_actors_act])\
      .then(update_turn_queue, outputs=[turn_order, tokenID])\
      .then(update_dead_units)\
      .then(update_action_list, outputs=[selected_action])\
      .then(render_field, inputs=[gridScale, target_x, target_y], outputs=im_canvas)

if __name__ == "__main__":
    demo.launch()