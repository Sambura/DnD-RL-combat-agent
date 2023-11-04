from .game_board import DnDBoard
from .deep_q_network import DnDEvalModel
from torch import nn
import numpy as np
import random
from .actions import ActionInstance
from .utils import to_tuple
import pickle
import torch
import os

def bytes_to_human_readable(bytes):
    if bytes < 1024:
        return f'{bytes} bytes'
    elif bytes < 1024**2:
        return f'{bytes / 1024:.2f} KB'
    elif bytes < 1024**3:
        return f'{bytes / (1024**2):.2f} MB'
    elif bytes < 1024**4:
        return f'{bytes / (1024**3):.2f} GB'
    else:
        return f'{bytes / (1024**4):.2f} TB'

class DnDAgent():
    def __init__(self,
                 board_shape: tuple[int, int], 
                 in_channels: int=7, 
                 out_actions: int=2, 
                 lr: float=0.001, 
                 epsilon: float=0.99, 
                 min_epsilon: float=0.01, 
                 epsilon_delta: float=1e-5, 
                 gamma: float=0.9, 
                 memory_capacity: int=100000, 
                 batch_size: int=128) -> None:
        """"""
        self.in_channels = in_channels
        self.out_channels = out_actions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DnDEvalModel(self.in_channels, self.out_channels).train().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_delta = epsilon_delta
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory_position = 0
        state_shape = (memory_capacity, self.in_channels, *board_shape)
        actions_shape = (memory_capacity, self.out_channels, 2) # 2 - [x, y] coordinates
        self.state_memory = np.zeros(state_shape, dtype=np.float32)
        self.new_state_memory = np.zeros(state_shape, dtype=np.float32)
        self.actions_memory = np.zeros(actions_shape, dtype=np.float32)
        self.reward_memory = np.zeros(memory_capacity, dtype=np.float32)
        self.game_over_memory = np.zeros(memory_capacity, dtype=np.bool_)
        self.memory_bound = 0
    
    def estimate_memory_size_self(self, return_result=False):
        return DnDAgent.estimate_memory_size(self.state_memory.shape[2:], self.in_channels, self.out_channels, self.memory_capacity, return_result)

    def estimate_memory_size(board_shape: tuple[int, int], in_channels: int, out_actions: int, memory_capacity: int=100000, return_result=False) -> int:
        states_size = np.prod((memory_capacity, in_channels, *board_shape))
        actions_size = np.prod((memory_capacity, out_actions, 2))

        #        states + new_states +    actions +      game_overs +        rewards
        memory_size = states_size * 2 + actions_size + memory_capacity + memory_capacity * 4

        if return_result: return memory_size
        print(bytes_to_human_readable(memory_size))

    def predict(self, state):
        return self.model(torch.tensor(state).to(self.device).unsqueeze(0)).detach().cpu().numpy()[0]
    
    def choose_action(self, game, state):
        current_unit, player_id = game.get_current_unit()

        if random.random() < self.epsilon:
            legal_moves = game.get_legal_positions()
            new_pos = random.choice(list(zip(*np.where(legal_moves))))
            target_pos = random.choice(list(zip(*np.where(state[1]))))
            target_unit = game.board[target_pos]

            action = ActionInstance(current_unit.actions[0], source_unit=current_unit, target_unit=target_unit)

            return new_pos, action, (new_pos, target_pos)

        output = self.predict(state)
        new_pos = to_tuple(np.argwhere(output[0] == np.max(output[0]))[0])
        target_pos = to_tuple(np.argwhere(output[1] == np.max(output[1]))[0])
        target_unit = game.board[target_pos]
        action = ActionInstance(current_unit.actions[0], source_unit=current_unit, target_unit=target_unit)
        return new_pos, action, (new_pos, target_pos)
        
    def save_model(self, path='../checkpoints/'):
        torch.save(self.model.state_dict(), os.path.join(path, f'eval_model.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, f'eval_model_optim.pt'))

    def load_model(self, path='../checkpoints/'):
        self.model.load_state_dict(torch.load(os.path.join(path, f'eval_model.pt')))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, f'eval_model_optim.pt')))

    def save_agent(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'agent.pkl'), 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        self.save_model(path)

    def load_agent(path: str):
        agent = None
        with open(os.path.join(path, 'agent.pkl'), 'rb') as file:
            agent = pickle.load(file)
        agent.load_model(path)
        return agent

    def memorize(self, state, actions, reward, new_state, game_over):
        self.state_memory[self.memory_position] = state
        self.reward_memory[self.memory_position] = reward
        self.new_state_memory[self.memory_position] = new_state
        self.game_over_memory[self.memory_position] = game_over
        self.actions_memory[self.memory_position] = actions

        self.memory_position = (self.memory_position + 1) % self.memory_capacity
        self.memory_bound = max(self.memory_bound, self.memory_position) 

    def learn(self):
        if self.memory_bound < self.batch_size: return

        self.optimizer.zero_grad()

        batch_indices = np.random.choice(self.memory_bound, self.batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch_indices]).to(self.device)
        new_states = torch.tensor(self.new_state_memory[batch_indices]).to(self.device)
        rewards = torch.tensor(self.reward_memory[batch_indices]).to(self.device)
        actions = self.actions_memory[batch_indices] # (128, 2, 2)
        # TODO: do something with this??
        game_overs = self.game_over_memory[batch_indices]

        q_evals = self.model(states) # [B, 2, H, W]
        q_nexts = self.model(new_states).view(self.batch_size, self.out_channels, -1) # [B, 2, H*W]

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target = torch.clone(q_evals)
        for i in range(self.out_channels):
            action_map = actions[:, i]
            q_target[batch_index, i, action_map[:, 0], action_map[:, 1]] = rewards + self.gamma * torch.max(q_nexts[:,i], dim=1)[0]

        loss = self.loss_fn(q_evals, q_target)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon - self.epsilon_delta, self.min_epsilon)
