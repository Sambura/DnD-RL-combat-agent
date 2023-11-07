from ..utils.common import bytes_to_human_readable, get_random_coords
from .deep_q_network import DnDEvalModel
from torch import nn
import numpy as np
import random
from typing import Optional
import pickle
import torch
import os

class DnDAgent():
    def __init__(self,
                 board_shape: tuple[int, int], 
                 in_channels: int=7, 
                 out_actions: int=2, 
                 lr: float=0.001, 
                 epsilon: float=0.99, 
                 min_epsilon: float=0.01, 
                 epsilon_delta: float=1e-5, 
                 epsilon_strategy: str='linear',
                 gamma: float=0.9, 
                 memory_capacity: int=100000, 
                 batch_size: int=128,
                 dual_learning: bool=False,
                 replace_model_interval: int=10000,
                 loss_fn: Optional[nn.Module]=None) -> None:
        """"""
        self.in_channels = in_channels
        self.out_channels = out_actions
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_delta = epsilon_delta
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.board_shape = board_shape
        self.dual_learning = dual_learning
        self.replace_model_interval = replace_model_interval
        self.replace_model_counter = 0
        self.on_replace = None
        
        epsilon_strategies = {
            'linear': self.linear_epsilon_step,
            'exp': self.exp_epsilon_step
        }
        self.epsilon_step = epsilon_strategies[epsilon_strategy]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_model = DnDEvalModel(self.in_channels, self.out_channels).train().to(self.device)
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.next_model = self.eval_model
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr = lr)

        if self.dual_learning:
            self.next_model = DnDEvalModel(self.in_channels, self.out_channels).train().to(self.device)
        
        self.memory_position = 0
        self.memory_bound = 0
        state_shape = (memory_capacity, self.in_channels, *board_shape)
        actions_shape = (memory_capacity, self.out_channels, 2) # 2 - [x, y] coordinates
        self.state_memory = np.zeros(state_shape, dtype=np.float32)
        self.new_state_memory = np.zeros(state_shape, dtype=np.float32)
        self.actions_memory = np.zeros(actions_shape, dtype=np.float32)
        self.reward_memory = np.zeros(memory_capacity, dtype=np.float32)
        self.game_over_memory = np.zeros(memory_capacity, dtype=np.bool_)

    def predict(self, state):
        with torch.no_grad(): # this just makes prediction a bit faster (I checked)
            return self.eval_model(torch.tensor(state).to(self.device).unsqueeze(0)).detach().cpu().numpy()[0]
    
    def choose_action_vector(self, state, random_action_resolver: Optional[callable]=None):
        """
        Chooses `out_actions` tuples of integers based on the given board state.
        """
        if random.random() < self.epsilon:
            if random_action_resolver is None:
                return [get_random_coords(*self.board_shape) for _ in range(self.out_channels)]
            else:
                return random_action_resolver(state)

        output = self.predict(state)
        return np.unravel_index(np.argmax(output.reshape(output.shape[0], -1), axis=1), output.shape[1:])

    def save_agent(self, path: str, only_models: bool=False) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if not only_models:
            with open(os.path.join(path, 'agent.pkl'), 'wb') as file:
                pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        
        torch.save(self.eval_model.state_dict(), os.path.join(path, f'eval_model.pt'))
        torch.save(self.next_model.state_dict(), os.path.join(path, f'next_model.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, f'optimizer.pt'))

    def load_agent(path: str):
        agent_path = os.path.join(path, 'agent.pkl')

        if not os.path.exists(agent_path):
            raise RuntimeError('Agent state was not found')

        with open(agent_path, 'rb') as file:
            agent = pickle.load(file)
        
        agent.eval_model.load_state_dict(torch.load(os.path.join(path, f'eval_model.pt')))
        agent.next_model.load_state_dict(torch.load(os.path.join(path, f'next_model.pt')))
        agent.optimizer.load_state_dict(torch.load(os.path.join(path, f'optimizer.pt')))

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
        if self.dual_learning:
            self.replace_model_counter += 1
            if self.replace_model_counter >= self.replace_model_interval:
                self.replace_model_counter = 0
                self.next_model.load_state_dict(self.eval_model.state_dict())
                if self.on_replace is not None: self.on_replace()

        self.optimizer.zero_grad()

        batch_indices = np.random.choice(self.memory_bound, self.batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch_indices]).to(self.device)
        new_states = torch.tensor(self.new_state_memory[batch_indices]).to(self.device)
        rewards = torch.tensor(self.reward_memory[batch_indices]).to(self.device)
        actions = self.actions_memory[batch_indices] # (batch_size, 2, 2)
        # TODO: do something with this??
        game_overs = self.game_over_memory[batch_indices]

        q_evals = self.eval_model(states) # [B, 2, H, W]
        q_nexts = self.next_model(new_states).view(self.batch_size, self.out_channels, -1) # [B, 2, H*W]

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target = torch.clone(q_evals)
        for i in range(self.out_channels):
            q_target[batch_index, i, actions[:, 0, i], actions[:, 1, i]] = rewards + self.gamma * torch.max(q_nexts[:, i], dim=1)[0]

        loss = self.loss_fn(q_evals, q_target)
        loss.backward()
        self.optimizer.step()
        self.epsilon_step()

    def linear_epsilon_step(self) -> None:
        self.epsilon = max(self.epsilon - self.epsilon_delta, self.min_epsilon)
    
    def exp_epsilon_step(self) -> None:
        self.epsilon = max(self.epsilon * (1 - self.epsilon_delta), self.min_epsilon)
    
    def estimate_memory_size_self(self, return_result: bool=False):
        return DnDAgent.estimate_memory_size(self.board_shape, self.in_channels, self.out_channels, self.memory_capacity, return_result)

    def estimate_memory_size(board_shape: tuple[int, int], in_channels: int, out_actions: int, memory_capacity: int=100000, return_result=False) -> int:
        states_size = np.prod((memory_capacity, in_channels, *board_shape))
        actions_size = np.prod((memory_capacity, out_actions, 2))

        #        states + new_states +    actions +      game_overs +        rewards
        memory_size = states_size * 2 + actions_size + memory_capacity + memory_capacity * 4

        if return_result: return memory_size
        print(bytes_to_human_readable(memory_size))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['on_replace']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.on_replace = None
