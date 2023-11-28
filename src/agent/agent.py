from ..utils.common import bytes_to_human_readable, get_random_coords, get_random_coords_3d
from .deep_q_network import DnDEvalModel
from torch import nn
import numpy as np
import random
from typing import Optional
import pickle
import torch
import os

def get_default_random_action_resolver(board_shape, out_channels, sequential_actions):
    h, w = board_shape

    def sequential_resolver(state):
        return get_random_coords_3d(out_channels, h, w)

    def resolver(state):
        return [get_random_coords(h, w) for _ in range(out_channels)]
    
    return sequential_resolver if sequential_actions else resolver

class RandomAgent():
    def __init__(self, board_shape, out_actions: int=2, action_resolver=None):
        self.board_shape = board_shape
        self.out_channels = out_actions
        self.random_action_resolver = get_default_random_action_resolver(board_shape, out_actions)
        if action_resolver is not None: self.random_action_resolver = action_resolver

    def choose_action_vector(self, state):
        return self.random_action_resolver(state)

class DnDAgent():
    BASE_ATTRS = ['model_class', 'eval_model', 'epsilon', 'board_shape', 'in_channels', 
                  'out_channels', 'device', 'sequential_actions', 'random_action_resolver']
    """Attributes that should not be stripped upon loading agent"""

    def __init__(self,
                 board_shape: tuple[int, int], 
                 in_channels: int, 
                 out_actions: int, 
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
                 loss_fn: Optional[nn.Module]=None,
                 random_action_resolver=None,
                 model_class: type[nn.Module]=DnDEvalModel,
                 sequential_actions: bool=False) -> None:
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
        self.model_class = model_class
        self.sequential_actions = sequential_actions
        self.random_action_resolver = get_default_random_action_resolver(board_shape, out_actions, sequential_actions)
        if random_action_resolver is not None: self.random_action_resolver = random_action_resolver
        
        epsilon_strategies = {
            'linear': self.linear_epsilon_step,
            'exp': self.exp_epsilon_step
        }
        self.epsilon_step = epsilon_strategies[epsilon_strategy]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_model = model_class(self.in_channels, self.out_channels).train().to(self.device)
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.next_model = self.eval_model
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=lr)

        if self.dual_learning:
            self.next_model = model_class(self.in_channels, self.out_channels).eval().to(self.device)
        
        self.memory_position = 0
        self.memory_bound = 0
        state_shape = (memory_capacity, self.in_channels, *board_shape)
        if sequential_actions:
            actions_shape = (memory_capacity, 3) # action plane + 2 : [x, y] coordinates
        else:
            actions_shape = (memory_capacity, self.out_channels, 2) # 2 - [x, y] coordinates
        self.state_memory = np.zeros(state_shape, dtype=np.float32)
        self.new_state_memory = np.zeros(state_shape, dtype=np.float32)
        self.actions_memory = np.zeros(actions_shape, dtype=np.int64)
        self.reward_memory = np.zeros(memory_capacity, dtype=np.float32)
        self.game_over_memory = np.zeros(memory_capacity, dtype=np.bool_)
        self.stripped = False

    def predict(self, state):
        with torch.no_grad(): # this just makes prediction a bit faster (I checked)
            return self.eval_model(torch.tensor(state).to(self.device).unsqueeze(0)).detach().cpu().numpy()[0]
    
    def choose_action_vector(self, state):
        """
        Chooses `out_actions` tuples of integers based on the given board state.
        """
        if random.random() < self.epsilon:
            return self.random_action_resolver(state)

        output = self.predict(state)
        return np.unravel_index(np.argmax(output.reshape(output.shape[0], -1), axis=1), output.shape[1:])
    
    def choose_single_action(self, state):
        if random.random() < self.epsilon:
            return self.random_action_resolver(state)

        output = self.predict(state)
        return np.unravel_index(np.argmax(output.reshape(output.shape[0], -1)), output.shape)

    def save_agent(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'agent.pkl'), 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        
        torch.save(self.eval_model.state_dict(), os.path.join(path, f'eval_model.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, f'optimizer.pt'))
        if self.dual_learning:
            torch.save(self.next_model.state_dict(), os.path.join(path, f'next_model.pt'))

    def load_agent(path: str, strip: bool=False, **kwargs):
        """
        Loads the agent from the specified directory

        Parameters:
        path (str): the directory containing agent's save files
        strip (bool): if True, sets the agent to evaluation mode, deleting most of its attributes \
            to take less memory. Note that it is not possible to train a stripped agent.
        **kwargs: list of agent's attributes whose value should be modified. i.e. `epsilon = 0`
        """
        agent_path = os.path.join(path, 'agent.pkl')

        with open(agent_path, 'rb') as file:
            agent = pickle.load(file)

        if 'model_class' in kwargs: # delete these 5 lines asap
            agent.model_class = kwargs['model_class']
            agent.eval_model = agent.model_class(agent.in_channels, agent.out_channels).to(agent.device).train()
            if not strip:
                if agent.dual_learning:
                    agent.next_model = agent.model_class(agent.in_channels, agent.out_channels).to(agent.device).eval()
                else:
                    agent.next_model = agent.eval_model
                agent.optimizer = torch.optim.Adam(agent.eval_model.parameters())
        
        agent.eval_model.load_state_dict(torch.load(os.path.join(path, f'eval_model.pt')))
        if not strip:
            agent.optimizer.load_state_dict(torch.load(os.path.join(path, f'optimizer.pt')))
            if agent.dual_learning:
                agent.next_model.load_state_dict(torch.load(os.path.join(path, f'next_model.pt')))
        
        if strip:
            for x in agent.__dict__.copy():
                if x in DnDAgent.BASE_ATTRS: continue
                delattr(agent, x)

            agent.stripped = True
            agent.eval_model.eval()

        for name, value in kwargs.items():
            assert hasattr(agent, name), f'attribute {name} does not exist'
            setattr(agent, name, value)

        return agent

    def memorize(self, state, actions, reward, new_state, game_over):
        self.state_memory[self.memory_position] = state
        self.reward_memory[self.memory_position] = reward
        self.new_state_memory[self.memory_position] = new_state
        self.game_over_memory[self.memory_position] = not game_over
        self.actions_memory[self.memory_position] = actions

        self.memory_position = (self.memory_position + 1) % self.memory_capacity
        self.memory_bound = max(self.memory_bound, self.memory_position) 

    def clear_memory(self):
        self.memory_position = 0
        self.memory_bound = 0

    def random_learn(self):
        """Take a random batch of memories and learn"""
        if self.memory_bound < self.batch_size: return

        batch_indices = np.random.choice(self.memory_bound, self.batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch_indices], dtype=torch.float32)
        new_states = torch.tensor(self.new_state_memory[batch_indices], dtype=torch.float32)
        rewards = torch.tensor(self.reward_memory[batch_indices], dtype=torch.float32)
        game_not_overs = torch.tensor(self.game_over_memory[batch_indices], dtype=torch.bool)
        actions = torch.tensor(self.actions_memory[batch_indices] , dtype=torch.int64)

        self.learn(states, actions, rewards, new_states, game_not_overs)
    
    def learn(self, states, actions, rewards, new_states, game_not_overs):
        if self.dual_learning:
            self.replace_model_counter += 1
            if self.replace_model_counter >= self.replace_model_interval:
                self.replace_model_counter = 0
                self.next_model.load_state_dict(self.eval_model.state_dict())
                if self.on_replace is not None: self.on_replace()

        states = states.to(self.device)
        actions = actions.to(self.device) # (batch_size, 2, 2)
        rewards = rewards.to(self.device)
        new_states = new_states.to(self.device)
        game_not_overs = game_not_overs.to(self.device)

        batch_size = len(game_not_overs)
        self.optimizer.zero_grad()
        q_evals = self.eval_model(states) # [B, out_channeles, H, W]
        q_nexts = self.next_model(new_states).view(batch_size, self.out_channels, -1) # [B, out_channeles, H*W]

        batch_index = torch.tensor(np.arange(batch_size, dtype=np.int32), dtype=torch.long, device=self.device)

        q_target = torch.clone(q_evals)
        if self.sequential_actions: # actions: [B, 3]
            q_nexts = q_nexts.view(batch_size, -1) # [B, out_channeles * H * W]
            q_target[batch_index, actions[:, 0], actions[:, 1], actions[:, 2]] = rewards + self.gamma * torch.max(q_nexts, dim=1)[0] * game_not_overs
        else:
            for i in range(self.out_channels):
                q_target[batch_index, i, actions[:, 0, i], actions[:, 1, i]] = rewards + self.gamma * torch.max(q_nexts[:, i], dim=1)[0] * game_not_overs

        loss = self.loss_fn(q_evals, q_target)
        loss.backward()
        self.optimizer.step()
        self.epsilon_step()

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

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
        for x in ['on_replace', 'random_action_resolver', 'eval_model', 'next_model', 'optimizer']:
            if x not in state and self.stripped: continue
            state.pop(x)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.on_replace = None
        if not hasattr(self, 'sequential_actions'): self.sequential_actions = False
        self.random_action_resolver = get_default_random_action_resolver(self.board_shape, self.out_channels, self.sequential_actions)
        if not hasattr(self, 'model_class'): self.model_class = DnDEvalModel # delete this line asap
        self.eval_model = self.model_class(self.in_channels, self.out_channels).to(self.device).train()
        self.next_model = self.model_class(self.in_channels, self.out_channels).to(self.device).eval()
        self.optimizer = torch.optim.Adam(self.eval_model.parameters())

class IdleDnDAgent():
    def choose_action_vector(self, state):
        new_coords = np.where(state[2])
        tys, txs = np.where(np.logical_or(state[0], state[1]) == 0)

        return ((new_coords[0][0], tys[0]), (new_coords[1][0], txs[0]))
