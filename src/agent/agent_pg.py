from .deep_q_network import DnDEvalModel
from .agent import passthrough_masker
from torch import nn
import numpy as np
import random
from typing import Optional
import pickle
import torch
import os

class DnDAgentPolicyGradient():
    BASE_ATTRS = ['model_class', 'model', 'board_shape', 'in_channels', 
                  'out_channels', 'device', 'sequential_actions']
    """Attributes that should not be stripped upon loading agent"""

    def __init__(self,
                 board_shape: tuple[int, int], 
                 in_channels: int, 
                 out_actions: int, 
                 lr: float=0.001,
                 momentum: float=0,
                 weight_decay: float=0,
                 gamma: float=0.9, 
                 memory_capacity: int=10000,
                 batch_size: int=64,
                 model_class: type[nn.Module]=DnDEvalModel,
                 sequential_actions: bool=False,
                 legal_moves_masker: callable=None) -> None:
        """"""
        self.in_channels = in_channels
        self.out_channels = out_actions
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.board_shape = board_shape
        self.replace_model_counter = 0
        self.model_class = model_class
        self.sequential_actions = sequential_actions
        self.legal_moves_masker = passthrough_masker if legal_moves_masker is None else legal_moves_masker

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_class(self.in_channels, self.out_channels).train().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        self.action_space = np.zeros((self.out_channels, *self.board_shape), dtype=np.int64)
        for i in range(self.action_space.size):
            self.action_space[np.unravel_index(i, self.action_space.shape)] = i
        self.action_space = self.action_space.flatten()
        
        self.discounts = self.gamma ** np.arange(self.memory_capacity, dtype=np.int64)
        self.memory_position = 0
        self.memory_bound = 0
        state_shape = (memory_capacity, self.in_channels, *board_shape)
        if sequential_actions:
            actions_shape = memory_capacity # action plane + 2 : [x, y] coordinates
        else:
            actions_shape = (memory_capacity, self.out_channels, 2) # 2 - [x, y] coordinates
        self.state_memory = np.zeros(state_shape, dtype=np.float32)
        self.actions_memory = np.zeros(actions_shape, dtype=np.int64)
        self.reward_memory = np.zeros(memory_capacity, dtype=np.float32)
        self.future_reward_memory = np.zeros(memory_capacity, dtype=np.float32)
        self.stripped = False

    def predict(self, state):
        with torch.no_grad(): # this just makes prediction a bit faster (I checked)
            return self.model(torch.tensor(state).to(self.device).unsqueeze(0)).detach().cpu().numpy()[0]
    
    def choose_action_vector(self, state):
        raise NotImplementedError()
    
    def predict_probabilities(self, state):
        with torch.no_grad():
            output = torch.flatten(self.model(torch.tensor(state).to(self.device).unsqueeze(0))[0])

            if filter: 
                self.last_mask = self.legal_moves_masker(state, self.out_channels)
                output -= ~torch.tensor(self.last_mask, dtype=torch.bool, device=self.device).view(-1) * 1e30

            activated = torch.nn.functional.softmax(output, dim=0)

        probabilities = activated.detach().cpu().numpy().reshape(self.out_channels, *self.board_shape)
        probabilities *= self.last_mask
        
        sum = np.sum(probabilities)
        return probabilities / sum

    def choose_single_action(self, state):
        probabilities = self.predict_probabilities(state).reshape(-1)

        index = np.random.choice(self.action_space, p=probabilities)
        return np.unravel_index(index, (self.out_channels, *self.board_shape))

    def save_agent(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'agent.pkl'), 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        
        torch.save(self.model.state_dict(), os.path.join(path, f'model.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, f'optimizer.pt'))

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
            agent.model = agent.model_class(agent.in_channels, agent.out_channels).to(agent.device).train()
            if not strip:
                agent.optimizer = torch.optim.Adam(agent.model.parameters())
        
        agent.model.load_state_dict(torch.load(os.path.join(path, f'model.pt')))
        if not strip:
            agent.optimizer.load_state_dict(torch.load(os.path.join(path, f'optimizer.pt')))
        
        if strip:
            for x in agent.__dict__.copy():
                if x in DnDAgentPolicyGradient.BASE_ATTRS: continue
                delattr(agent, x)

            agent.stripped = True
            agent.model.eval()

        for name, value in kwargs.items():
            assert hasattr(agent, name), f'attribute {name} does not exist'
            setattr(agent, name, value)

        return agent

    def memorize_episode(self, states, actions, rewards):
        ep_len = len(rewards)
        states = np.array(states)
        actions = np.ravel_multi_index(list(zip(*actions)), (self.out_channels, *self.board_shape))
        rewards = np.array(rewards)

        Gs = np.zeros_like(rewards, dtype=np.float32)
        for t in range(ep_len):
            Gs[t] = np.sum(rewards[t:] * self.discounts[:ep_len - t])

        memory_end = self.memory_position + ep_len
        if memory_end > self.memory_capacity:
            memory_left = self.memory_capacity - self.memory_position
            self.state_memory[self.memory_position:self.memory_capacity] = states[:memory_left]
            self.actions_memory[self.memory_position:self.memory_capacity] = actions[:memory_left]
            self.future_reward_memory[self.memory_position:self.memory_capacity] = Gs[:memory_left]
            self.reward_memory[self.memory_position:self.memory_capacity] = rewards[:memory_left]
            excess = ep_len - memory_left
            self.state_memory[:excess] = states[memory_left:]
            self.actions_memory[:excess] = actions[memory_left:]
            self.future_reward_memory[:excess] = Gs[memory_left:]
            self.reward_memory[:excess] = rewards[memory_left:]
            self.memory_position = excess
            self.memory_bound = self.memory_capacity
        else:
            self.state_memory[self.memory_position:memory_end] = states
            self.actions_memory[self.memory_position:memory_end] = actions
            self.future_reward_memory[self.memory_position:memory_end] = Gs
            self.reward_memory[self.memory_position:memory_end] = rewards
            self.memory_position += ep_len
            self.memory_bound = max(self.memory_position, self.memory_bound)

    def memorize(self, state, actions, reward):
        self.state_memory[self.memory_position] = state
        self.reward_memory[self.memory_position] = reward
        self.actions_memory[self.memory_position] = np.ravel_multi_index(actions, (self.out_channels, *self.board_shape))

        self.memory_position += 1
    
    def calculate_Gs_fast(self):
        ep_len = self.memory_position
        rewards = self.reward_memory[:ep_len]

        Gs = np.zeros_like(rewards)
        for t in range(ep_len):
            Gs[t] = np.sum(rewards[t:] * self.discounts[:ep_len - t])

        return Gs

    def random_learn(self):
        if self.memory_bound < self.batch_size: return

        batch_indices = np.random.choice(self.memory_bound, self.batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch_indices], dtype=torch.float32)
        actions = torch.tensor(self.actions_memory[batch_indices] , dtype=torch.int64)
        frewards = torch.tensor(self.future_reward_memory[batch_indices], dtype=torch.float32)

        self.learn(states, actions, frewards)

    def learn(self, states, actions, frewards):
        self.optimizer.zero_grad()

        states = states.to(self.device)
        actions = actions.to(self.device)
        frewards = frewards.to(self.device)

        # std, mean = torch.std_mean(frewards)
        # frewards = (frewards - mean) / (std if std.item() > 0 else 1)

        predictions = self.model(states).view(len(frewards), -1)
        predictions = predictions[:, :129]
        loss = torch.mean(self.loss_fn(predictions, actions) * frewards)

        loss.backward()
        self.optimizer.step()

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def __getstate__(self):
        state = self.__dict__.copy()
        for x in ['model', 'optimizer', 'legal_moves_masker']:
            state.pop(x)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = self.model_class(self.in_channels, self.out_channels).to(self.device).train()
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
