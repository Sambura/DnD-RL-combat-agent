from .deep_q_network import DnDEvalModel
from torch import nn
import numpy as np
import random
from typing import Optional
import pickle
import torch
import os

def passthrough_filter(state, probs):
    return probs

class DnDAgentPolicyGradient():
    BASE_ATTRS = ['model_class', 'model', 'board_shape', 'in_channels', 
                  'out_channels', 'device', 'sequential_actions', 'action_space']
    """Attributes that should not be stripped upon loading agent"""

    def __init__(self,
                 board_shape: tuple[int, int], 
                 in_channels: int, 
                 out_actions: int, 
                 lr: float=0.001,
                 gamma: float=0.9, 
                 memory_capacity: int=10000,
                 batch_size: int=64,
                 model_class: type[nn.Module]=DnDEvalModel,
                 sequential_actions: bool=False,
                 legal_moves_filter: callable=None) -> None:
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
        self.legal_moves_filter = passthrough_filter if legal_moves_filter is None else legal_moves_filter

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_class(self.in_channels, self.out_channels).train().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.init_action_space()
        
        self.discounts = self.gamma ** np.arange(self.memory_capacity, dtype=np.int64)
        self.memory_position = 0
        state_shape = (memory_capacity, self.in_channels, *board_shape)
        if sequential_actions:
            actions_shape = memory_capacity # action plane + 2 : [x, y] coordinates
        else:
            actions_shape = (memory_capacity, self.out_channels, 2) # 2 - [x, y] coordinates
        self.state_memory = np.zeros(state_shape, dtype=np.float32)
        self.actions_memory = np.zeros(actions_shape, dtype=np.int64)
        self.reward_memory = np.zeros(memory_capacity, dtype=np.float32)
        self.stripped = False

    def init_action_space(self):
        self.action_space = np.zeros((self.out_channels, *self.board_shape), dtype=np.int64)
        for i in range(self.action_space.size):
            self.action_space[np.unravel_index(i, self.action_space.shape)] = i
        self.action_space = self.action_space.flatten()

    def predict(self, state):
        with torch.no_grad(): # this just makes prediction a bit faster (I checked)
            return self.model(torch.tensor(state).to(self.device).unsqueeze(0)).detach().cpu().numpy()[0]
    
    def choose_action_vector(self, state):
        raise NotImplementedError()
    
    def predict_probabilities(self, state, filter=True):
        with torch.no_grad():
            output = self.model(torch.tensor(state).to(self.device).unsqueeze(0))[0]
            activated = torch.nn.functional.softmax(torch.flatten(output), dim=0)

        probabilities = activated.detach().cpu().numpy().reshape(self.out_channels, *self.board_shape)
        if filter: probabilities = self.legal_moves_filter(state, probabilities)
        return probabilities / np.sum(probabilities)

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

        agent.init_action_space()

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

    def learn(self):
        self.optimizer.zero_grad()

        ep_len = self.memory_position
        states = torch.tensor(self.state_memory[:ep_len], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions_memory[:ep_len], dtype=torch.int64, device=self.device)
        rewards = self.reward_memory[:ep_len]

        Gs = np.zeros_like(rewards)
        for t in range(ep_len):
            Gs[t] = np.sum(rewards[t:] * self.discounts[:ep_len - t])

        # mean = np.mean(Gs)
        # std = np.std(Gs)
        # Gs = (Gs - mean) / (std if std > 0 else 1)
        Gs = torch.tensor(Gs, dtype=torch.float32, device=self.device)

        # print(Gs)

        predictions = self.model(states).view(ep_len, -1)
        losses = self.loss_fn(predictions, actions) * Gs
        loss = torch.mean(losses)
        loss.backward()
        self.optimizer.step()

        #for i in range(0, ep_len, self.batch_size):
        #    self.optimizer.zero_grad()
#
        #    predictions = self.model(states[i : i + self.batch_size]).view(-1, len(self.action_space))
        #    loss = torch.mean(self.loss_fn(predictions, actions[i : i + self.batch_size]) * Gs[i : i + self.batch_size])
        #    loss.backward()
        #    self.optimizer.step()

        self.memory_position = 0

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def __getstate__(self):
        state = self.__dict__.copy()
        for x in ['model', 'optimizer', 'legal_moves_filter']:
            state.pop(x)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = self.model_class(self.in_channels, self.out_channels).to(self.device).train()
        self.optimizer = torch.optim.Adam(self.model.parameters())
