import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Normal, Categorical, OneHotCategorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR 
from .net import PolicyNet, ValueNet
from open_spiel.python.policy import Policy
import random

def kl_divergence(prob_a, prob_b):
    
    ori_shape = prob_b.shape
    prob_a = prob_a.reshape(-1)
    prob_b = prob_b.reshape(-1)
    prob_b[prob_a==0] = - np.inf

    prob_b = prob_b.reshape(ori_shape)
    prob_b = torch.softmax(prob_b, -1)
    prob_b = prob_b.reshape(-1)
    
    res = (prob_a[prob_a>0] * torch.log(prob_a[prob_a>0]/prob_b[prob_a>0])).sum()

    return res


class DQNAgent(Policy):

    batch_size = 512
    buffer_capacity = int(1e4)
    lr = 5e-3
    epsilon = 0.05
    gamma = 1
    target_update = 5

    def __init__(self, game, playerids, state_dim, hidden_dim,
                 action_dim, device, ckp_dir):

        super().__init__(game, playerids)
        self.device = device
        self.action_dim = action_dim
        self.q_net = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.tar_q_net = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        for param, target_param in zip(self.q_net.parameters(), self.tar_q_net.parameters()):
            target_param.data.copy_(param.data)
            
        self.buffer = []
        self.buffer_count = 0
        self.counter = 0
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
    
    def action_probabilities(self, state, player_id=None):

        cur_player = state.current_player()
        s = state.information_state_tensor(cur_player)
        legal_actions = state.legal_actions()
        all_act_probs, _, _ = self.select_action(s, legal_actions, noise=False)

        return dict(zip(legal_actions, all_act_probs[legal_actions]))
            
    def select_action(self, state, legal_actions=None, noise=True):
        state = torch.tensor(state).unsqueeze(0).to(self.device)

        if noise and (np.random.random() < self.epsilon):
            if not legal_actions is None:
                action = np.random.choice(legal_actions)
            else:
                action = np.random.randint(self.action_dim)
            action = torch.tensor(action)
            act_prob = torch.zeros(self.action_dim)
            act_prob[legal_actions] = 1 / len(legal_actions)
        else:
            if not legal_actions is None:
                with torch.no_grad():
                    all_act = self.q_net(state)[0]
                legal_act = torch.ones_like(all_act) * - np.inf
                legal_act[legal_actions] = all_act[legal_actions]
                action = legal_act.argmax().item()

                act_prob = torch.zeros_like(all_act)
                act_prob[legal_actions] = torch.softmax(all_act[legal_actions], -1)
            else:
                with torch.no_grad():
                    logits = self.q_net(state)
                    act_prob = torch.softmax(logits, -1)
                    action = logits.argmax().item()

        return act_prob, action, act_prob[action]
        
    def store_transition(self, transition):
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(transition)
            self.buffer_count += 1
        else:
            index = int(self.buffer_count % self.buffer_capacity)
            self.buffer[index] = transition
            self.buffer_count += 1
    
    def clean_buffer(self):
        self.buffer = []

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def update(self, anchor=None, div_weight=None):
        self.q_net = self.q_net.to(self.device)
        self.tar_q_net = self.tar_q_net.to(self.device)
        if len(self.buffer) < self.batch_size:
            return 
        sample_data = random.sample(self.buffer, self.batch_size)
        state = torch.tensor([t.state for t in sample_data]).to(self.device)
        action = torch.tensor([t.action for t in sample_data]).view(-1,1).to(torch.int64).to(self.device)
        next_state = torch.tensor([t.next_state for t in sample_data]).to(self.device)
        done = torch.tensor([t.done for t in sample_data]).to(self.device)
        reward = torch.tensor([t.reward for t in sample_data]).to(torch.float).to(self.device)
        actions_prob = torch.stack([t.actions_prob for t in sample_data], axis=0)
        legal_actions = torch.stack([t.legal_action for t in sample_data], axis=0)
        
        q_values = self.q_net(state).gather(1, action) 
        max_next_q_values = self.tar_q_net(next_state).max(1)[0]
        q_targets = (reward + self.gamma * max_next_q_values * (1 - done)).view(-1, 1)
        # q_targets = reward.view(-1, 1)

        dqn_loss = F.mse_loss(q_values, q_targets)
        if not anchor is None:
            logits = self.q_net(state)
            logits[legal_actions==0]=-np.inf
            act_prob_main = torch.softmax(logits, -1)
            # act_prob_main[actions_prob==0] = 0
            with torch.no_grad():
                logits_archor = anchor.q_net(state)
                logits_archor[legal_actions==0] = -np.inf
                act_prob_anchor = torch.softmax(logits_archor, -1)
            kl_loss = - kl_divergence(act_prob_main, act_prob_anchor) / self.batch_size * div_weight
        # print(dqn_loss.item())
        self.optimizer.zero_grad() 
        if not anchor is None:
            loss = dqn_loss + kl_loss
            loss.backward()
        else:        
            dqn_loss.backward()
        self.optimizer.step()

        self.q_net = self.q_net.to("cpu")
        self.tar_q_net = self.tar_q_net.to("cpu")
        
        self.counter += 1
        if self.counter % self.target_update == 0:
            for param, target_param in zip(self.q_net.parameters(), self.tar_q_net.parameters()):
                target_param.data.copy_(param.data)