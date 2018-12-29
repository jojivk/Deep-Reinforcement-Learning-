import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#def hidden_init(layer):
#   if isinstance(layer, nn.Linear):
#      size = layer.weight.size()
#      fan_out = size[0] # number of rows
#      fan_in = size[1] # number of columns
#      variance = np.sqrt(2.0/(fan_in + fan_out))
#      layer.weight.data.normal_(0.0, variance)
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=256, dropout=0.5, lruSlope=0.1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bnorm1 = nn.BatchNorm1d(state_size)
        self.bnorm2 = nn.BatchNorm1d(fc1_units)
        self.dropout = nn.Dropout(dropout)
        self.LRU = nn.LeakyReLU(negative_slope = lruSlope)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        state = self.bnorm1(state)
        x = self.LRU(self.fc1(state))
        #x = self.dropout(x)
        x = self.bnorm2(x)
        x = self.LRU(self.fc2(x))
        #x = self.dropout(x)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed=0, fcs1_units=256, fc2_units=256, dropout=0.5, lruSlope= 0.1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bnorm1 = nn.BatchNorm1d(state_size)
        self.bnorm2 = nn.BatchNorm1d(fcs1_units)
        self.dropout = nn.Dropout(dropout)
        self.LRU = nn.LeakyReLU(negative_slope = lruSlope)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        state = self.bnorm1(state)
        xs = self.LRU(self.fcs1(state))
        #xs = self.dropout(xs)
        xs = self.bnorm2(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.LRU(self.fc2(x))
        #x = self.dropout(x)
        return self.fc3(x)
