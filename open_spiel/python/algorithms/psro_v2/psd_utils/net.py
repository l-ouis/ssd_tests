import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, init_ort=True):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, action_dim)
        if init_ort:
            torch.nn.init.orthogonal_(self.fc1.weight)
            torch.nn.init.orthogonal_(self.fc2.weight)
            torch.nn.init.orthogonal_(self.fc3.weight)
            torch.nn.init.orthogonal_(self.fc4.weight)
            torch.nn.init.orthogonal_(self.fc5.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return self.fc5(x)
        # return F.softmax(self.fc5(x), dim=1)

class PolicyNet2(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, init_ort=True):
        super(PolicyNet2, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, action_dim)
        if init_ort:
            torch.nn.init.orthogonal_(self.fc1.weight)
            torch.nn.init.orthogonal_(self.fc2.weight)
            torch.nn.init.orthogonal_(self.fc3.weight)
            torch.nn.init.orthogonal_(self.fc4.weight)
            torch.nn.init.orthogonal_(self.fc5.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # return self.fc5(x)
        return F.softmax(self.fc5(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, init_ort=False):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
        if init_ort:
            torch.nn.init.orthogonal_(self.fc1.weight)
            torch.nn.init.orthogonal_(self.fc3.weight)
            torch.nn.init.orthogonal_(self.fc4.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)