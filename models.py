import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64, fc3_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, feature_units=64, 
                 Advantage_units_1=64, Advantage_units_2=32, value_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            feature_units (int): Number of nodes in first hidden layer
            Advantage_units_1 (int): Number of nodes in first hidden layer in Advantage branch
            Advantage_units_2 (int): Number of nodes in second hidden layer in Advantage branch
            value_units (int): Number of nodes in first hidden layer in value branch
        """
        super(DuelingQNetwork, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_size, feature_units),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(feature_units, Advantage_units_1),
            nn.ReLU(),
            nn.Linear(Advantage_units_1, Advantage_units_2),
            nn.ReLU(),
            nn.Linear(Advantage_units_2, action_size)
        )
        
        self.value = nn.Sequential(
            nn.Linear(feature_units, value_units),
            nn.ReLU(),
            nn.Linear(value_units, 1)
        )
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
    