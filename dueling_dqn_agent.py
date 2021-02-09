import numpy as np
import random
from collections import namedtuple, deque

from model import DuelingQNetwork as QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

MIN_PROP = 1e-8         # minimum value to avoind zero property
ALPHA = 0.7             # prioritization coefficient p = p_i^ALPHA /sum(p_i^ALPHA)
BETA = 0.6              # weight coefficient w = (1/N * 1/P_i)^BETA
BETA_LIMIT =300         # after this, beta will be equal to 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, i_episode, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(i_episode)
                self.learn(i_episode, experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, i_episode, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, weights, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        losses  = (Q_expected - Q_targets).pow(2) * weights
        loss  = losses.mean()

        # Update the losses in memory
        self.memory.update_losses(i_episode, losses.cpu().data.numpy())

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "loss", "done"])
        self.seed = random.seed(seed)

        self.losses = np.array([MIN_PROP]*buffer_size)
        self.indx = np.array(batch_size, dtype=int)
        self.max_loss = MIN_PROP
        self.alpha = ALPHA
        self.beta = BETA
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, self.max_loss, done)
        self.memory.append(e)
    
    def update_losses(self, i_episode, losses):
        if (len(self.memory) >= self.batch_size):
            for i in range(self.batch_size):
                state, action, reward, next_state, self.max_loss, done = self.memory[self.indx[i]]
#                del self.memory[self.indx[i]]
#                self.memory.insert(self.indx[i], self.experience(state, action, reward, next_state, np.abs(losses[i][0]), done))
                self.memory[self.indx[i]] = self.experience(state, action, reward, next_state, np.abs(losses[i][0]), done)

            self.max_loss = max(self.max_loss,losses.max())
            self.alpha = ALPHA
            self.beta = BETA + (1-BETA)*min((i_episode-self.batch_size)/BETA_LIMIT, 1)

    
    def sample(self, i_episode):
        """Randomly sample a batch of experiences from memory."""

        losses = np.array([e.loss for e in self.memory])
        self.max_loss = losses.max()
        losses = (losses + MIN_PROP)**(self.alpha)
        loss_sum = losses.sum()
        p = losses/loss_sum
        self.indx=np.random.choice(list(range(len(self.memory))),self.batch_size, p=p, replace=False)

        experiences = [self.memory[i] for i in self.indx]

        states = torch.from_numpy(np.vstack([self.memory[i].state for i in self.indx])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in self.indx])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in self.indx])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in self.indx])).float().to(device)
        weights = torch.from_numpy(np.vstack([(p[i]*self.batch_size)**(-self.beta) for i in self.indx])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in self.indx]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, weights, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)