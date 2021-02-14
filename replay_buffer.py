import numpy as np
import random
from collections import namedtuple, deque

import torch

MIN_PROP = 1e-8         # minimum value to avoind zero property
ALPHA = 0.7             # prioritization coefficient p = p_i^ALPHA /sum(p_i^ALPHA)
BETA = 0.4              # weight coefficient w = (1/N * 1/P_i)^BETA
BETA_LIMIT =300         # after this, beta will be equal to 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizrdReplayBuffer:
    """Fixed-size buffer to store experience tuples modified to add prioritization."""

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
        """Add a new experience to memory."""
        if (len(self.memory) >= self.batch_size):
            for i in range(self.batch_size):
                state, action, reward, next_state, self.max_loss, done = self.memory[self.indx[i]]
                self.memory[self.indx[i]] = self.experience(state, action, reward, next_state, np.abs(losses[i][0]), done)

            self.max_loss = max(self.max_loss,losses.max())
            self.alpha = ALPHA
            self.beta = BETA + (1-BETA)*min(i_episode/BETA_LIMIT, 1)

    
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
        weights = np.vstack([(p[i]*self.batch_size)**(-self.beta) for i in self.indx])
        weights = torch.from_numpy(weights/weights.max()).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in self.indx]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, weights, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)