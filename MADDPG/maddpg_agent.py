#################################################################
# DDPG: ddpg_agent.py
#################################################################
import numpy as np
import random
import copy
from collections import namedtuple, deque
from ddpg_agent import Agent

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TRAIN_SIZE = BATCH_SIZE * 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_count, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.started_learning = False;
        self.cur_step =0
        self.agent_count = agent_count
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(agent_count, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.agents =[]
        for i in range(agent_count) :
           agent = Agent(agent_count,state_size, action_size, random_seed, i)
           self.agents.append(agent)

    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if max(np.array(rewards)) > 0.0000 or self.started_learning == True :
          self.memory.add(states, actions, rewards, next_states, dones)
        #print("\t\t\t", np.array(actions),np.array(rewards))

        #if len(self.memory) > 512 or max(rewards) > 0.01:
        #   self.started_learning = True
        # Learn, if enough samples are available in memory
        #self.cur_step =self.cur_step + 1
        if len(self.memory) > TRAIN_SIZE :
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            self.started_learning = True

    def getinfo(self) :
        return len(self.memory)

    def clear_mem(self) :
        self.memory.clear()

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        acts = []
        for i in range(len(self.agents)) :
            agent =self.agents[i]
            action=agent.act(states[i], add_noise)
            acts.append(action)
        return acts

    def reset(self):
        for agent in self.agents :
          agent.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        next_acts =[]
        next_preds =[]
        for i in range(len(self.agents)) :
            nacts, npreds = self.agents[i].get_actor_next_actions(states[i], next_states[i])
            next_acts.append(nacts)
            next_preds.append(npreds)

        for i in range(len(self.agents)) :
            ex = states[i],actions,rewards[i], next_states[i], dones[i]
            self.agents[i].learn(ex, next_acts, next_preds, gamma)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, agent_count, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.agent_count = agent_count
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        allstates =[]
        allactions =[]
        allrewards = []
        allnext_states =[]
        alldones =[]
        for i in range(self.agent_count) :
          states = torch.from_numpy(np.vstack([e.states[i] for e in experiences if e is not None])).float().to(device)
          actions = torch.from_numpy(np.vstack([e.actions[i] for e in experiences if e is not None])).float().to(device)
          rewards = torch.from_numpy(np.vstack([e.rewards[i] for e in experiences if e is not None])).float().to(device)
          next_states = torch.from_numpy(np.vstack([e.next_states[i] for e in experiences if e is not None])).float().to(device)
          dones = torch.from_numpy(np.vstack([e.dones[i] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

          allstates.append(states)
          allactions.append(actions)
          allrewards.append(rewards)
          allnext_states.append(next_states)
          alldones.append(dones)

        return (allstates, allactions, allrewards, allnext_states, alldones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def clear(self) :
        for i in range(len(self.memory)) :
          ex = self.memory[i]
          states, actions, rewards, next_states, dones = ex
          rwd = np.array(rewards)
          mx = max(rwd)
          #print(rwd, mx)
          if max(np.array(rewards)) <= 0.0000 :
              print("Remx")
              del self.memory[i]
          
          #self.memory.clear()
