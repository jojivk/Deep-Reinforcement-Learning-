import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
#display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
score_show =100

print("========================The Brain==========================================")
from unityagents import UnityEnvironment
# please do not modify the line below
env = UnityEnvironment(file_name="Banana.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print("=========================Show env==========================================")
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

print("===========================================================================")
from dqn_agent import Agent
agent = Agent(state_size=37, action_size=4, seed=0)

# load the weights from file
def run_model(x=0, dir='.') :
  fname = 'checkpoint.pth'
  if not x==0 :
      fname = dir + '/' + fname + "."+ str(x)
  
  agent.qnetwork_local.load_state_dict(torch.load(fname))
  print("Running :", fname)
  for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score =0;
    for j in range(200):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
    print("Score =",score)

#for x in range(100,1800,100) :
run_model()
env.close()
