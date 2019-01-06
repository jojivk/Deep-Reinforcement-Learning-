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
score_show =1000
best_score =0.0
model_scores = [ 0.5994, 0.7045, 0.9003, 0.9720, 0.6528, 0.8361, 0.8069, 0.7988, 
                 1.1690, 0.8518, 0.8858, 1.3067, 0.7585, 1.0258, 1.2590, 0.7016,
                 1.0565, 1.1596, 1.1943, 0.4905, 0.5633, 1.0615, 0.9635, 0.7829,
                 1.2917, 1.0358, 1.2372, 1.3608, 0.9348]

###############################################################################
# Inittialize Unity Env for Reacher
# Get the brain from the env check no of agents,actions state etc..
###############################################################################
print("========================The Brain==========================================")
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="Tennis.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print("=========================Show env==========================================")
env_info = env.reset(train_mode=True)[brain_name]
print('Number of agents:', len(env_info.agents))
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print("Both States:")
print(states)
###############################################################################
# Define an Agent which is the class thaholds both reference and
# target n/w and saves episodes in mem
###############################################################################
print("===========================================================================")
from maddpg_agent import MultiAgent
agent = MultiAgent(agent_count=2, state_size=24, action_size=2, random_seed=13)


def loadModel(agent, ind, dirc, score) :
  fname_actor = 'checkpoint_actor.pth'
  fname_critic = 'checkpoint_critic.pth'
  x = '_'+str(ind)+'_'+str(score)
  fname_actor = dir + '/' + fname_actor + "."+ str(x)
  fname_critic = dir + '/' + fname_critic + "."+ str(x)
  
  agent.actor_local.load_state_dict(torch.load(fname_actor))
  agent.critic_local.load_state_dict(torch.load(fname_critic))

###############################################################################
# Load models for the test run
###############################################################################
def loadModels(agent, score) :
  for i in range(len(agent.agents)) :
    loadModel(agent.agents[i], i, "./models", score)
###############################################################################
# A function to check the saved model
###############################################################################
def test() :
  print("========================Test Model========================================")
  mscores = []
  for score in model_scores :
    print('Models with score :', score)
    loadModel(agent,score)
    scores =[]
    for i in range(12) :
      env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
      states = env_info.vector_observations                  # get the current state (for each agent)
      scores = np.zeros(num_agents)                          # initialize the score (for each agent)
      score0 = score1 =0
      while True:
          #actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
          actions = agent.act(states)                    # get the action from agent : has the DNN
          #actions = np.clip(actions, -1.5, 1.5)                  # all actions between -1 and 1
          env_info = env.step(actions)[brain_name]           # send all actions to tne environment
          next_states = env_info.vector_observations         # get next state (for each agent)
          rewards = env_info.rewards                         # get reward (for each agent)
          dones = env_info.local_done                        # see if episode finished
          states = next_states                               # roll over states to next time step

          score0 += np.array(rewards)[0]
          score1 += np.array(rewards)[1]
          if any(dones):
             break
      score = max(score0,score1)
      scores.append(score)              # save most recent scores
      print('\t\tScore (max over agents) from episode {}: {}'.format(i, np.max(scores)))
    mscores = np.mean(scores)
    print('\t Mean score from 12 episodes: {}'.format(mscore))
    print('\t --------------------------------------')
  return mscores
###############################################################################
# Call the trainng routine
###############################################################################
print("===========================================================================")
print("========================Network Training===================================")
scores = test()


