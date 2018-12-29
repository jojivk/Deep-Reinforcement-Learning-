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

###############################################################################
# Inittialize Unity Env for Reacher
# Get the brain from the env check no of agents,actions state etc..
###############################################################################
print("========================The Brain==========================================")
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="Reacher.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print("=========================Show env==========================================")
env_info = env.reset(train_mode=True)[brain_name]
print('Number of agents:', len(env_info.agents))
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States length:', state_size)

###############################################################################
# Define an Agent which is the class thaholds both reference and
# target n/w and saves episodes in mem
###############################################################################
print("===========================================================================")
from ddpg_agent import Agent
agent = Agent(state_size=33, action_size=4, random_seed=0)

###############################################################################
# A function to check the env using random actions
###############################################################################
def random_test() :
  print("========================Random Test========================================")
  env_info = env.reset(train_mode=False)[brain_name] # reset the environment
  state = env_info.vector_observations[0]            # get the current state
  score = 0                                          # initialize the score
  while True:
    action = agent.act(state)                     # select the action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

  print("Score: ", score)

###############################################################################
# Save model
###############################################################################
def save_model(lagent) :
   torch.save(lagent.actor_local.state_dict(), 'checkpoint_actor.pth')
   torch.save(lagent.critic_local.state_dict(), 'checkpoint_critic.pth')

###############################################################################
# A routine to train the network
#    Deep Deterministic Policy Gradient Algorithm.
###############################################################################
def ddpg(n_episodes=1800, max_t=1000):
    """
    Params
    ======
     n_episodes (int): maximum number of training episodes
     max_t (int): maximum timesteps per episode
     """ 
    reached = False
    largest = 0;
    mean_score = 0;
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=100)   # last 100 episodes
    for i_episode in range(1, n_episodes+1):
       env_info = env.reset(train_mode=True)[brain_name]
       state = env_info.vector_observations[0]            # get the current state
       score = 0
       agent.reset()                                      #reset agent also
       for t in range(max_t):
          action = agent.act(state)                      # select an action from the DDPG N/W
          env_info = env.step(action)[brain_name]        # Send the action to the env
          next_state = env_info.vector_observations[0]   # get the next state
          reward = env_info.rewards[0]                   # get the reward
          done = env_info.local_done[0]                  # see if episode finished
          agent.step(state, action, reward, next_state, done) # Record the step in agent 
          state = next_state                             # Set state to next state
          score += reward                                # update the score
          if done:                                       # exit loop if episode finished
             break
       scores_window.append(score)       # save recent score
       scores.append(score)              # save score
       mean_score = np.mean(scores_window);
       agent.update_lr(mean_score)
       print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
       if i_episode % score_show == 0:
          print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
          if reached and mean_score > largest :
             largest = mean_score
             save_model(agent)
       # Save the model after reached the target score
       if not reached and np.mean(scores_window)>=30.0:
          print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, mean_score))
          largest = mean_score
          save_model(agent)
          reached = True

    return scores
###############################################################################
# Call the trainng routine
###############################################################################
print("===========================================================================")
print("========================Network Training===================================")
#random_test()
scores = ddpg(700)

###############################################################################
# plot the scores
###############################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')

#fig.show()
plt.show(block=True)
