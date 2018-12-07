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
# Inittialize Unity Env for Banana picking.
# Get the brain from the env check no of agents,actions state etc..
###############################################################################
print("========================The Brain==========================================")
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="Banana.app")

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
from dqn_agent import Agent
agent = Agent(state_size=37, action_size=4, seed=0)

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
# A routine to train the network
###############################################################################
def dqn(n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    reached =False
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                 # get the action from agent : has the DNN
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done) #Record the step in agent
            state = next_state                                  # set the next state
            score += reward                                     # Compte the rewards accumulated
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % score_show == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'saved/checkpoint.pth.'+str(i_episode))
               #Save the model after each score_show episodes.
        if not reached and np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            reached = True
    torch.save(agent.qnetwork_local.state_dict(), 'saved/checkpoint.pth') #save the final model
    return scores

###############################################################################
# Call the trainng routine
###############################################################################
print("===========================================================================")
print("========================Network Training===================================")
#random_test()
scores = dqn()

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
