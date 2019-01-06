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

###############################################################################
# A function to check the env using random actions
###############################################################################

def random_test() :
  print("========================Random Test========================================")
  for i in range(7) :
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        #print("Actions :")
        #print(actions)
        actions = np.clip(actions, -1.5, 1.5)                  # all actions between -1 and 1
        #print("Clipped Actions :")
        #print(actions)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        #print(states[1])
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('\t\tScore (max over agents) from episode {}: {}'.format(i, np.max(scores)))

def save_modelac(agent,avgscore, index) :
   ascore = '_' + str(avgscore)
   aindex = '_' + str(index)
   astr = aindex+ascore
   torch.save(agent.actor_local.state_dict(),  './models/checkpoint_actor_local.pth'    + astr)
   torch.save(agent.critic_local.state_dict(), './models/checkpoint_critic_local.pth'  + astr)
   torch.save(agent.actor_target.state_dict(), './models/checkpoint_actor_target.pth'  + astr)
   torch.save(agent.critic_target.state_dict(), './models/checkpoint_critic_target.pth' + astr)

def save_models(agent,avg_score) :
   global best_score
   if avg_score > best_score :
      for i in range(len(agent.agents)) :
         save_modelac(agent.agents[i], avg_score, i)
      best_score = avg_score

###############################################################################
# A routine to train the network
###############################################################################
def maddpg(n_episodes=9000, max_t=1500, rand_episodes=2000):
    """Deep DPG Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    global score_show
    cleaned = False
    reached =False
    scores = []                        # list containing scores from each episode
    avg_scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    printed = False
    print("\t First", rand_episodes, "episodes, actions are random generated...")
    print("\t ---------------------------------------------------------")
    for i_episode in range(1, n_episodes+1):
        agent.reset()
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations             # get the current state
        score0 = 0
        score1 = 0
        for t in range(max_t):
            if (i_episode < rand_episodes) :
                actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
                actions = np.clip(actions, -1.5, 1.5)
                #print (actions)
            else :
                actions = agent.act(states)                    # get the action from agent : has the DNN
                if not printed :
                    print("")
                    print("\t Switiching to Network generated action...")
                    print("\t -----------------------------------------")
                    printed = True
                    score_show = 100
            #print(np.array(actions))
            #print("==========================")
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished

            agent.step(states, actions, rewards, next_states, dones) #Record the step in agent
            states = next_states                           # set the next state
            #cscore = max(np.array(rewards))                # Compte the rewards accumulated
            score0 += np.array(rewards)[0]
            score1 += np.array(rewards)[1]
            if any(dones):
                break
        score = max(score0,score1)
        scores_window.append(score)       # save most recent scores
        scores.append(score)              # save most recent scores
        mscore = np.mean(scores_window)
        avg_scores.append(mscore)
        print('\rEpisode {}\t Average Score (100 Epi) : {:.4f} '.format(i_episode, mscore), end="")
        if i_episode % score_show == 0:
            print('\rEpisode {}\t Average Score (100 Epi) : {:.2f}'.format(i_episode, np.mean(scores_window)))
            if reached :
               save_models(agent,mscore)
        if (not reached and np.mean(scores_window)>=0.5) :
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            reached = True
            save_models(agent,mscore)
    return scores, avg_scores

###############################################################################
# Call the trainng routine
###############################################################################
print("===========================================================================")
print("========================Network Training===================================")
#random_test()
scores, avg_scores = maddpg()

###############################################################################
# plot the scores
###############################################################################
fig = plt.figure()
#ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, np.arange(len(scores)), avg_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')

#fig.show()
plt.show(block=True)

