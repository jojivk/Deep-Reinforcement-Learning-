
# 1. Problem : Collaboration and Competition   

   Train multiple agents to play a game of Tennis. 

   In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a  
   reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal 
   of each agent is to keep the ball in play.

   The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives 
   its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and   
   jumping. The generated actions should be two continuous values, corresponding to movement to & fro from the net, and jumping.

   The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive 
   episodes, after taking the maximum over both agents). Specifically,
   
      •	After each episode, we add up the rewards that each agent received (without discounting), 
         to get a score for each agent. This yields 2 (potentially different) scores. 
         We then take the maximum of these 2 scores.
   
      •	This yields a single score for each episode.
         The environment is considered solved, when the average (over 100 episodes) of 
         those scores is at least +0.5.

   
   In this implementation a single MADDPG agent trains two DDPG agents. 

# 2. Frameworks
   The solution uses pytorch to build the training n/w and to deply the model network
   
# 3. Methodology
   The methodology used to solve this env is, [Multi Agent Deep Deterministic Policy Gradients (MADDPG](https://arxiv.org/abs/1706.02275) algorithm. In MADDPG, each agent(player is modeled based on [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) algorithm. They share some info regarding their states/actions while training, but none is shared when testing. In short, the solution uses the DDPG algorithm to optimize learning of two agents controlled based on MADDPG algorithm.  Please see the architecture for more details
   
  ## 3.1. Modules
     Tennis.py       : The code for training Unity Tennis Env.
     maddpg_agent.py : The file that holds two DDPG agents and trains them to play the game.
     ddpg_agent.py   : Holds the class for DDPG networks the primary and target (actor-critic) networks and replay buffer.
     model.py        : The actual MLP for the DDPG Actor and Critic Networks
     run_model.py    : Runs the saved model
   
# 4. How to run
 ## 4.1 Dependencies
  You would need to install 
   [pytorch](https://github.com/pytorch/pytorch)
  and 
   [Unity ML-Agents environment](https://github.com/Unity-Technologies/ml-agents)
  to run this.
  
  To set up your python environment to run the code in this repository, follow the 
  [instructions ](https://github.com/udacity/deep-reinforcement-learning#dependencies)
  in the [Udacity Deep Reinforcement Learning github repository](https://github.com/udacity/deep-reinforcement-learning)
  
  
 ## 4.2 To run
 
   ### Step 1: Activate the Environment
   If you haven't already, please follow the instructions in the DRLND GitHub repository to set up your Python environment. These          instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch,        the ML-Agents toolkit, and a few more Python packages required to complete the project.

   (For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other      versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM    such as Bootcamp or Parallels.

SPECIAL NOTE TO BETA TESTERS - please also download the p3_collab-compet folder from here and place it in the DRLND GitHub repository.

   ### Step 2: Download the Unity Environment
   For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can        download it from one of the links below. You need only select the environment that matches your operating system:

   Linux: click here
   Mac OSX: click here
   Windows (32-bit): click here
   Windows (64-bit): click here
   Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

   (For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit          version of the Windows operating system.

   
  Change the Unity env depending on the OS to point to the Tennis App
  Change the file_name parameter to match the location of the Unity environment that you downloaded.

         Mac: "path/to/Tennis.app"
         Windows (x86): "path/to/Tennis_Windows_x86/Tennis.exe"
         Windows (x86_64): "path/to/Tennis_Windows_x86_64/Tennis.exe"
         Linux (x86): "path/to/Tennis_Linux/Tennis.x86"
         Linux (x86_64): "path/to/Tennis_Linux/Tennis.x86_64"
         Linux (x86, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86"
         Linux (x86_64, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86_64"

  For instance, if you are using a Mac, then you downloaded Tennis.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
  
  ## 4.3 To train.
    Download following files
      1. Tennis.app.zip	
         The Unity Tennis app (Only for Mac). Untar this in Mac
      2. Tennis.py	 
          Model weights will be saved automatically once the target goal of 0.5+ is reached
      3. maddpg_agent.py
      4. ddpg_agent.py	
      5. model.py	
         
     To run training
      > python Tennis.py
      
  ## 4.4 To run,  the trained model 
      1. Download all the above files for training except Tennis.py, plus the below files
      2. run_model.py
      3. models/checkpoint*.pth*
         Files with trained weights
         
      To run trained models
         > python model.py
         
