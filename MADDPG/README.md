
# 1. Problem : Collaboration and Competition   
   Train multiple agents to play a game of Tennis. 

   In this environment, the observation space consists of 24 variables corresponding to the position and velocity of the
   ball and racket for each individual agent. The generated actions are two continuous values, corresponding to movement 
   to & fro from the net, and jumping.   

   In this implementation a single MADDPG agent trains two DDPG agents. The task is episodic, and to solve the environment,    the agents must get an average score of 0.50 score over 100 consecutive episodes.

# 2. Frameworks
   The solution uses pytorch to build the training n/w and to deply the model network
   
# 3. Methodology
   The solution uses the DDPG algorithm to optimize learning of two agents controlled based on MADDPG algorithm
   Please see the architecture for more details
   
# 4. Modules
   Tennis.py : The code for training Unity Tennis Env.
   maddpg_agent.py : The file that holds two DDPG agents and trains them to play the game
   ddpg_agent.py Holds the class for DDPG networks the primary and target (actor-critic) networks and replay buffer.
   model.py : The actual MLP for the DDPG Actor and Critic Networks
   run_model.py : Runs the saved model
   
# 5. How to run
 ## 5.1 Dependencies
  You would need to install 
   [pytorch](https://github.com/pytorch/pytorch)
  and 
   [Unity ML-Agents environment](https://github.com/Unity-Technologies/ml-agents)
  to run this.
  
  To set up your python environment to run the code in this repository, follow the 
  [instructions ](https://github.com/udacity/deep-reinforcement-learning#dependencies)
  in the [Udacity Deep Reinforcement Learning github repository](https://github.com/udacity/deep-reinforcement-learning)
  
  
 ## 5.2 To run
  Change the Unity env depending on the OS to point to the Reacher App
  Change the file_name parameter to match the location of the Unity environment that you downloaded.

         Mac: "path/to/Tennis.app"
         Windows (x86): "path/to/Tennis_Windows_x86/Tennis.exe"
         Windows (x86_64): "path/to/Tennis_Windows_x86_64/Tennis.exe"
         Linux (x86): "path/to/Tennis_Linux/Tennis.x86"
         Linux (x86_64): "path/to/Tennis_Linux/Tennis.x86_64"
         Linux (x86, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86"
         Linux (x86_64, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86_64"

  For instance, if you are using a Mac, then you downloaded Reacher.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
  
  ## 5.3 To train.
      1. Tennis.app.zip	
         The Unity Tennis app (Only for Mac). Untar this in Mac
      2. Tennis.py	 
         This has the code to train the model. Model wiights will be saved automatically once the target goal of 0.5+ is reached
      3. maddpg_agent.py
         Holds the two agents that are trained to play the game and the replay buffer 
      4. ddpg_agent.py	
         File that has definition of DDPG class.
      5. model.py	
         File that has definition of the MLP model for Actor and Critic.
         
     To run training
      > python Tennis.py
      
  ## 5.4 To run,  the trained model 
      1. Download all the above files for training except Tennis.py, plus the below files
      2. run_model.py
         Runs the trained models from the weights
      3. models/checkpoint*.pth*
         Files with trained weights
         
      To run trained models
         > python model.py
         
