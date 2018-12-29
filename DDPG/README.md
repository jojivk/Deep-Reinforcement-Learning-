# 1. Problem 
   Train an agent to control a robotic arm  

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In this implementation a single agent version of the Reacher environment is solved. The task is episodic, and to solve the environment, the agent must get an average score of +30 score over 100 consecutive episodes.

# 2. Frameworks
   The solution uses pytorch to build the training n/w and to deply the model network
   
# 3. Methodology
   The solution uses the DDPG algorithm to optimize controlling of the double jointed arm.
   Please see the architecture for more details
   
# 4. Modules
   Reacher.py : The code for training banana collector.
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

 ## 5.2 To run
  Change the Unity env depending on the OS to point to the Reacher App
  Change the file_name parameter to match the location of the Unity environment that you downloaded.

         Mac: "path/to/Reacher.app"
         Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"
         Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"
         Linux (x86): "path/to/Reacher_Linux/Reacher.x86"
         Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"
         Linux (x86, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86"
         Linux (x86_64, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86_64"

  For instance, if you are using a Mac, then you downloaded Reacher.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
