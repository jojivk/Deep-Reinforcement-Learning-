

# 1. Problem 
   Train an agent to navigate a large world and collect yellow bananas while avoiding blue bananbas. 

   A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

   The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

	0 - move forward.
	1 - move backward.
	2 - turn left.
	3 - turn right.
   The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.The project environment is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page.

# 2. Frameworks
   The solution uses pytorch to build the training n/w and to deply the model network
# 3. Methodology
   The solution uses the DQN algorithm to optimize picking yellow bananas and avoiding blue ones.
   Please see the architecture for more details
# 4. Modules
   Bananas.py : The code for training banana collector.
   dqn_agent.py Holds the class for DQN networks the primary and target networks. The networks 
   model.py : The actual MLP for the DQN
   run_model.py : Runs the saved model
   
# 5. How to run
 ## 5.1 Dependencies
     You would need to install pytorch.
     	[pytorch] [https://github.com/pytorch/pytorch]
     and Unity ML-Agents environment.
        [Unity ML-Agents environment] (https://github.com/Unity-Technologies/ml-agents)
	
 ## 5.2 To run
     Change the Unit env depending on the OS to point to the Bananas App
     Change the file_name parameter to match the location of the Unity environment that you downloaded.

         Mac: "path/to/Banana.app"
         Windows (x86): "path/to/Banana_Windows_x86/Banana.exe"
         Windows (x86_64): "path/to/Banana_Windows_x86_64/Banana.exe"
         Linux (x86): "path/to/Banana_Linux/Banana.x86"
         Linux (x86_64): "path/to/Banana_Linux/Banana.x86_64"
         Linux (x86, headless): "path/to/Banana_Linux_NoVis/Banana.x86"
         Linux (x86_64, headless): "path/to/Banana_Linux_NoVis/Banana.x86_64"

     For instance, if you are using a Mac, then you downloaded Banana.app. If this file is in the same folder as the notebook, then the line below should appear as follows:

         env = UnityEnvironment(file_name="Banana.app")

