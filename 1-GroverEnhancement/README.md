# Grover-like amplitude enhancement 

This folder contains the code implementing the Quantum Reinforcement Learning à la Grover. 

#### `groverMazeLearner`
This script contains the **class** defining the Quantum Learner (the agent). This class should be general enough to be used in all those environments where the task is to reach a final target state, given that evertything is discrete, and that both actions and states are labaled using nonnegative integers. 

#### `frozenLakeTest`
This script uses the `GroverMazeLearner` defined above, in the specific case of the FrozenLake environment. 

