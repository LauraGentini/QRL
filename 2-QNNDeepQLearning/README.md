# Deep Q-Learning with Quantum Neural Networks

In this folder you can find the code for implementing the Quantum Reinforcement Learning approach based on Deep Q-Learning, but using Quantum Neural Networks (i.e. Parametrized Quantum Circuits) instead of classical deep neural networks. 

## File organization

#### Main codes  
The main code, including all the theory and explanations, is contained in the `Deep_Q_Learning.ipynb` Jupyter Notebook. In the `DQN-analysis.ipynb`, you can find essentially the same things of the previous notebook, but without the explanations. In addition, in this notebook we show how we can test the performances of the Quantum Agent Elliot in the presence of noise coming from stochastic measurement outcomes. The file `dqn_definitions` contains the same custom functions defined in `Deep_Q_Learning.ipynb`, but in the form of a python script in order to be easily imported in new notebooks, as is with `DQN-analysis.ipynb`.

#### Pre-trained weights  
Since the training of the agent is very computationally intensive (our simulations took ~24 hours to complete), we attach 3 pre-trained set of weights to be loaded in the quantum model: 
* `model_best_weights_6reps_longtraining.pth`: contains weights coming from a successful training run achieving perfect score;
* `model_best_weights_6reps_longtraining2.pth`: contains weights coming from a training run which did not converged to an optimal solution in the allowed number of training episodes;
* `model_best_weights_6reps_longtraining2.pth`: contains weights coming from a successful training run achieving perfect score; 

The files `training_rewards` and `training_rewards2` contains the rewards history during two different training runs: one failing to find an optimal policy (`training_rewards`), and the other instead reaching perfect score (`training_rewards2`).

#### Other media
In addition to the other files, you can find two plots of the trained agent, one showing the training process and the other the performances with shot noise. 

Also, we upload a cute example of the trained agent solving the environment. 

