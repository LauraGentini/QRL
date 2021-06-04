import gym
from groverMazeLearner import GroverMazeLearner

# test
if __name__ == "__main__":
    # choose env
    envtest = gym.make("FrozenLake-v0", is_slippery=False)
    # init learner
    Elliot = GroverMazeLearner(envtest)
    # good hyperparms (hand-tuned)
    hyperp = {'k': 0.1,
              'alpha': 0.1,
              'gamma': 0.99,
              'eps': 0.01,
              'max_epochs': 3000,
              'max_steps': 15,
              'graphics': False}
    # set hyperparms
    Elliot.set_hyperparams(hyperp)

    # TRAIN
    trajectories = Elliot.train()

    # Show trajectories
    for key in trajectories.keys():
        print(key, trajectories[key])

    # final state values
    print(Elliot.state_vals.reshape((4, 4)))

    # grover flags
    for state, flag in enumerate(Elliot.grover_steps_flag):
        print(state, '\t', flag)

    # state-action circuits
    for s, circ in enumerate(Elliot.acts_circs):
        print('action circuit for state ', s)
        print(circ.draw())

