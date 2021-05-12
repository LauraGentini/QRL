__author__ = 'sgruba'

import gym
from qiskit import *
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
import numpy as np

#
# """Class reproducing DONG"""
#


class QuantumQlearner:
    """
    Inits a quantum QLearner object.
    TODO: accept random env
    """
    def __init__(self):
        self.env = gym.make("FrozenLake-v0", is_slippery=False)
        self.obs_dim = self.env.observation_space.n
        self.acts_dim = self.env.action_space.n
        self.acts_reg_dim = int(np.log2(self.acts_dim))  # TODO: handle the case acts_dim != 2**n
        # using a list rather than a dict (obs are integers)
        # self.acts_regs = [(QuantumRegister(self.acts_reg_dim, 'a{}'.format(i)),
        #                    ClassicalRegister(self.acts_reg_dim, 'c{}'.format(i))) for i in range(self.obs_dim)]
        self.state_vals = np.zeros(self.obs_dim)
        self.grover_steps = np.zeros((self.obs_dim, self.acts_dim), dtype=int)
        self.grover_steps_flag = np.zeros((self.obs_dim, self.acts_dim), dtype=bool)
        self.hyperparams = {'k': -1, 'alpha': 0.05, 'gamma': 0.99, 'eps': 0.01, 'max_epochs': 1000, 'max_steps': 100
                            , 'graphics': True}
        self.state = 0  # self.env.reset()
        self.action = 0
        self.grover_ops = self._init_grover_ops()
        self.acts_circs = self._init_acts_circs()
        self.SIM = Aer.get_backend('qasm_simulator')

    def set_hyperparams(self, hyperdict):
        self.hyperparams = hyperdict

    def _init_acts_circs(self):
        circs = [QuantumCircuit(self.acts_reg_dim, name='|as_{}>'.format(i)) for i in range(self.obs_dim)]
        for c in circs:
            c.h(list(range(self.acts_reg_dim)))
        return circs

    def _update_statevals(self, reward, new_state):
        self.state_vals[self.state] += self.hyperparams['alpha']*(reward
                                                                  + self.hyperparams['gamma']*self.state_vals[new_state]
                                                                  - self.state_vals[self.state])

    def _eval_grover_steps(self, reward, new_state):
        steps_num = int(self.hyperparams['k']*(reward + self.state_vals[new_state]))
        return steps_num if steps_num <= 1 else 1

    def _init_grover_ops(self):
        states_binars = [format(i, '0{}b'.format(self.acts_reg_dim)) for i in range(self.acts_dim)]
        targ_states = [Statevector.from_label(s) for s in states_binars]
        grops = [GroverOperator(oracle=ts) for ts in targ_states]
        return [g.to_instruction() for g in grops]

    def _run_grover(self):
        # deploy grover ops on acts_circs
        gsteps = self.grover_steps[self.state, self.action]
        circ = self.acts_circs[self.state]
        op = self.grover_ops[self.action]
        for _ in range(gsteps):
            circ.append(op, list(range(self.acts_reg_dim)))
        self.acts_circs[self.state] = circ  # TODO: check if useless
        # once taken, reset gsteps
        self.grover_steps[self.state, self.action] = 0

    def _run_grover_bool(self):
        # deploy grover ops on acts_circs
        flag = self.grover_steps_flag[self.state, :]
        gsteps = self.grover_steps[self.state, self.action]
        circ = self.acts_circs[self.state]
        op = self.grover_ops[self.action]
        if not flag.any():
            for _ in range(gsteps):
                circ.append(op, list(range(self.acts_reg_dim)))
        if gsteps >= 1 and not flag.any():
            self.grover_steps_flag[self.state, self.action] = True
        self.acts_circs[self.state] = circ  # TODO: check if useless

    def _take_action(self):
        circ = self.acts_circs[self.state]
        circ_tomeasure = circ.copy()
        circ_tomeasure.measure_all()
        # circ_tomeasure = transpile(circ_tomeasure)
        # print(circ.draw())
        job = execute(circ_tomeasure, backend=self.SIM, shots=1)
        result = job.result()
        counts = result.get_counts()
        action = int((list(counts.keys()))[0], 2)
        return action

    def train(self):

        # TODO: devise termination condition!!!!!

        """
        groverize and measure action qstate -> take corresp action
        obtain: newstate, reward, terminationflag
        update stateval, grover_iter
        for epoch in epochs until either max_epochs or termination or convergence criterion is reached
        :return:
        dictionary of trajectories
        """
        traj_dict = {}

        # set initial max_steps
        optimal_steps = self.hyperparams['max_steps']

        for epoch in range(self.hyperparams['max_epochs']):
            if epoch % 10 == 0:
                print("Processing epoch {} ...".format(epoch))
            # reset env
            self.state = self.env.reset()
            # init list for traj
            traj = [0]

            if self.hyperparams['graphics']:
                self.env.render()
            for step in range(optimal_steps):
                print('Taking step {0}/{1}'.format(step, optimal_steps), end='\r')
                # print('STATE: ', self.state)
                # Select action
                # self.action = self._run_grover()
                self.action = self._take_action()  #self._run_grover_bool()
                # take action
                new_state, reward, done, _ = self.env.step(self.action)
                if new_state == self.state:
                    reward -= 10
                    done = True
                if new_state == 15:
                    reward += 99
                    # update optimal traj len
                    optimal_steps = step + 1
                elif not done:
                    reward -= 1
                # print('REWARD: ', reward)
                # update statevals and grover steps
                self._update_statevals(reward, new_state)
                # NOT SURE IF RIGHT
                # if self.grover_steps[self.state, self.action] == 0:
                #     self.grover_steps[self.state, self.action] = self._eval_grover_steps(reward, new_state)
                self.grover_steps[self.state, self.action] = self._eval_grover_steps(reward, new_state)
                # amplify amplitudes with zio grover
                # self._run_grover()
                self._run_grover_bool()
                # render if curious
                if self.hyperparams['graphics']:
                    self.env.render()
                # save transition
                traj.append(new_state)
                # quit epoch if done
                if done:
                    break
                # move to new state
                self.state = new_state
                # print('STATE_VALS: ', self.state_vals)
                # print('GROVER_STEPS: ', self.grover_steps)

            traj_dict['epoch_{}'.format(epoch)] = traj

        # return trajectories
        return traj_dict


# test
if __name__ == "__main__":

    qlearner = QuantumQlearner()
    hyperp = {'k': 0.1,
              'alpha': 0.1,
              'gamma': 0.99,
              'eps': 0.01,
              'max_epochs': 5000,
              'max_steps': 15,
              'graphics': False}

    qlearner.set_hyperparams(hyperp)

    trajectories = qlearner.train()

    for key in trajectories.keys():
        print(key, trajectories[key])
        # if trajectories[key][-1] == 15:
        #     print(key, trajectories[key])

    print(qlearner.state_vals.reshape((4, 4)))
    for state, flag in enumerate(qlearner.grover_steps_flag):
        print(state, '\t', flag)

    for s, circ in enumerate(qlearner.acts_circs):
        print('action circuit for state ', s)
        print(circ.draw())

