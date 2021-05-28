__author__ = 'sgruba'

import gym
from qiskit import *
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
import numpy as np
from math import ceil

from qttt import QTicTacToeEnv

class Groverlearner:
    """
    Inits a quantum QLearner object.
    TODO: accept random env
    """
    def __init__(self, env):
        self.env = env
        self.obs_dim = 1
        self.acts_dim = self.env.action_space.n
        self.acts_reg_dim = ceil(np.log2(self.acts_dim))
        self.state = self.env.reset()
        self.action = 0
        self.state_vals = {str(self.state): 0.}
        self.grover_steps = {str(self.state): np.zeros(self.acts_dim, dtype=int)}
        self.grover_steps_flag = {str(self.state): np.zeros(self.acts_dim, dtype=bool)}
        self.hyperparams = {'k': -1, 'alpha': 0.05, 'gamma': 0.99}

        self.grover_ops = self._init_grover_ops()
        self.acts_circs = self._init_acts_circs()
        self.SIM = Aer.get_backend('qasm_simulator')

    def set_hyperparams(self, hyperdict):
        self.hyperparams = hyperdict

    def _new_state_check(self, newstate):
        if str(newstate) in self.state_vals.keys():
            return
        else:
            self.state_vals[str(newstate)] = 0.
            self.grover_steps[str(newstate)] = np.zeros(self.acts_dim, dtype=int)
            self.grover_steps_flag[str(newstate)] = np.zeros(self.acts_dim, dtype=bool)
            self._append_new_circ(newstate)

    def _init_acts_circs(self):
        circs = {str(self.state): QuantumCircuit(self.acts_reg_dim)}
        for _, c in circs.items():
            c.h(list(range(self.acts_reg_dim)))
        return circs

    def _append_new_circ(self, state):
        self.acts_circs[str(state)] = QuantumCircuit(self.acts_reg_dim)
        self.acts_circs[str(state)].h(list(range(self.acts_reg_dim)))

    def _update_statevals(self, reward, new_state):
        self.state_vals[str(self.state)] += self.hyperparams['alpha']\
                                            * (reward + self.hyperparams['gamma']*self.state_vals[str(new_state)]
                                                - self.state_vals[str(self.state)])

    def _eval_grover_steps(self, reward, new_state):
        steps_num = int(self.hyperparams['k']*(reward + self.state_vals[str(new_state)]))
        return steps_num if steps_num <= 1 else 1

    def _init_grover_ops(self):
        states_binars = [format(i, '0{}b'.format(self.acts_reg_dim)) for i in range(self.acts_dim)]
        targ_states = [Statevector.from_label(s) for s in states_binars]
        grops = [GroverOperator(oracle=ts) for ts in targ_states]
        return [g.to_instruction() for g in grops]

    def _run_grover(self):
        # deploy grover ops on acts_circs
        gsteps = self.grover_steps[str(self.state)][self.action]
        circ = self.acts_circs[str(self.state)]
        op = self.grover_ops[self.action]
        for _ in range(gsteps):
            circ.append(op, list(range(self.acts_reg_dim)))
        self.acts_circs[str(self.state)] = circ  # TODO: check if useless
        # once taken, reset gsteps
        # self.grover_steps[self.state, self.action] = 0

    # def _run_grover_bool(self):
    #     # deploy grover ops on acts_circs
    #     flag = self.grover_steps_flag[self.state, :]
    #     gsteps = self.grover_steps[self.state, self.action]
    #     circ = self.acts_circs[self.state]
    #     op = self.grover_ops[self.action]
    #     if not flag.any():
    #         for _ in range(gsteps):
    #             circ.append(op, list(range(self.acts_reg_dim)))
    #     if gsteps >= 1 and not flag.any():
    #         self.grover_steps_flag[self.state, self.action] = True
    #     self.acts_circs[self.state] = circ  # TODO: check if useless

    def _take_action(self):
        action = self.acts_dim + 1
        while action >= self.acts_dim:
            circ = self.acts_circs[str(self.state)]
            circ_tomeasure = circ.copy()
            circ_tomeasure.measure_all()
            # circ_tomeasure = transpile(circ_tomeasure)
            # print(circ.draw())
            job = execute(circ_tomeasure, backend=self.SIM, shots=1)
            result = job.result()
            counts = result.get_counts()
            action = int((list(counts.keys()))[0], 2)
        return action


# test
if __name__ == "__main__":

    def train(env, pl1, pl2, hyperparams):

        traj_dict = {}
        stats = {"Pl1 wins": 0, "Pl2 wins": 0, "Draws": 0}
        # set initial max_steps
        gamelen = hyperparams['game_length']

        for epoch in range(hyperparams['max_epochs']):
            if epoch % 10 == 0:
                print("Processing epoch {} ...".format(epoch))
            # reset env
            state = env.reset()
            # init list for traj
            traj = [state]

            if hyperparams['graphics']:
                env.render()
            for step in range(gamelen):
                print('\rTurn {0}/{1}'.format(step, gamelen))
                # pl1 goes first, then pl2
                for player in (pl1, pl2):
                    player._new_state_check(state)
                    player.state = state
                    # Select action
                    action = player._take_action()  #self._run_grover_bool()
                    player.action = action
                    # take action
                    new_state, reward, done = env.step(action)
                    player._new_state_check(new_state)
                    player.state = state
                    # print('REWARD: ', reward)
                    # update statevals and grover steps
                    player._update_statevals(reward, new_state)
                    # NOT SURE IF RIGHT
                    player.grover_steps[str(state)][action] = player._eval_grover_steps(reward, new_state)
                    # amplify amplitudes with zio grover
                    player._run_grover()
                    # render if curious
                    if hyperparams['graphics']:
                        env.render()
                    # save transition
                    traj.append(new_state)
                    state = new_state

            # measure and observe outcome
            final = env.collapse_board()
            print("Observed board state: ", final)
            winner = env.check_end(final)
            if winner == 1:
                stats["Pl1 wins"] += 1
                pl1._new_state_check(state)
                pl1._update_statevals(100, state)
                pl2._new_state_check(state)
                pl2._update_statevals(-10, state)
            elif winner == 2:
                stats["Pl2 wins"] += 1
                pl2._new_state_check(state)
                pl2._update_statevals(100, state)
                pl1._new_state_check(state)
                pl1._update_statevals(-10, state)
            else:
                stats["Draws"] += 1
                pl1._new_state_check(state)
                pl1._update_statevals(-5, state)
                pl2._new_state_check(state)
                pl2._update_statevals(-5, state)

            traj_dict['epoch_{}'.format(epoch)] = traj

        # return trajectories
        return traj_dict, stats


    board_dim = 2
    # game_length = 5
    env = QTicTacToeEnv(board_dim)

    player_1 = Groverlearner(env)
    player_2 = Groverlearner(env)

    game_hyperparms = {'max_epochs': 100,
                       'game_length': 5,
                       'graphics': False}

    player_hyperparms = {'k': 0.1, 'alpha': 0.05, 'gamma': 0.99}
    player_1.set_hyperparams(player_hyperparms)
    player_2.set_hyperparams(player_hyperparms)

    game_trajectories, game_stats = train(env, player_1, player_2, game_hyperparms)
    # print(game_trajectories)
    print(game_stats)
    print(player_1.state_vals)
    print(player_1.grover_steps)
