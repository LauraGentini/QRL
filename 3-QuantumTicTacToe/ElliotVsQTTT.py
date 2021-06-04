__author__ = 'QRL_team'

from qiskit import *
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
import numpy as np
from math import ceil

from qttt import QTicTacToeEnv


class GroverQuantumBoardLearner:
    """
    Inits a quantum QLearner object.
    Chosen environment must be discrete!
    """
    def __init__(self, env):
        self.env = env
        # in this approach we do not know in advance how many possible states are there,
        # they will be added during training
        self.obs_dim = 1
        # number of possible actions extracted from env
        self.acts_dim = self.env.action_space.n
        # evaluate number of needed qubits to encode actions
        self.acts_reg_dim = ceil(np.log2(self.acts_dim))
        # evaluate maximum number of grover steps
        self.max_grover_steps = int(round(np.pi/(4*np.arcsin(1./np.sqrt(2**self.acts_reg_dim))) - 0.5))
        # state variable
        self.state = self.env.reset()
        # action variable
        self.action = 0
        # init dictionary of quality values, str(state) is used for better comparison
        self.state_vals = {str(self.state): 0.}
        # init dict of grover steps for each state-action pair
        self.grover_steps = {str(self.state): np.zeros(self.acts_dim, dtype=int)}
        # init dict of flags to stop grover amplification, needed when acts_dim = 4
        self.grover_steps_flag = {str(self.state): np.zeros(self.acts_dim, dtype=bool)}
        # learner hyperparameters
        self.hyperparams = {'k': -1, 'alpha': 0.05, 'gamma': 0.99}
        # grover oracles
        self.grover_ops = self._init_grover_ops()
        # state-action circuits
        self.acts_circs = self._init_acts_circs()
        self.SIM = Aer.get_backend('qasm_simulator')

    def set_hyperparams(self, hyperdict):
        """
        Set new values for learner's hyperparameters
        :param hyperdict:
        :return: nthg
        """
        self.hyperparams = hyperdict

    def _new_state_check(self, newstate):
        """
        Checks if newstate was already observed
        :param newstate:
        :return: nthg
        """
        if str(newstate) in self.state_vals.keys():
            return
        else:
            self.state_vals[str(newstate)] = 0.
            self.grover_steps[str(newstate)] = np.zeros(self.acts_dim, dtype=int)
            self.grover_steps_flag[str(newstate)] = np.zeros(self.acts_dim, dtype=bool)
            self._append_new_circ(newstate)

    def _init_acts_circs(self):
        """
        Creates the state-action circuits and inits them in full superposition
        :return: dict of said circuits, keys are strings of state vectors
        """
        circs = {str(self.state): QuantumCircuit(self.acts_reg_dim)}
        for _, c in circs.items():
            c.h(list(range(self.acts_reg_dim)))
        return circs

    def _append_new_circ(self, state):
        """
        Inits a new state-action circuit
        :param state:
        :return:
        """
        self.acts_circs[str(state)] = QuantumCircuit(self.acts_reg_dim)
        self.acts_circs[str(state)].h(list(range(self.acts_reg_dim)))

    def _update_statevals(self, reward, new_state):
        """
        Bellman equation to update state values
        :param reward: the instantaneous reward received by the agent
        :param new_state: the new state visited by the agent
        :return:
        """
        self.state_vals[str(self.state)] += self.hyperparams['alpha']\
                                            * (reward + self.hyperparams['gamma']*self.state_vals[str(new_state)]
                                                - self.state_vals[str(self.state)])

    def _eval_grover_steps(self, reward, new_state):
        """
        Choose how many grover step to take based on instantaneous reward and value of new state
        :param reward: the instantaneous reward received by the agent
        :param new_state: the new state visited by the agent
        :return: number of grover steps to be taken,
        if it exceeds the theoretical optimal number the latter is returned instead
        """
        steps_num = int(self.hyperparams['k']*(reward + self.state_vals[str(new_state)]))
        return min(steps_num, self.max_grover_steps)

    def _init_grover_ops(self):
        """
        Inits grover oracles for the actions set
        :return: a list of qiskit instructions ready to be appended to circuit
        """
        states_binars = [format(i, '0{}b'.format(self.acts_reg_dim)) for i in range(self.acts_dim)]
        targ_states = [Statevector.from_label(s) for s in states_binars]
        grops = [GroverOperator(oracle=ts) for ts in targ_states]
        return [g.to_instruction() for g in grops]

    def _run_grover(self):
        """
        Deploy grover ops on acts_circs
        :return:
        """
        gsteps = self.grover_steps[str(self.state)][self.action]
        circ = self.acts_circs[str(self.state)]
        op = self.grover_ops[self.action]
        for _ in range(gsteps):
            circ.append(op, list(range(self.acts_reg_dim)))
        self.acts_circs[str(self.state)] = circ

    def _run_grover_bool(self):
        """
        Update state-action circuits based on evaluated steps
        :return:
        """
        flag = self.grover_steps_flag[str(self.state)]
        gsteps = self.grover_steps[str(self.state)][self.action]
        circ = self.acts_circs[str(self.state)]
        op = self.grover_ops[self.action]
        if not flag.any():
            for _ in range(gsteps):
                circ.append(op, list(range(self.acts_reg_dim)))
        if gsteps >= self.max_grover_steps and not flag.any():
            self.grover_steps_flag[str(self.state)][self.action] = True
        self.acts_circs[str(self.state)] = circ

    def _take_action(self):
        """
        Measures state-action circuit and chooses which action to take
        :return: int, chosen action
        """
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
        stats = {"Pl1 wins": [], "Pl2 wins": [], "Draws": []}
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
                    player.grover_steps[str(state)][action] = player._eval_grover_steps(reward, new_state)
                    # amplify amplitudes with zio grover
                    # player._run_grover()
                    player._run_grover_bool()
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
                stats["Pl1 wins"].append(epoch)
                pl1._new_state_check(state)
                pl1._update_statevals(100, state)
                pl2._new_state_check(state)
                pl2._update_statevals(-10, state)
            elif winner == 2:
                stats["Pl2 wins"].append(epoch)
                pl2._new_state_check(state)
                pl2._update_statevals(100, state)
                pl1._new_state_check(state)
                pl1._update_statevals(-10, state)
            else:
                stats["Draws"].append(epoch)
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

    player_1 = GroverQuantumBoardLearner(env)
    player_2 = GroverQuantumBoardLearner(env)

    game_hyperparms = {'max_epochs': 100,
                       'game_length': 4,
                       'graphics': False}

    player_hyperparms = {'k': 0.1, 'alpha': 0.05, 'gamma': 0.99}
    player_1.set_hyperparams(player_hyperparms)
    player_2.set_hyperparams(player_hyperparms)

    game_trajectories, game_stats = train(env, player_1, player_2, game_hyperparms)
    # print(game_trajectories)
    print(game_stats)
    print(player_1.state_vals)
    print(player_1.grover_steps)
