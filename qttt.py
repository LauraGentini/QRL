import numpy as np
import gym
from gym import spaces
from itertools import permutations
from qiskit import (
    QuantumCircuit,
    execute,
    Aer)

from qiskit.circuit.library import (
    HGate,
    XGate,
    CXGate)


# class QTicTacToeEnv(gym.Env):

class QTicTacToeEnv:
    def __init__(self, grid_size):
        # super(QTicTacToeEnv, self).__init__()
        self.simulator = Aer.get_backend('qasm_simulator')
        self.statevec_sim = Aer.get_backend('statevector_simulator')
        self.qnum = grid_size ** 2
        # self.game_len = game_len
        self.circuit = QuantumCircuit(self.qnum)
        self.moves = self._init_moves_dict()
        self.action_space = spaces.Discrete(len(self.moves))
        self.endings_lookuptable = self._init_outcomes_dict()
        self.status_id = ""
        # self.status = np.zeros(self.qnum)
        # self.board = np.zeros(self.qnum)
        # self.player = 1
        # self.move_counter = 0
        # self.turn = 1
        # self.win = False
        # self.draw = False
        # adding

    def _init_moves_dict(self):
        mvs_dict = {}
        mv_indx = 0
        for q in range(self.qnum):
            mvs_dict[mv_indx] = ([q], HGate())
            mv_indx += 1
            mvs_dict[mv_indx] = ([q], XGate())
            mv_indx += 1
        for (c, t) in permutations(list(range(self.qnum)), 2):
            mvs_dict[mv_indx] = ([c, t], CXGate())
            mv_indx += 1
        return mvs_dict

    def _win_check(self, board):
        d = int(np.sqrt(self.qnum))
        # transofrm board string to rows, cols and diags
        rows = [board[i*d:(i+1)*d] for i in range(d)]
        cols = ["".join([rows[i][j] for i in range(d)]) for j in range(d)]
        diags = ["".join([rows[i][i] for i in range(d)]), "".join([rows[i][d-i-1] for i in range(d)])]
        winner = 0
        cond_1 = bin(0)[2:].zfill(d)
        cond_2 = bin(2**d - 1)[2:].zfill(d)

        for line in [*rows, *cols, *diags]:
            if line == cond_1:
                if winner == 0 or winner == 1:
                    winner = 1
                elif winner == 2:
                    return 0  # because both players won
            elif line == cond_2:
                if winner == 0 or winner == 2:
                    winner = 2
                elif winner == 1:
                    return 0  # because both players won

        return winner

    def _init_outcomes_dict(self):
        out_dict = {1: [], 2: [], 0: []}
        # init all possible observed board states
        all_states = [bin(x)[2:].zfill(self.qnum) for x in range(2**self.qnum)]
        for state in all_states:
            winner = self._win_check(state)
            out_dict[winner].append(int(state, 2))

        return out_dict

    def move(self, action):
        self.status_id += "{}-".format(action)
        self.circuit.append(self.moves[action][1], self.moves[action][0])

    def _get_statevec(self):
        job = execute(self.circuit, self.statevec_sim)
        result = job.result()
        output_state = result.get_statevector()
        return np.around(output_state, decimals=2)

    def collapse_board(self):
        self.circuit.measure_all()
        job = execute(self.circuit, backend=self.simulator, shots=1)
        res = job.result()
        counts = res.get_counts()
        collapsed_state = int(list(counts.keys())[0][:self.qnum], 2)  # int((list(counts.keys()))[0], 2)
        return collapsed_state

    def check_end(self, board_state):
        if board_state in self.endings_lookuptable[1]:
            print("\nPlayer 1 wins!!!\n")
            return 1
        elif board_state in self.endings_lookuptable[2]:
            print("\nPlayer 2 wins!!!\n")
            return 2
        else:
            print("\nIt's a draw!\n")
            return 0

    def step(self, action):
        self.move(action)
        new_state = self._get_statevec()
        reward = -0.1
        return new_state, reward, False

    def reset(self):
        self.circuit = QuantumCircuit(self.qnum, self.qnum)
        self.circuit.h(list(range(self.qnum)))
        self.status_id = ""
        return self._get_statevec()

    # def render(self):
    #     print(self.circuit.draw())
    #     print('Status =', self.status)
    #     print('Board =', self.board)
    #     print('Turn:', self.turn)
    #     print("Player {}'s turn".format(self.player))
    #     print('Moves: {}'.format(self.move_counter))
    #     print('')


if __name__ == "__main__":
    board_dim = 2
    # game_length = 5
    env = QTicTacToeEnv(board_dim)
    env.reset()

    print(env._win_check('1110'))


