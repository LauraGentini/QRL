import numpy as np
import gym
from gym import spaces
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)

#class QTicTacToeEnv(gym.Env):

class QTicTacToeEnv():
    def __init__(self) -> None:
        #super(QTicTacToeEnv, self).__init__()
        self.simulator = Aer.get_backend('statevector_simulator')
        self.circuit = QuantumCircuit(9,9)
        self.status = np.zeros(9)
        self.board = np.zeros(9)
        self.player = 1
        self.move_counter = 0
        self.turn = 1
        self.win = False
        self.draw = False

    def update_status(self, move, reg0):
        if move == 0:
            self.status[reg0] = 1
        elif move == 3:
            self.status[reg0] = 3

    def move(self, reg0, playermove):
        if playermove == 0:
            if self.player == 1:
                self.circuit.id(reg0)
            
            if self.player == 2:
                self.circuit.x(reg0)
                self.circuit.id(reg0)

        elif playermove == 1:
            self.circuit.h(reg0)
        
        elif playermove == 2:
            self.circuit.x(reg0)
        
        elif playermove == 3:
            self.circuit.measure(reg0, reg0)
        
        job = execute(self.circuit, self.simulator)
        result = job.result()
        output_state = result.get_statevector()

        return output_state

    def is_valid_move(self, playermove, reg0):
        if playermove == 0:
            if self.status[reg0] == 0:
                self.move_counter += 1
                self.update_status(playermove, reg0)
                return True
            else:
                #Invalid move: A qubit has been allocated to the box.
                return False
        
        elif playermove == 1 or playermove == 2:
            if self.status[reg0] == 1:
                self.move_counter += 1
                return True
            
            if self.status[reg0] == 0:
                #Invalid move: There is no qubit in the box.'
                return False
            
            if self.status[reg0] == 3:
                #Invalid move: The qubit has been measured.
                return False
        
        elif playermove == 3:
            if self.status[reg0] == 1 and self.move_counter == 0:
                self.move_counter += 2
                self.update_status(playermove, reg0)
                return True
            
            elif self.move_counter == 1:
                #Invalid move: You cannot perform a measurement after a unitary operation.
                return False
            
            elif self.status[reg0] == 0:
                #Invalid move: There is no qubit in the box.
                return False

            elif self.status[reg0] == 3:
                #Invalid move: The qubit has been measured.
                return False
        
        elif playermove == 4:
            self.move_counter = 2
            return True
        
        else:
            return False

    def measurement_result(self, output_state, measured_register, qubitnumber):
        for index, element in enumerate(output_state):
            if element != 0:
                ket = bin(index)[2:].zfill(qubitnumber)
                result = ket[qubitnumber - measured_register] #the ket is read from right to left(|987654321>)
                break
        return result

    def turn_counter(self):
        if self.move_counter == 2 and not self.win and not self.draw:
            self.move_counter = 0
            self.turn += 1
            self.player = 2 - (self.player + 1) % 2
    
    def check_for_win(self):
        b = self.board
        
        for p in range(1, 3):
            if ((b[0] == p and b[1] == p and b[2] == p) or # across the top
            (b[3] == p and b[4] == p and b[5] == p) or # across the middle
            (b[6] == p and b[7] == p and b[8] == p) or # across the bottom
            (b[0] == p and b[3] == p and b[6] == p) or # down the left side
            (b[1] == p and b[4] == p and b[7] == p) or # down the middle
            (b[2] == p and b[5] == p and b[8] == p) or # down the right side
            (b[0] == p and b[4] == p and b[8] == p) or # diagonal
            (b[2] == p and b[4] == p and b[6] == p)): # diagonal
                self.player = p
                return True
        return False

    def check_for_draw(self):
        draw = np.count_nonzero(self.status == 3)
        
        if draw == 9:
            return True
        else:
            return False

    def collapse_all(self):
        if self.turn == 10:
            for register, item in enumerate(self.status):
                if item == 1:
                    output_state = self.move(register, 3)
                    res = self.measurement_result(output_state, register, 9)
                    
                    if str(res) == '0':
                        self.board[register] = 1
                    else:
                        self.board[register] = 2
            
            self.win = self.check_for_win()

            if not self.win:
                self.draw = True

    def step(self, action, reg0):
        if not self.is_valid_move(action, reg0):
            return
        
        output_state = self.move(action, reg0)
        self.update_status(action, reg0)

        if action == 3:
            res = self.measurement_result(output_state, reg0, 9)
            
            if str(res) == '0':
                self.board[reg0] = 1
            else:
                self.board[reg0] = 2

            self.win = self.check_for_win()
        
        if not self.win:
            self.draw = self.check_for_draw()

        if self.win:
            print('Player {} won.'.format(self.player))

        self.collapse_all()
        self.turn_counter()

    def reset(self):
        self.status = np.zeros(9)
        self.board = np.zeros(9)
        self.player = 1
        self.move_counter = 0
        self.turn = 0
        self.win = False
        self.draw = False
    
    def render(self):
        #print(self.circuit)
        print('Status =', self.status)
        print('Board =', self.board)
        print('Turn:', self.turn)
        print("Player {}'s turn".format(self.player))
        print('Moves: {}'.format(self.move_counter))
        print('')

env = QTicTacToeEnv()

env.render()

#Player 1 initializes qubit at register 1
env.step(0, 1)

env.render()

#Player 1 initializes qubit at register 4
env.step(0, 4)

env.render()

#Player 2 measures qubit at register 4
env.step(3, 4)

env.render()

#Player 1 initializes qubit at register 7 and then skips
env.step(0, 7)
env.step(4, 7)

env.render()

#Player 2 measures qubit at register 1
env.step(3, 1)

env.render()

#Player 1 measures qubit at register 7
env.step(3, 7)