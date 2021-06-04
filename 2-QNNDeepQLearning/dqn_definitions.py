"""

Python script containing functions definitions for the Quantum Deep-Q Learning approach,
used in the DQN-analysis Jupyter Notebook.
 
Prepared for Qiskit Hackathon Europe by:
Stefano, Paolo, Jani & Laura, 2021. 

"""

# General imports
import numpy as np
import matplotlib.pyplot as plt

# Qiskit Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import TwoLocal

# Qiskit imports
import qiskit as qk
from qiskit.utils import QuantumInstance

# Qiskit Machine Learning imports
import qiskit_machine_learning as qkml
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

# PyTorch imports
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import LBFGS, SGD, Adam, RMSprop


def encoding_circuit(inputs, num_qubits=4, *args):
    """
    Encode classical input data (i.e. the state of the enironment) on a quantum circuit. 
    To be used inside the `parametrized_circuit` function. 
    
    Args
    -------
    inputs (list): a list containing the classical inputs.
    num_qubits (int): number of qubits in the quantum circuit.
    
    Return
    -------
    qc (QuantumCircuit): quantum circuit with encoding gates.
    
    """

    qc = qk.QuantumCircuit(num_qubits)

    # Encode data with a RX rotation
    for i, data in enumerate(inputs):
        qc.rx(inputs[i], i)

    return qc


def parametrized_circuit(num_qubits=4, reuploading=False, reps=2, insert_barriers=True, meas=False):
    """
    Create the Parameterized Quantum Circuit (PQC) for estimating Q-values.
    It implements the architecure proposed in Skolik et al. arXiv:2104.15084.
    
    Args
    -------
    num_qubit (int): number of qubits in the quantum circuit. 
    reuploading (bool): True if want to use data reuploading technique.
    reps (int): number of repetitions (layers) in the variational circuit. 
    insert_barrirerd (bool): True to add barriers in between gates, for better drawing of the circuit. 
    meas (bool): True to add final measurements on the qubits. 
    
    Return
    -------
    qc (QuantumCircuit): the full parametrized quantum circuit. 
    """

    qr = qk.QuantumRegister(num_qubits, 'qr')
    qc = qk.QuantumCircuit(qr)

    if meas:
        qr = qk.QuantumRegister(num_qubits, 'qr')
        cr = qk.ClassicalRegister(num_qubits, 'cr')
        qc = qk.QuantumCircuit(qr, cr)

    if not reuploading:

        # Define a vector containg Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Encode classical input data
        qc.compose(encoding_circuit(
            inputs, num_qubits=num_qubits), inplace=True)
        if insert_barriers:
            qc.barrier()

        # Variational circuit
        qc.compose(TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'circular',
                            reps=reps, insert_barriers=insert_barriers,
                            skip_final_rotation_layer=True), inplace=True)
        if insert_barriers:
            qc.barrier()

        # Add final measurements
        if meas:
            qc.measure(qr, cr)

    elif reuploading:

        # Define a vector containg Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Define a vector containng variational parameters
        θ = qk.circuit.ParameterVector('θ', 2 * num_qubits * reps)

        # Iterate for a number of repetitions
        for rep in range(reps):

            # Encode classical input data
            qc.compose(encoding_circuit(
                inputs, num_qubits=num_qubits), inplace=True)
            if insert_barriers:
                qc.barrier()

            # Variational circuit (does the same as TwoLocal from Qiskit)
            for qubit in range(num_qubits):
                qc.ry(θ[qubit + 2*num_qubits*(rep)], qubit)
                qc.rz(θ[qubit + 2*num_qubits*(rep) + num_qubits], qubit)
            if insert_barriers:
                qc.barrier()

            # Add entanglers (this code is for a circular entangler)
            qc.cz(qr[-1], qr[0])
            for qubit in range(num_qubits-1):
                qc.cz(qr[qubit], qr[qubit+1])
            if insert_barriers:
                qc.barrier()

        # (Optional) Add final measurements
        if meas:
            qc.measure(qr, cr)

    return qc


################
# PyTorch Code #
################

class encoding_layer(torch.nn.Module):
    def __init__(self, num_qubits=4):
        super().__init__()

        # Define weights for the layer
        weights = torch.Tensor(num_qubits)
        self.weights = torch.nn.Parameter(weights)
        # Initialization strategy
        torch.nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        """Forward step, as explained above."""

        if not isinstance(x, Tensor):
            x = Tensor(x)
        # if len(x.shape) == 1: # Not needed, may cause problems
        #    x = torch.unsqueeze(x, 0)

        x = self.weights * x
        x = torch.atan(x)

        return x


class exp_val_layer(torch.nn.Module):
    def __init__(self, action_space=2):
        super().__init__()

        # Define the weights for the layer
        weights = torch.Tensor(action_space)
        self.weights = torch.nn.Parameter(weights)
        # <-- Initialization strategy (heuristic choice)
        torch.nn.init.uniform_(self.weights, 35, 40)

        # Check that these masks take the vector of probabilities to <Z_0*Z_1> and <Z_2*Z_3>
        self.mask_ZZ_12 = torch.tensor(
            [1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1.], requires_grad=False)
        self.mask_ZZ_34 = torch.tensor(
            [-1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1.], requires_grad=False)

    def forward(self, x):
        """Forward step, as described above."""

        expval_ZZ_12 = self.mask_ZZ_12 * x
        expval_ZZ_34 = self.mask_ZZ_34 * x

        if len(x.shape) == 1:
            # , dim = 1, keepdim = True)
            expval_ZZ_12 = torch.sum(expval_ZZ_12)
            # , dim = 1, keepdim = True)
            expval_ZZ_34 = torch.sum(expval_ZZ_34)
            out = torch.cat((expval_ZZ_12.unsqueeze(0),
                            expval_ZZ_34.unsqueeze(0)))
        else:
            expval_ZZ_12 = torch.sum(expval_ZZ_12, dim=1, keepdim=True)
            expval_ZZ_34 = torch.sum(expval_ZZ_34, dim=1, keepdim=True)
            out = torch.cat((expval_ZZ_12, expval_ZZ_34), 1)

        return self.weights * ((out + 1.) / 2.)


#################
# Training code #
#################

def epsilon_greedy_policy(state, epsilon=0):
    """Manages the transition from the *exploration* to *exploitation* phase"""

    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        with torch.no_grad():
            Q_values = model(Tensor(state[np.newaxis])).numpy()
        return np.argmax(Q_values[0])


def sample_experiences(batch_size):
    """Sample some past experiences from the replay memory"""
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon):
    """Perform one action in the environment and register the state of the system"""
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


def sequential_training_step(batch_size):
    """
    Actual training routine. Implements the Deep Q-Learning algorithm.
    
    This implementation evaluates individual losses sequentially instead of using batches. 
    This is due to an issue in the TorchConnector, which yields vanishing gradients if it 
    is called with a batch of data (see https://github.com/Qiskit/qiskit-machine-learning/issues/100).
    
    Use this training for the quantum model. If using the classical model, you can use indifferently 
    this implementation or the batched one below. 
    """
    
    # Sample past experiences 
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # Evaluates the Target Q-values
    with torch.no_grad():
        next_Q_values = model(Tensor(next_states)).numpy()
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) *
                       discount_rate * max_next_Q_values)

    # Accumulate Loss (this is the only way it works. If batching data, gradients are vanishing)
    loss = 0.
    for j, state in enumerate(states):
        single_Q_value = model(Tensor(state))
        Q_value = single_Q_value[actions[j]]
        loss += (target_Q_values[j] - Q_value)**2

    # Evaluate the gradients and update the parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def training_step(batch_size):
    """
    This is exactly the same function as sequential_training_step, except that it 
    evaluates loss with batch of data, instead of using a for loop. 
    Can use this if training the classical model.
    """
    
    # Sample past experiences
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    
    # Evaluate Target Q-values
    with torch.no_grad():
        next_Q_values = model(Tensor(next_states)).numpy()
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = torch.nn.functional.one_hot(Tensor(actions).long(), n_outputs)

    # Evaluate the loss
    all_Q_values = model(Tensor(states))
    Q_values = torch.sum(all_Q_values * mask, dim=1, keepdims=True)
    loss = loss_fn(Tensor(target_Q_values), Q_values)

    # Evaluate the gradients and update the parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    