#!/usr/bin/env python3

import numpy as np
import pickle
from scipy import signal
from progressbar import ProgressBar
#  from numba import jit

#  import matplotlib
#  matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=16)



def spec_rad(mat):
    return max(abs(np.linalg.eigvals(mat)))


def random_matrix(n_rows, n_cols, a=-1, b=1, dist="uniform", density=None):
    """
    Generate a random matrix of dimension n_rows x n_cols drawing from a
    distribution given by 'dist', either uniform or normal. If uniform is chosen,
    a is the low and b the high value of the drawing interval. If normal is chosen,
    a is the mean and b the standart deviation. If the density parameter is given,
    then 1-density entries will be set to zero.
    """
    dim = (n_rows, n_cols)
    mask = np.random.uniform(0, 1, dim)
    mat = getattr(np.random, dist)(a, b, dim)
    if density is not None:
        mat[mask>density] = 0
    return mat


def input_matrix(n_rows, n_cols):
    mat = np.zeros((n_rows, n_cols))
    if n_cols > 0:
        for row in range(n_rows):
            col = np.random.randint(0, n_cols)
            mat[row,col] = np.random.normal(0, 1)
    return mat


def transfer_func(x):
    return np.tanh(x)


def inverse_transfer_func(x):
    return np.arctanh(x)


class ForceNetwork:
    """
    Implements FORCE learning as described in 'Sussillo, Abbott - 2009 - Generating
    coherent patterns of activity from chaotic neural networks'. The notation follows
    the one in the paper:
        I: input
        G: generator network
        z: output
        F: feedback network
        x: state of generator network
        y: state of the feedback network
        g_: weight scaling factor
        p_: sparsity (each element has a probability of 1-p to be held at zero)
    """
    def __init__(self, dimensions, weight_scales, densities, tau, alpha):
        # Timescale
        self.tau = tau

        # Dictioanry containing the layer sizes, possible keys are
        # 'I', 'G', 'z', 'F'
        self.dimensions = dimensions

        # Dictionaries containing parameters, possible keys are
        # 'GG', 'zG', 'Gz', 'GF', 'FF', 'FG'.
        # Densities have no 'Gz' and entries. Note that the paper uses
        # the word 'sparsities' instead of 'densities', but mathematically they
        # describe the latter
        self.weight_scales = weight_scales
        self.densities = densities

        # State of the generator network
        self.x = random_matrix(
                dimensions["G"],
                1,
                a=0,
                b=1,
                dist="normal",
        )
        #  self.x = np.zeros((dimensions["G"], 1))

        # State of the feedback network
        self.y = random_matrix(
                dimensions["F"],
                1,
                a=0,
                b=1,
                dist="normal",
        )
        #  self.y = np.zeros((dimensions["F"], 1))

        # The matrices J are the weights connecting the right index to the
        # left index, e.g. J_GI connects the input to the generator network
        self.J_GG = random_matrix(
                dimensions["G"],
                dimensions["G"],
                a=0,
                b=np.sqrt(1 / (densities["GG"]*dimensions["G"])),
                dist="normal",
                density=densities["GG"]
        )
        self.J_GF = random_matrix(
                dimensions["G"],
                dimensions["F"],
                a=0,
                b=np.sqrt(1 / (densities["GF"]*dimensions["F"])),
                dist="normal",
                density=densities["GG"]
        )
        self.J_FG = random_matrix(
                dimensions["F"],
                dimensions["G"],
                a=0,
                b=np.sqrt(1 / (densities["FG"]*dimensions["G"])),
                dist="normal",
                density=densities["FG"]
        )
        self.J_FF = random_matrix(
                dimensions["F"],
                dimensions["F"],
                a=0,
                b=np.sqrt(1 / (densities["FF"]*dimensions["F"])),
                dist="normal",
                density=densities["FF"]
        )

        self.J_Gz = random_matrix(
                dimensions["G"],
                dimensions["z"],
                a=-1,
                b=1,
                dist="uniform",
        )
        # In the paper, densities are applied to each column of the readout matrix,
        # as they describe each readout by a single vector w. This should probably be
        # changed to be consistend with the paper
        self.J_zG = random_matrix(
                dimensions["z"],
                dimensions["G"],
                a=0,
                b=np.sqrt(1 / (densities["zG"]*dimensions["G"])),
                dist="normal",
                density=densities["zG"]
        )

        self.J_FI = input_matrix(dimensions["F"], dimensions["I"])
        self.J_GI = input_matrix(dimensions["G"], dimensions["I"])

        # Running estimate of the inverse of the correlation matrix of the network
        # rates r (plus regularization term), necessary for learning update
        self.P = np.eye(dimensions["G"]) / alpha

    def update(self, input, dt):
        # See "Experimental Procedures"
        r = transfer_func(self.x)
        s = transfer_func(self.y)
        z = self.J_zG @ r

        dx = (dt / self.tau) * (
            -self.x +
            self.weight_scales["GG"] * self.J_GG @ r +
            self.weight_scales["Gz"] * self.J_Gz @ z +
            self.weight_scales["GF"] * self.J_GF @ s +
            self.J_GI @ input
        )

        dy = (dt / self.tau) * (
            -self.y +
            self.weight_scales["FF"] * self.J_FF @ s +
            self.weight_scales["FG"] * self.J_FG @ r +
            self.J_FI @ input
        )

        self.x += dx
        self.y += dy

    def learning_step(self, input, f):
        self.update(input, self.tau)
        r = transfer_func(self.x)
        z = self.J_zG @ r
        #  errors = (z - f)[:,0]
        errors = z - f

        # In the paper they say P and J_zG get updated simultaneously, but starting with P(0)
        # leads to numerical problems -> update P before J_zG
        self.P = self.P - ((self.P @ r) @ (r.transpose() @ self.P)) / (1 + r.transpose() @ self.P @ r)
        #  for i, error in enumerate(errors):
            #  self.J_zG[i,:] = self.J_zG[i,:] - (error * self.P @ r)[:,0]
        weight_change = - errors * ((self.P @ r).repeat(self.dimensions["z"], 1)).T
        self.J_zG = self.J_zG + weight_change
        #  dw = np.linalg.norm(error * self.P @ r[:,0])
        dw = np.linalg.norm(weight_change)
        return dw

    def train(self, input_sequence, output_sequence):
        bar = ProgressBar(max_value=len(input_sequence))
        for input, f in bar(zip(input_sequence, output_sequence)):
            self.learning_step(input, f)

    def eval(self, input):
        self.update(input)
        r = transfer_func(self.x)
        z = self.J_zG @ r
        return z


def mat_vec(mat, vec, coeff):
    return coeff * mat @ vec


def get_pos2d(theta_1, theta_2):
    l_1 = 1
    l_2 = 1
    x = l_1 * np.cos(theta_1) + l_2 * np.cos(theta_1 + theta_2)
    y = l_1 * np.sin(theta_1) + l_2 * np.sin(theta_1 + theta_2)
    return x, y


def get_angles2d(x, y):
    l_1 = 1
    l_2 = 2
    print(1 - ((x**2 + y**2 - l_1**2 - l_2**2) / (2*l_1*l_2))**2)
    theta_2 = np.arctan2(
        np.sqrt(1 - ((x**2 + y**2 - l_1**2 - l_2**2) / (2*l_1*l_2))**2),
        (x**2 + y**2 - l_1**2 - l_2**2) / (2*l_1*l_2)
    )
    theta_1 = np.arctan2(x, y) - np.arctan2(l_2 * np.sin(theta_2), l_1 + l_2 * np.cos(theta_2))
    return theta_1, theta_2


if __name__ == "__main__":
    # Network setup
    dimensions = {"I": 1, "G": 300, "z": 2, "F": 1}
    weight_scales = {"GG": 1.5, "zG": 1.0, "Gz": 1.0, "GF": 0, "FF": 0, "FG": 0}
    densities = {"GG": 0.1, "zG": 1.0, "GF": 1.0, "FF": 1.0, "FG": 1.0}
    tau = 10e-3 # milliseconds
    alpha = 1.0
    network = ForceNetwork(dimensions, weight_scales, densities, tau, alpha)

    # Idle network activity before training
    output = []
    sample_activities = [[] for _ in range(10)]
    weight_changes = []

    print("\n--- Idle network activity before training ---")
    bar = ProgressBar(max_value=1000)
    for _ in bar(range(1000)):
        weight_changes.append(0)
        network.update(np.array([[0]]), 10e-3)
        output.append(network.J_zG @ transfer_func(network.x))
        for i in range(10):
            sample_activities[i].append(network.x[i,0])

    # Training
    period = 0.1
    #  functions = [np.sin, signal.sawtooth, lambda x: np.sin(2*x) + 3 * np.cos(np.sin(x**2))*np.tanh(x)]
    functions = [np.sin]
    inputs = np.random.uniform(-1, 1, len(functions))

    print("\n--- Training ---")
    for function in functions:
        x = np.linspace(0, 1, 1000)
        #  y = function((2 * np.pi / period) * x)
        #  x = [np.array([[val]]) for val in x]
        #  y = [np.array([[val]]) for val in y]
        y1 = np.sin((2 * np.pi / period) * x)
        y2 = np.cos((2 * np.pi / period) * x)
        x = [np.array([[val]]) for val in x]
        y = [np.array([[val1], [val2]]) for (val1, val2) in zip(y1, y2)]

        bar = ProgressBar(max_value=len(x))
        for input, f in bar(zip(x, y)):
            dw = network.learning_step(np.array([[inputs[0]]]), f)
            weight_changes.append(dw)
            output.append(network.J_zG @ transfer_func(network.x))
            for i in range(10):
                sample_activities[i].append(network.x[i,0])

        for _ in range(100):
            weight_changes.append(0)
            network.update(np.array([[0]]), 10e-3)
            output.append(network.J_zG @ transfer_func(network.x))
            for i in range(10):
                sample_activities[i].append(network.x[i,0])


    #  with open('network', 'wb') as file:
        #  pickle.dump(network, file)

    #  with open('network', 'rb') as file:
        #  pickle.load(file)

    # Idle network activity after training
    print("\n--- Idle network activity after training ---")
    bar = ProgressBar(max_value=1000)
    for _ in bar(range(1000)):
        weight_changes.append(0)
        network.update(np.array([[0]]), 10e-3)
        output.append(network.J_zG @ transfer_func(network.x))
        for i in range(10):
            sample_activities[i].append(network.x[i,0])

    for j in range(len(functions)):
        print("\n--- Testing ---")
        bar = ProgressBar(max_value=1000)
        for _ in bar(range(1000)):
            weight_changes.append(0)
            network.update(np.array([[inputs[j]]]), 10e-3)
            output.append(network.J_zG @ transfer_func(network.x))
            for i in range(10):
                sample_activities[i].append(network.x[i,0])

        print("\n--- Idle network ---")
        bar = ProgressBar(max_value=1000)
        for _ in bar(range(1000)):
            weight_changes.append(0)
            network.update(np.array([[0]]), 10e-3)
            output.append(network.J_zG @ transfer_func(network.x))
            for i in range(10):
                sample_activities[i].append(network.x[i,0])


    # Plotting
    fig, axes = plt.subplots(6)
    axes[0].plot([val[0,0] for val in output], c='r')
    axes[0].plot([val[1,0] for val in output], c='g')
    for i in range(4):
        axes[i+1].plot(sample_activities[i], c='b')
    axes[5].plot(weight_changes)

    axes[0].set_ylabel('activity')
    axes[1].set_ylabel('neuron 1')
    axes[2].set_ylabel('neuron 2')
    axes[3].set_ylabel('neuron 3')
    axes[4].set_ylabel('neuron 4')
    axes[5].set_ylabel(r'$|\dot{w}|$')
    axes[5].set_xlabel('step')
    fig.tight_layout()
    plt.show()
