#!/usr/bin/env python3

import numpy as np
import theano
#  theano.config.profile = True
theano.config.floatX = 'float32'
#  theano.config.allow_gc = False # Disable garbace collection to buffer intermediate results
import theano.tensor as T
import pickle
from scipy import signal
from progressbar import ProgressBar

#  import matplotlib
#  matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=16)



def spec_rad(mat):
    return max(abs(np.linalg.eigvals(mat)))


def random_matrix(n_rows, n_cols, a=-1, b=1, dist="uniform", density=None, name=None):
    """
    Generate a random matrix of dimension n_rows x n_cols drawing from a
    distribution given by 'dist', either uniform or normal. If uniform is chosen,
    a is the low and b the high value of the drawing interval. If normal is chosen,
    a is the mean and b the standart deviation. If the density parameter is given,
    then 1-density entries will be set to zero.
    """
    dim = (n_rows, n_cols)
    mask = np.random.uniform(0, 1, dim).astype(dtype=theano.config.floatX)
    mat = getattr(np.random, dist)(a, b, dim).astype(dtype=theano.config.floatX)
    if density is not None:
        mat[mask>density] = 0
    if name is not None:
        mat = theano.shared(mat, borrow=True, name=name)
    else:
        mat = theano.shared(mat, borrow=True)
    return mat


def input_matrix(n_rows, n_cols, name=None):
    mat = np.zeros((n_rows, n_cols), dtype=theano.config.floatX)
    if n_cols > 0:
        for row in range(n_rows):
            col = np.random.randint(0, n_cols)
            mat[row,col] = np.random.normal(0, 1)
    if name is not None:
        mat = theano.shared(mat, borrow=True, name=name)
    else:
        mat = theano.shared(mat, borrow=True)
    return mat


def transfer_func(x):
    return T.tanh(x)


def inverse_transfer_func(x):
    return T.arctanh(x)


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
    def __init__(self, dimensions, weight_scales, densities, dt, tau, alpha):
        #---------- DEFINE CONSTANTS ----------#

        # Timescale
        self.dt = dt
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

        #---------- BUILD MODEL ----------#

        # State of the generator network
        self.x = random_matrix(
                dimensions["G"],
                1,
                a=0,
                b=1,
                dist="normal",
                name="x",
        )
        r = transfer_func(self.x)

        # State of the feedback network
        self.y = random_matrix(
                dimensions["F"],
                1,
                a=0,
                b=1,
                dist="normal",
                name="y",
        )
        s = transfer_func(self.y)

        # The matrices J are the weights connecting the right index to the
        # left index, e.g. J_GI connects the input to the generator network
        self.J_GG = random_matrix(
                dimensions["G"],
                dimensions["G"],
                a=0,
                b=np.sqrt(1 / (densities["GG"]*dimensions["G"])),
                dist="normal",
                density=densities["GG"],
                name="J_GG",
        )
        self.J_GF = random_matrix(
                dimensions["G"],
                dimensions["F"],
                a=0,
                b=np.sqrt(1 / (densities["GF"]*dimensions["F"])),
                dist="normal",
                density=densities["GF"],
                name="J_GF",
        )
        self.J_FG = random_matrix(
                dimensions["F"],
                dimensions["G"],
                a=0,
                b=np.sqrt(1 / (densities["FG"]*dimensions["G"])),
                dist="normal",
                density=densities["FG"],
                name="J_FG",
        )
        self.J_FF = random_matrix(
                dimensions["F"],
                dimensions["F"],
                a=0,
                b=np.sqrt(1 / (densities["FF"]*dimensions["F"])),
                dist="normal",
                density=densities["FF"],
                name="J_FF",
        )

        self.J_Gz = random_matrix(
                dimensions["G"],
                dimensions["z"],
                a=-1,
                b=1,
                dist="uniform",
                name="J_Gz",
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
                density=densities["zG"],
                name="J_zG",
        )

        self.J_FI = input_matrix(dimensions["F"], dimensions["I"], name="J_FI")
        self.J_GI = input_matrix(dimensions["G"], dimensions["I"], name="J_GI")

        # Running estimate of the inverse of the correlation matrix of the network
        # rates r (plus regularization term), necessary for learning update
        self.P = theano.shared(
            np.eye(dimensions["G"], dtype=theano.config.floatX) / alpha,
            borrow=True,
            name="P"
        )

        # Network input and output
        input = T.matrix("input", dtype=theano.config.floatX)
        z = T.dot(self.J_zG, r)

        # Teacher output
        f = T.matrix("f", dtype=theano.config.floatX)

        # Network error
        errors = z - f

        #---------- UPDATES ----------#

        # Update network state
        dx = (self.dt / self.tau) * (
            -self.x +
            self.weight_scales["GG"] * T.dot(self.J_GG, r) +
            self.weight_scales["Gz"] * T.dot(self.J_Gz, z) +
            self.weight_scales["GF"] * T.dot(self.J_GF, s) +
            T.dot(self.J_GI, input)
        )

        dy = (self.dt / self.tau) * (
            -self.y +
            self.weight_scales["FF"] * T.dot(self.J_FF, s) +
            self.weight_scales["FG"] * T.dot(self.J_FG, r) +
            T.dot(self.J_FI, input)
        )

        state_updates = [(self.x, self.x + dx), (self.y, self.y + dy)]

        #TODO: Theano shared objects are not broadcastable by default, and setting
        # the relevant axes broadcastable doesnt seem to work. To circumvent this, I
        # sum theses axes (luckyly, they all have size 1, so the sum doesnt change anything).
        # This guarantees that the axes dimensions will always be 1, making them broadcastable.
        # A less 'hacky' solution is desirable

        # Update prediciton of correlation
        dP = T.dot(T.dot(self.P, r), T.dot(r.T, self.P)) * T.inv(1 + T.dot(T.dot(r.T, self.P), r).sum())
        new_P = self.P - dP

        correlation_updates = [(self.P, new_P)]

        # Update weights
        weight_change = ((T.dot(new_P, r)).repeat(self.dimensions["z"], 1)).T \
                        * errors.sum(axis=1, keepdims=True)
        weight_updates = [(self.J_zG, self.J_zG - weight_change)]

        #---------- FUNCTIONS ----------#

        self.output = theano.function(
            inputs = [],
            outputs = T.dot(self.J_zG, r),
            name = 'output'
        )

        self.update_state = theano.function(
            inputs = [input],
            updates = state_updates,
            name = 'update_state'
        )

        self.update_weights = theano.function(
            inputs = [f],
            outputs = T.sqrt(T.sum(T.sqr(weight_change))),
            updates = correlation_updates + weight_updates,
            name = 'update_weights'
        )

    def learning_step(self, input, f):
        self.update_state(input)
        return self.update_weights(f)

    def train(self, inputs, teacher_outputs, show_progress=True, const_input=False):
        weight_changes = []
        output = []

        if const_input:
            inputs = [inputs] * len(teacher_outputs)

        # Make inputs and outputs numpy arrays in case they are not already
        inputs = [np.array(val, dtype=theano.config.floatX) for val in inputs]
        teacher_outputs = [np.array(val, dtype=theano.config.floatX) for val in teacher_outputs]

        # Convert to column vectors (nx1 matrices)
        if inputs[0].size == 1:
            inputs = [val.reshape(1, 1) for val in inputs]
        else:
            inputs = [val.reshape(val.shape[0], 1) for val in inputs]
        if teacher_outputs[0].size == 1:
            teacher_outputs = [val.reshape(1, 1) for val in teacher_outputs]
        else:
            teacher_outputs = [val.reshape(val.shape[0], 1) for val in teacher_outputs]

        if show_progress:
            bar = ProgressBar(max_value=len(teacher_outputs))
            for input, f in bar(zip(inputs, teacher_outputs)):
                dw = self.learning_step(input, f)
                weight_changes.append(dw)
                output.append(self.output())
        else:
            for input, f in zip(inputs, teacher_outputs):
                dw = self.learning_step(input, f)
                weight_changes.append(dw)
                output.append(self.output())

        return output, weight_changes

    def activity(self, inputs, repeat=1, show_progress=True):
        weight_changes = []
        output = []

        if np.isscalar(inputs):
            # Set all input activities to 'inputs' if 'inputs' is a scalar value
            inputs = [np.ones((self.dimensions["I"], 1), dtype=theano.config.floatX) * inputs]
        else:
            # Make 'inputs' a numpy array in case it is not already
            inputs = [np.array(val, dtype=theano.config.floatX) for val in inputs]

            # Convert to column vectors (nx1 matrix)
            if inputs[0].size == 1:
                inputs = [val.reshape(1, 1) for val in inputs]
            else:
                inputs = [val.reshape(val.shape[0], 1) for val in inputs]

        if show_progress:
            bar = ProgressBar(max_value=len(inputs)*repeat)
            for input in bar(inputs*repeat):
                weight_changes.append(0)
                self.update_state(input)
                output.append(self.output())
        else:
            for input in inputs*repeat:
                weight_changes.append(0)
                self.update_state(input)
                output.append(self.output())

        return output, weight_changes

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.x.get_value(borrow=True), file, -1)
            pickle.dump(self.y.get_value(borrow=True), file, -1)
            pickle.dump(self.J_FF.get_value(borrow=True), file, -1)
            pickle.dump(self.J_FG.get_value(borrow=True), file, -1)
            pickle.dump(self.J_FI.get_value(borrow=True), file, -1)
            pickle.dump(self.J_Gz.get_value(borrow=True), file, -1)
            pickle.dump(self.J_GF.get_value(borrow=True), file, -1)
            pickle.dump(self.J_GG.get_value(borrow=True), file, -1)
            pickle.dump(self.J_GI.get_value(borrow=True), file, -1)
            pickle.dump(self.J_zG.get_value(borrow=True), file, -1)
            pickle.dump(self.P.get_value(borrow=True), file, -1)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.x.set_value(pickle.load(file), borrow=True)
            self.y.set_value(pickle.load(file), borrow=True)
            self.r = transfer_func(self.x)
            self.s = transfer_func(self.y)
            self.J_FF.set_value(pickle.load(file), borrow=True)
            self.J_FG.set_value(pickle.load(file), borrow=True)
            self.J_FI.set_value(pickle.load(file), borrow=True)
            self.J_Gz.set_value(pickle.load(file), borrow=True)
            self.J_GF.set_value(pickle.load(file), borrow=True)
            self.J_GG.set_value(pickle.load(file), borrow=True)
            self.J_GI.set_value(pickle.load(file), borrow=True)
            self.J_zG.set_value(pickle.load(file), borrow=True)
            self.P.set_value(pickle.load(file), borrow=True)


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


def main():
    ### NETWORK SETUP ###
    dimensions = {"I": 1, "G": 1000, "z": 1, "F": 1}
    weight_scales = {"GG": 1.5, "zG": 1.0, "Gz": 1.0, "GF": 0, "FF": 0, "FG": 0}
    densities = {"GG": 0.1, "zG": 1.0, "GF": 1.0, "FF": 1.0, "FG": 1.0}
    dt = 10e-3 # milliseconds
    tau = 10e-3 # milliseconds
    alpha = 1.0
    network = ForceNetwork(dimensions, weight_scales, densities, dt, tau, alpha)

    ### IDLE NETWORK ACTIVITY BEFORE TRAINING ###
    output = []
    sample_activities = [[] for _ in range(10)]
    weight_changes = []
    n_samples = 1000

    print("\n--- Idle network activity before training ---")
    o, w = network.activity(0, repeat=n_samples)
    weight_changes += w
    output += o

    #  inputs = [
        #  np.array([1, 0, 0], dtype=theano.config.floatX),
        #  np.array([0, 1, 0], dtype=theano.config.floatX),
        #  np.array([0, 0, 1], dtype=theano.config.floatX),
    #  ]

    ### TRAINING ###
    period = 0.1
    #  functions = [np.sin, lambda x: np.cos(x) * np.cos(x), lambda x: np.sin(x) + 3 * np.cos(np.sin(x))*np.tanh(x)]
    functions = [np.sin, lambda x: np.cos(x) * np.cos(x), lambda x: np.cos(x) + np.sin(x)]
    #  inputs = [np.array([val], dtype=theano.config.floatX) for val in np.random.uniform(-1, 1, len(functions))]
    inputs = np.random.uniform(-1, 1, len(functions))

    print("\n--- Training ---")
    for k, function in enumerate(functions):
        #  x = np.linspace(0, 1, 1000)
        #  #  y = function((2 * np.pi / period) * x)
        #  #  x = [np.array([[val]]) for val in x]
        #  #  y = [np.array([[val]]) for val in y]
        #  y1 = np.sin((2 * np.pi / period) * x)
        #  y2 = np.cos((2 * np.pi / period) * x)
        #  x = [np.array([[val]], dtype=theano.config.floatX) for val in x]
        #  y = [np.array([[val1], [val2]], dtype=theano.config.floatX) for (val1, val2) in zip(y1, y2)]

        x = np.linspace(0, 1, n_samples)
        y = function((2 * np.pi / period) * x)

        o, w = network.train(inputs[k], y, const_input=True)
        weight_changes += w
        output += o

        o, w = network.activity(0, repeat=100)
        weight_changes += w
        output += o
            #  for i in range(10):
                #  sample_activities[i].append(network.x[i,0])

    #  network.save('testsave')
    #  network.load('testsave')


    ### IDLE NETWORK ACTIVITY AFTER TRAINING ###
    print("\n--- Idle network activity after training ---")
    o, w = network.activity(0, repeat=n_samples)
    weight_changes += w
    output += o

    for j in range(len(functions)):
        print("\n--- Testing ---")
        o, w = network.activity(inputs[j], repeat=n_samples)
        weight_changes += w
        output += o

        print("\n--- Idle network ---")
        o, w = network.activity(0, repeat=n_samples)
        weight_changes += w
        output += o

    ### PLOTTING ###
    fig, axes = plt.subplots(6)
    axes[0].plot([val[0,0] for val in output], c='r')
    #  axes[0].plot([val[1,0] for val in output], c='g')
    #  for i in range(4):
        #  axes[i+1].plot(sample_activities[i], c='b')
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


if __name__ == "__main__":
    main()
