import numpy as np
from matplotlib import pyplot as plt


def spec_rad(mat):
    return max(abs(np.linalg.eigvals(mat)))


def random_matrix(n_rows, n_cols, density=None, new_spec_rad=None):
    dim = (n_rows, n_cols)
    mask = np.random.uniform(0, 1, dim)
    mat = np.random.uniform(-1, 1, dim)
    if density is not None:
        mat[mask>density] = 0
    if new_spec_rad is not None:
        old_spec_rad = spec_rad(mat)
        mat *= new_spec_rad / old_spec_rad
    return mat


def transfer_func(x):
    return np.tanh(x)


def inverse_transfer_func(x):
    return np.arctanh(x)


class RNN:

    def __init__(self, input_size, hidden_size, output_size):
        self.dimensions = (input_size, hidden_size, output_size)
        self.hidden_state = np.zeros((hidden_size, 1))
        self.input_weights = random_matrix(hidden_size, input_size)
        self.hidden_weights = random_matrix(hidden_size, hidden_size, density=0.2, new_spec_rad=0.75)
        self.output_weights = random_matrix(output_size, input_size + hidden_size)
        #  self.back_weights = random_matrix(hidden_size, output_size)
        self.back_weights = np.zeros((hidden_size, output_size))

    def update(self, input, output, noise=None):
        if noise is not None:
            noise_val = np.random.normal(0, noise)
        else:
            noise_val = 0
        self.hidden_state = transfer_func(
            self.input_weights @ input +
            self.hidden_weights @ self.hidden_state +
            self.back_weights @ output +
            noise_val
        )

    def train(self, teacher_in, teacher_out, orders=[1]):
        # Setup
        self.orders = orders
        transient = 400
        assert len(teacher_in) == len(teacher_out), "Teacher input and output has to have same size"
        assert len(teacher_in) > 2*transient, "Not enough training samples"

        self.hidden_state = transfer_func(self.input_weights @ teacher_in[0] +
                            self.hidden_weights @ self.hidden_state)

        # Skip the transient
        for i in range(transient):
            self.update(teacher_in[i+1], teacher_out[i], noise=0.001)

        state_collect = np.zeros((len(teacher_in) - transient, len(orders) * (self.dimensions[0] + self.dimensions[1])))
        teacher_collect = np.zeros((len(teacher_in) - transient, self.dimensions[2]))

        # Iterate over the rest of the training set
        for i in range(transient, len(teacher_in) - 1):
            state = np.vstack([np.vstack((teacher_in[i], self.hidden_state))**exponent for exponent in orders])
            state_collect[i - transient,:] = state[:,0]
            teacher_collect[i - transient,:] = teacher_out[i][:,0]
            self.update(teacher_in[i+1], teacher_out[i], noise=0.001)

        self.output_weights = (np.linalg.pinv(state_collect) @ teacher_collect).transpose()

        state = np.vstack([np.vstack((teacher_in[-1], self.hidden_state))**exponent for exponent in orders])
        self.last_output = self.output_weights @ state

    def eval(self, input):
        for _ in range(400):
            self.update(input, self.last_output)
            state = np.vstack([np.vstack((input, self.hidden_state))**exponent for exponent in self.orders])
            tmp = self.last_output
            self.last_output = self.output_weights @ state
            print(np.linalg.norm(tmp - self.last_output))
        exit()
        return self.last_output


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
    a = np.pi / 16
    b = np.pi / 2
    network = RNN(2, 1000, 2)

    args1 = np.random.permutation(np.linspace(a, b, 1000))
    args2 = np.random.permutation(np.linspace(a, b, 1000))
    teacher_in = np.array([np.array([[x],[y]]) for (x, y) in zip(*get_pos2d(args1, args2))])
    teacher_out = np.array([np.array([[x],[y]]) for (x, y) in zip(args1, args2)])

    network.train(teacher_in, teacher_out, orders=[1,2,3,4])

    args1 = np.random.permutation(np.linspace(a, b, 680))
    args2 = np.random.permutation(np.linspace(a, b, 680))
    inputs = np.array([np.array([[x],[y]]) for (x, y) in zip(*get_pos2d(args1, args2))])
    outputs = np.array([np.array([[x],[y]]) for (x, y) in zip(args1, args2)])
    errors = []
    results = []
    for input, output in zip(inputs, outputs):
        result = network.eval(input)
        err = np.linalg.norm(output - result) / np.linalg.norm(output)
        errors.append(err)
        results.append(result)

    fig, (ax1, ax2) = plt.subplots(2)
    #  ax1.scatter(inputs, errors)
    ax1.plot(errors)
    ax1.set_xlabel('step')
    ax1.set_ylabel('relative error')
    #  ax1.set_xlim(a, b)
    #  ax2.plot(x, y, linewidth=2.0, color='r')
    #  ax2.scatter(inputs, results)
    #  ax2.set_xlim(a, b)
    ax2.scatter([mat[0,0] for mat in outputs], [mat[1,0] for mat in outputs], marker='x', color='b', label='target')
    ax2.scatter([mat[0,0] for mat in results], [mat[1,0] for mat in results], marker='o', color='r', label='acquired')
    ax2.legend()
    ax2.set_xlabel(r"$\theta_1$")
    ax2.set_ylabel(r"$\theta_2$")
    plt.show()
