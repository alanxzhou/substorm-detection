import numpy as np
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
import random
import matplotlib.pyplot as plt
from pymap3d.vincenty import vdist


def split_data(list_of_data, split, random=True, batch_size=None):
    """this function splits a list of equal length (first dimension) data arrays into two lists. The length of the data
    put into the second list is determined by the 'split' argument. This can be used for slitting [X, y] into
    [X_train, y_train] and [X_val, y_val]
    """

    split_idx = int((1 - split) * list_of_data[0].shape[0])

    idx = np.arange(list_of_data[0].shape[0])
    if random:
        np.random.shuffle(idx)

    split_a = []
    split_b = []

    for data in list_of_data:
        split_a.append(data[idx[:split_idx]])
        split_b.append(data[idx[split_idx:]])

    if batch_size:
        for i in range(len(split_a)):
            if i == 0:
                n_samples_a = np.shape(split_a[i])[0]
                n_samples_b = np.shape(split_b[i])[0]

                remainder_a = n_samples_a % batch_size
                remainder_b = n_samples_b % batch_size

            split_a[i] = split_a[i][:-remainder_a]
            split_b[i] = split_b[i][:-remainder_b]

    return split_a, split_b


def true_positive(y_true, y_pred):
    y_pred_pos = K.round(y_pred[:, 0])
    y_pos = K.round(y_true[:, 0])
    return K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + K.epsilon())


def false_positive(y_true, y_pred):
    y_pred_pos = K.round(y_pred[:, 0])
    y_pos = K.round(y_true[:, 0])
    y_neg = 1 - y_pos
    return K.sum(y_pred_pos * y_neg) / (K.sum(y_neg) + K.epsilon())


def distance_matrix(station_locations):
    # figure out good ordering for the stations (rows)
    a = station_locations[:, None, :] * np.ones((1, station_locations.shape[0], station_locations.shape[1]))
    locs = np.reshape(np.concatenate((a, np.transpose(a, [1, 0, 2])), axis=2), (-1, 4)).astype(float)

    d, a1, a2 = vdist(locs[:, 1], locs[:, 0], locs[:, 3], locs[:, 2])
    dists = d.reshape((station_locations.shape[0], station_locations.shape[0]))
    dists[np.isnan(dists)] = 0

    return dists


def rnn_format_x(list_of_x):
    """
    reformats feature arrays to match input dimensions for RNNs
    :param list_of_x: list of feature arrays (e.g., X_train, X_val, etc.)
    :return: list of reshaped feature arrays
    """
    x_rnn = []
    for i in range(len(list_of_x)):
        a0, a1, a2, a3 = np.shape(list_of_x[i])
        x_rnn.append(np.reshape(list_of_x[i], (a0, a2, a1*a3)))
    return x_rnn


def rnn_format_y(list_of_y):
    """
    reformats labels into onehot encoding for rnn inputs
    :param list_of_y: list of label arrays (e.g., y_train, y_val, etc.)
    :return: y_rnn: list of label arrays in onehot encoding
    """
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    y_rnn = []
    for i in range(len(list_of_y)):
        integer_encoded = label_encoder.fit_transform(list_of_y[i])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_rnn.append(onehot_encoder.fit_transform(integer_encoded))
    return y_rnn


def linear_format_x(list_of_x):
    """
    reformats list of feature arrays for linear classification
    :param list_of_x: list of feature arrays (e.g., X_train, X_val, etc.)
    :return: x_linear: list of feature arrays
    """
    x_linear = []
    for i in range(len(list_of_x)):
        a0, a1, a2 = np.shape(list_of_x[i])
        x_linear.append(np.reshape(list_of_x[i], (a0, a1 * a2)))
        #a0, a1, a2, a3 = np.shape(list_of_x[i])
        #x_linear.append(np.reshape(list_of_x[i], (a0, a1*a2*a3)))
    return x_linear


def linear_format_y(list_of_y):
    """
    reformats list of label arrays for linear classification
    :param list_of_y: list of feature arrays (e.g., y_train, y_val, etc.)
    :return: x_linear: list of feature arrays
    """
    y_linear = []
    for i in range(len(list_of_y)):
        y_linear.append(np.ravel(list_of_y[i]))
    return y_linear



"""
I found this code online, modified it only slightly to take in a pre-computed distance matrix
https://github.com/chncyhn/simulated-annealing-tsp

This will approximate solution to travelling salesman problem using simulated annealing. This can be used to find a
good ordering for the stations.
"""

def plotTSP(paths, points, num_iters=1):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    """

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x))/float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                    head_width = a_scale, color = 'r',
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
            color ='g', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'g', length_includes_head = True)

    #Set axis too slitghtly larger than the set of x and y
    plt.xlim(min(x)*1.1, max(x)*1.1)
    plt.ylim(min(y)*1.1, max(y)*1.1)
    plt.show()


class SimAnneal(object):
    def __init__(self, coords, dists,
                 T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.dists = dists
        self.N = dists.shape[0]
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour).
        """
        cur_node = random.choice(self.nodes)  # start from a random node
        solution = [cur_node]

        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def dist(self, node_0, node_1):
        """
        Euclidean distance between two nodes.
        """
        return self.dists[node_0, node_1]

    def fitness(self, solution):
        """
        Total distance of the current solution path.
        """
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.initial_solution()

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print(f"Iteration {i}/{times} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()

    def visualize_routes(self):
        """
        Visualize the TSP route with matplotlib.
        """
        plotTSP([self.best_solution], self.coords)
