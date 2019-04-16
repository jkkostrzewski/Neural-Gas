import pandas as pd
from matplotlib import pyplot as plt
import random
import math
from sklearn import preprocessing


class CNode:
    def __init__(self, nr_of_weights):
        self.weights = []
        self.distance_to_input = None

        # initialize random weights
        for i in range(nr_of_weights):
            self.weights.append(random.uniform(0.35, 0.75))

    def get_distance_to_input(self):
        return self.distance_to_input

    # euclidean distance method
    def get_distance(self, input_array):    # input_array ---> the point on the chart that is chosen by random
        distance = 0
        for i in range(len(self.weights)):
            distance += (input_array[i] - self.weights[i]) ** 2
        return math.sqrt(distance)

    def adjust_weight(self, input_array, learning_rate, influence):     # adjust neuron position in every epoch
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * influence * (input_array[i] - self.weights[i])


def plot(data, neurons, color, fig_arr):
    for n in fig_arr:
        n.clear()

    i = 0
    no_of_plots = len(neurons[0].weights)/2
    for fig in fig_arr:
        for j in range(2):
            if no_of_plots > 0:
                ax = fig.add_subplot(2, 1, j+1)
                temp_array_x = []
                temp_array_y = []
                for n in neurons:
                    temp_array_x.append(n.weights[i])
                    temp_array_y.append(n.weights[i+1])
                ax.plot(data[i+1], data[i+2], 'k.')
                ax.plot(temp_array_x, temp_array_y, '{}o'.format(color))
                i += 2
            no_of_plots -= 1

    plt.pause(0.001)


class NeuralGas:
    def __init__(self, data, no_of_neurons=25, iterations=1000, display_animation=True, display_current_iteration=True,
                 skip_frames_count=1):
        data = data.reset_index()

        x = data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self._data = pd.DataFrame(x_scaled)

        self._node_array = []
        self._k_max = iterations
        self._k = 0        # current_iteration
        self._influence = 0
        self._start_learning_rate = 0.8
        self._learning_rate_min = 0.003
        self._delta_max = no_of_neurons/2
        self._delta_min = 0.01
        self._error_array = []
        self._display_animation = display_animation
        self._display_current_iteration = display_current_iteration
        self._skip_frames_count = skip_frames_count
        self._fig_count = int((len(self._data.columns) - 1) / 4)
        if (len(self._data.columns) - 1) % 4 == 2:
            self._fig_count += 1
        self._fig_arr = []

        for fig in range(self._fig_count):
            self._fig_arr.append(plt.figure(figsize=(6, 10)))

        for n in range(no_of_neurons):
            self._node_array.append(CNode(len(self._data.columns) - 1))

    def sort_nodes(self, node_array, input_array):
        for node in node_array:
            node.distance_to_input = node.get_distance(input_array)

        for i in range(len(node_array)):
            for j in range(len(node_array) - 1):
                if node_array[j].distance_to_input > node_array[j+1].distance_to_input:
                    tmp = node_array[j]
                    node_array[j] = node_array[j+1]
                    node_array[j+1] = tmp

        return node_array

    def calculate_error(self):
        distance = 0
        for i in range(len(self._data)):
            point = []
            for j in range(1, len(self._data.columns)):
                point.append(self._data.iloc[i, j])
            self.sort_nodes(self._node_array, point)
            bmu = self._node_array[0]
            distance += bmu.get_distance(point)

        self._error_array.append([math.sqrt((distance**2) / len(self._data)), self._k])

    def run(self):
        while self._k <= self._k_max:
            if self._display_current_iteration:
                print(self._k)
            random_point = []
            random_int = random.randint(0, len(self._data)-1)

            for i in range(1, len(self._data.columns)):
                random_point.append(self._data.iloc[random_int, i])

            self._node_array = self.sort_nodes(self._node_array, random_point)

            for i in range(len(self._node_array)):
                delta = self._delta_max * math.pow((self._delta_min / self._delta_max), (self._k/self._k_max))
                influence = math.exp(-i/delta)
                learning_rate = self._start_learning_rate * math.pow((self._learning_rate_min /
                                                                      self._start_learning_rate), (self._k/self._k_max))
                self._node_array[i].adjust_weight(random_point, learning_rate, influence)

            self.calculate_error()

            if self._display_animation:
                if self._k % self._skip_frames_count == 0:
                    plot(self._data, self._node_array, 'b', self._fig_arr)
            self._k += 1
        if self._display_animation == False:
            plot(self._data, self._node_array, 'b', self._fig_arr)
        plt.show()
        plt.title("Quantization error chart for each epoch")
        plt.plot([row[1] for row in self._error_array], [row[0] for row in self._error_array])
        plt.show()

