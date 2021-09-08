import numpy as np


class RadialBasisFunctionNetwork:
    def __init__(self, input_nodes_n, rbf_nodes_n, output_nodes_n):
        self.debug = False
        self.input_nodes_n = input_nodes_n
        self.rbf_nodes_n = rbf_nodes_n
        self.rbf_nodes_centers = None
        self.output_nodes = output_nodes_n
        self.weights = None
        self.phi = None
        self.total_error_list = []
        self.sigma = 0.2

        self._manual_sin2x_rbf_placement()
        self._randomize_weights()

    def _print_debug(self, string_out, other1="", other2="", other3=""):
        if self.debug:
            print(string_out, other1, other2, other3)

    def _gaussian_transfer_function(self, x, mu):
        # Mu is the position of the center of gauss bell function, sigma squared is its variance.
        return np.exp(-((x - mu) ** 2) / (2 * self.sigma ** 2))

    def _randomize_weights(self):
        # Returns a numpy array of weights drawn randomly from a normal distribution
        centre = 0
        standard_deviation = 1
        seed = 132
        np.random.seed(seed)
        self.weights = np.random.normal(centre, standard_deviation, self.rbf_nodes_n)

    def _manual_sin2x_rbf_placement(self):
        self.rbf_nodes_centers = []
        for x in np.arange(0.1, 2*np.pi, 2*np.pi/self.rbf_nodes_n):
            # 1 dimension input
            self.rbf_nodes_centers.append(x)
        self.rbf_nodes_centers = np.array(self.rbf_nodes_centers)

    def _calculate_phi(self, data):
        # Calculate phi for multiple data points
        self.phi = []
        for i, sample in enumerate(data):
            row_list = []
            for k, rbf_center in enumerate(self.rbf_nodes_centers):
                row_list.append(self._gaussian_transfer_function(sample, rbf_center))
            self.phi.append(np.array(row_list))
        self.phi = np.array(self.phi)

    def _calculate_phi_point(self, data_point):
        # Calculate phi for a single data point
        self.phi = []
        for k, rbf_center in enumerate(self.rbf_nodes_centers):
            self.phi.append(self._gaussian_transfer_function(data_point, rbf_center))
        self.phi = np.array(self.phi)

    def _least_square(self, data, targets):
        # Batch learning method
        self._calculate_phi(data)

        self.phi_weight_product = np.matmul(self.phi, self.weights)
        self.total_error = np.square(np.linalg.norm(self.phi_weight_product-targets))

        coefficient_matrix = np.matmul(self.phi.T, self.phi)
        b = np.matmul(self.phi.T, targets)

        new_weights = np.linalg.solve(coefficient_matrix, b)
        self.phi_weight_product = np.matmul(self.phi, new_weights)
        self.total_error = np.square(np.linalg.norm(self.phi_weight_product - targets))
        self.weights = new_weights

    def _delta_learning(self, data_point, target, learning_rate=0.5):
        # Sequential learning method
        self._calculate_phi_point(data_point)
        error = target - np.matmul(self.phi.T, self.weights)
        delta_weights = learning_rate * error * self.phi
        self.weights += delta_weights.T
        return error

    def fit_batch(self, data, targets):
        self._least_square(data, targets)

    def fit_sequential(self, data, targets, epochs=100, learning_rate=0.1, sigma=0.1):
        self.sigma = sigma
        for _ in range(epochs):
            epoch_error_list = []
            if data.size > 1:
                for i, data_point in enumerate(data):
                    epoch_error_list.append(self._delta_learning(data_point, targets[i], learning_rate))
                self.total_error_list.append(np.mean(np.square(epoch_error_list)))
            else:
                print("SOMETHING IS VERY WRONG IN fit() method!!!")
                self._delta_learning(data, targets)

    def predict_batch(self, data):
        self._calculate_phi(data)
        return np.matmul(self.phi, self.weights)

    def predict_sequential(self, data):
        output = []
        if data.size > 1:
            for i, data_point in enumerate(data):
                self._calculate_phi_point(data_point)
                output.append(np.matmul(self.phi, self.weights.T))
            output = np.array(output)
        return output

    def get_rbf_node_placement(self):
        return self.rbf_nodes_centers


