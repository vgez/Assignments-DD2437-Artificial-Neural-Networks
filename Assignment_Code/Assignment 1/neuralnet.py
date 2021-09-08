import numpy as np
import activations

class NeuralLayer:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.delta_arr = np.zeros(self.n_nodes)
        self.y_arr = None
        self.inputs = None

    def get_nodes(self):
        return self.n_nodes

    def set_y(self, y):
        self.y_arr = y

    def get_y(self):
        return self.y_arr

    def set_d(self, delta):
        self.delta_arr = delta

    def get_d(self):
        return self.delta_arr

    def set_inputs(self, inputs):
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

    def __str__(self):
        return str(self.n_nodes)



class NeuralNet:
    def __init__(self):
        self.layers = []
        self.weight_matrices = []
        self.labels = []
        self.n_layers = 0

    def set_inputs(self, index, inputs):
        self.layers[index].set_inputs(inputs)

    def set_y(self, index, inputs):
        self.layers[index].set_y(inputs)

    def set_weights(self, weights):
        if type(weights) == list:
            self.weights = np.array(weights)
        else:
            self.weights = weights

    def get_weights(self):
        return self.weight_matrices

    def generate_weights(self):
        for l in range(len(self.layers)-1):
            weight_matrix = np.random.normal(loc=0, scale=0.2, size=(
                self.layers[l+1].get_nodes(),
                self.layers[l].get_nodes()+1
                ))
            self.weight_matrices.append(weight_matrix)
            print("Initial weights:\n", self.weight_matrices)
        
    def add_layer(self, nodes):
        self.layers.append(NeuralLayer(nodes))
        self.n_layers += 1

    def add_ones(self, input_v):
        out = np.append(input_v, [[1 for i in range(len(input_v[0]))]], axis=0)
        return out

    def get_outputs(self):
        return self.layers[self.n_layers-1].get_y()

    def get_error(self):
        return self.layers[self.n_layers-1].get_d()

    def forward_pass(self):
        sigmoid_v = np.vectorize(activations.sigmoid)
        for l in range(len(self.layers)-1):
            c_layer = self.layers[l]
            n_layer = self.layers[l+1]
            inputs = c_layer.get_inputs()
            y_in = np.matmul(self.weight_matrices[l], self.add_ones(inputs))
            n_layer.set_inputs(y_in)
            n_layer.set_y(activations.sigmoid(y_in))
#        print("Forward pass succesful!\n")

    def backward_pass(self, labels):
        # Rightmost layer 
        rightmost_layer = self.layers[self.n_layers-1]
        y_k = rightmost_layer.get_y()
        y_in = rightmost_layer.get_inputs()
        y_in = self.add_ones(y_in)
        e = (labels-y_k)
        sigmoid_d_v = np.vectorize(activations.sigmoid_d)
#         delta = np.multiply(e,sigmoid_d_v(y_in[0]))
        delta = np.multiply(e,activations.sigmoid_d(y_in[0]))
        rightmost_layer.set_d(delta)

        # The other layers
        for l in range(self.n_layers-2):
            index = self.n_layers-2-l
            l_c = self.layers[index]
            l_n = self.layers[index+1]
            y_in = l_c.get_y()
            y_in = self.add_ones(y_in)
            w_c = self.weight_matrices[index]
            if len(w_c.shape) == 1:
                w_c = np.array([w_c])
            delta_p = l_n.get_d()
            if l != 0:
                delta_p = np.delete(delta_p,len(delta_p)-1, axis=0)
            delta_j = np.matmul(w_c.T,delta_p)*sigmoid_d_v(y_in)
            l_c.set_d(delta_j)

    def update(self):
        eta = 0.005
        for w_i in range(len(self.weight_matrices)):
            l_c = self.layers[len(self.weight_matrices)-w_i]
            l_p = self.layers[len(self.weight_matrices)-w_i-1]
            delta = l_c.get_d()
            delta = np.array([np.mean(delta, axis=1)])
            if w_i != 0:
                delta = np.delete(delta, len(delta)-1, axis = 1)
            y_temp = np.mean(l_p.get_y(), axis=1)
            y_temp = np.append(y_temp,[1.0])
            if len(y_temp.shape) == 1:
                y_temp = [y_temp]
            y = np.transpose(y_temp)
            dw = eta*np.matmul(y,delta)
            self.weight_matrices[len(self.weight_matrices)-w_i-1] -= dw.T



    def __str__(self):
        s = ""
        for l in self.layers:
             s += str(l) + " "
        return s

