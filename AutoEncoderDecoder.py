import numpy as np


class AutoEncodeDecode(object):
    def __init__(self, n_features, n_hidden, n_output, actv='sigmoid', lrate=0.000001, reg=10, epoch=50000):
        self.actv = actv
        self.lrate = lrate
        self.epoch = epoch
        self.reg = reg
        self.num_inputs = n_features
        self.num_hidden = n_hidden
        self.num_out = n_output

        # Initialize weights
        self.w1 = 0.01*np.random.randn(n_features, n_hidden)
        self.delta_w1 = np.zeros((n_features, n_hidden))
        self.w2 = 0.01*np.random.randn(n_hidden, n_output)
        self.delta_w2 = np.zeros((n_hidden, n_output))

        #  Initialize biases
        self.b1 = np.zeros((1,n_hidden))
        self.delta_b1 = np.zeros((1,n_hidden))
        self.b2 = np.zeros((1, n_output))
        self.delta_b2 = np.zeros((1, n_output))

    @staticmethod
    def sigmoid(z):
        # print(z)
        return 1/(1 + np.exp(-z))

    @staticmethod
    def relu(z):
        if z < 0:
            return 0
        else:
            return z

    @staticmethod
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def sofmax(z):
        z_exp = [np.exp(i) for i in z]
        z_exp_sum = np.sum(z_exp)
        softmax_values = [(i/z_exp_sum) for i in z_exp]
        return softmax_values

    def gradient(self, z):
        if self.actv == 'sigmoid':
            vsigmoid = np.vectorize(self.sigmoid)
            return np.multiply((1 - vsigmoid(z)), vsigmoid(z))
        elif self.actv == 'relu':
            grad = []
            for i in range(len(z)):
                if z[i] > 0:
                    grad.append(1)
                else:
                    grad.append(0)
            return np.array(grad)
        elif self.actv == 'tanh':
            vtanh = np.vectorize(self.tanh)
            return 1 - np.square(vtanh(z))
        else:
            raise ValueError("activation function not supported")

    @staticmethod
    def no_act_outputs(activations, w, b):
        return np.matmul(activations, w) + b

    def activations(self, no_act_outputs):
        if self.actv == 'sigmoid':
            vsigmoid = np.vectorize(self.sigmoid)
            return vsigmoid(no_act_outputs)
        elif self.actv == 'relu':
            vrelu = np.vectorize(self.relu)
            return vrelu(no_act_outputs)
        elif self.actv == 'tanh':
            vtanh = np.vectorize(self.tanh)
            return vtanh(no_act_outputs)
        else:
            raise ValueError("activation function not supported")

    def softmax_activations(self, no_acts_outputs):
        activations = []
        for i in range(len(no_acts_outputs)):
            softmax_act = self.sofmax(no_acts_outputs[i])
            activations.append(np.array(softmax_act))
        return np.array(activations)

    def forward_prop(self, x):
        output1 = self.no_act_outputs(x, self.w1, self.b1)
        activation1 = self.activations(output1)
        output2 = self.no_act_outputs(activation1, self.w2, self.b2)
        # activation2 = self.softmax_activations(output2)
        activation2 = output2

        return output1, activation1, output2, activation2

    def backprop(self, x, y, output1, activation1, output2, activation2):
        # delta_out = np.multiply((activation2 - y), self.gradient())
        delta_out = (activation2 - y)
        delta_hid = np.multiply(np.matmul(self.w2, delta_out), self.gradient(output1))
        x = np.atleast_2d(x)
        delta_hid = np.atleast_2d(delta_hid)
        delta_out = np.atleast_2d(delta_out)
        w1_gradient = np.dot(x.T, delta_hid)
        activation1 = np.atleast_2d(activation1)
        w2_gradient = np.dot(activation1.T, delta_out)
        b1_gradient = delta_hid
        b2_gradient = delta_out

        return w1_gradient, w2_gradient, b1_gradient, b2_gradient

    def train(self, x, y, x_test, y_test):
        n = len(x)
        for i in range(self.epoch):
            output1, activation1, output2, activation2 = self.forward_prop(x)
            for j in range(len(x)):

                w1_gradient, w2_gradient, b1_gradient, b2_gradient = \
                    self.backprop(x[j], y[j], output1[j], activation1[j], output2[j], activation2[j])
                # print()
                self.delta_w1 += w1_gradient
                self.delta_w2 += w2_gradient
                self.delta_b1 += np.array(b1_gradient)
                self.delta_b2 += b2_gradient
            self.w1 -= self.lrate * (self.delta_w1 / n + self.reg * self.w1)
            self.w2 -= self.lrate * (self.delta_w2 / n + self.reg * self.w2)
            self.b1 -= self.lrate * (self.delta_b1 / n)
            self.b2 -= self.lrate * (self.delta_b2 / n)
            if i % 100 == 0:
                # error1 = self.test(x, y)
                error = self.test(x_test, y_test)
                print(error)
            if error < 50 :
                break
                # print(error1)

        return self.w1, self.b1, self.w2, self.b2

    def test(self, x, y):
        output1, activation1, output2, activation2 = self.forward_prop(x)
        error = np.square(np.linalg.norm((activation2 - y), ord=2))/len(y)

        return error


