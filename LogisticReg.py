import numpy as np
import random
import copy


class LogisticReg(object):
    def __init__(self, x_train, y_train, x_test, y_test, batch_size, learning_rate, lambda_set, epoch, k_fold):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.lambda_set = lambda_set
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.k_fold = k_fold
        intercept = []
        for i in range(len(self.x_train)):
            intercept.append([1])
        self.x_train = np.append(intercept, self.x_train, axis=1)
        intercept = []
        for i in range(len(self.x_test)):
            intercept.append([1])
        self.x_test = np.append(intercept, self.x_test, axis=1)
        self.train_samples = list(zip(self.x_train, self.y_train))
        self.test_samples = list(zip(self.x_test, self.y_test))


    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def loss(self, batch, w, l):
        error = 0
        temp_w = copy.deepcopy(w)
        temp_w[0] = 0
        #         print(batch)
        for (x, y) in batch:
            z = np.dot(x, w)
            #             print(z)
            error += (y * np.log(self.sigmoid(z))) + ((1 - y) * np.log(1 - self.sigmoid(z)))
        return (-error / len(batch)) + (l * np.square(np.linalg.norm(w, 2))) / len(batch)

    def gradient(self, samples, w, l):
        x_temp = np.array(list(zip(*samples))[0])
        y_temp = np.array(list(zip(*samples))[1])
        temp_w = copy.deepcopy(w)
        temp_w[0] = 0
        gradient = 0
        # error = y_temp - self.sigmoid(np.dot(x_temp, w))
        # gradient = np.dot(x_temp, error)
        for i in range(len(y_temp)):
            z = np.dot(x_temp[i], w)
            hyp = self.sigmoid(z)
            error = y_temp[i] - hyp
            gradient += error * x_temp[i]
        return (gradient.T / len(samples)) + (l * temp_w / len(samples))

    def sgd(self, train_samples, weight, l, is_cross_val):
        i = 1
        #         print(self.calculate_training_error(weight))
        while self.calculate_training_error(weight) != 0:
            if not is_cross_val:
                if i % 1000 == 0:
                    training_error = self.calculate_training_error(weight)
                    print("Epoch : ", i, " ; ", "Training_error = ", training_error)
            random.shuffle(train_samples)
            train_copy = copy.deepcopy(train_samples)
            # weight = weight + self.learning_rate * self.gradient(train_copy, weight, l)
            while len(train_copy) != 0:
                train_batch = train_copy[:self.batch_size]
                # print(len(train_batch))
                train_copy = train_copy[self.batch_size:]
                weight_new = weight + self.learning_rate * self.gradient(train_batch, weight, l)
            i += 1
            previous_loss = self.loss(train_samples, weight, l)
            #             print(previous_loss)
            new_loss = self.loss(train_samples, weight_new, l)
            #             print(previous_loss, new_loss)
            #             print(previous_loss - new_loss)
            if np.abs(previous_loss - new_loss) < 1e-10:
                break
            weight = weight_new
        return weight

    def train(self):
        print("enter")
        min_val_error = 100000
        num_of_features = self.x_train.shape[1]
        train_samples = list(zip(self.x_train, self.y_train))
        num_in_one_fold = int(len(self.train_samples) / self.k_fold)
        lambda_reg = self.lambda_set[0]
        print("cross-validating.....")
        for l in self.lambda_set:
            print("lambda = ", l)
            total_val_error = 0
            for i in range(0, len(train_samples), num_in_one_fold):
                j = i + num_in_one_fold
                train_batch = train_samples[:i] + train_samples[j:]
                val_batch = train_samples[i:j]
                weight = np.asmatrix(0.01*np.random.randn(num_of_features)).T
                weight_learned = self.sgd(train_batch, weight, l, True)
                val_error = self.loss(val_batch, weight_learned, l)
                total_val_error += val_error
            avg_val_error = total_val_error/self.k_fold
            print("validation error: ", avg_val_error)
            if avg_val_error < min_val_error:
                min_val_error = avg_val_error
                lambda_reg = l

        print("training......")
        print("lambda = ", lambda_reg)
        weight = np.array(0.01 * np.random.randn(num_of_features)).T
        weight = self.sgd(self.train_samples, weight, lambda_reg, False)
        training_error = self.calculate_training_error(weight)
        print("Training_error = ", training_error)
        return weight, self.x_test

    def calculate_training_error(self, weight):
        total_error = len(self.train_samples)
        for (x, y) in self.train_samples:
            z = np.dot(x, weight)
            if (self.sigmoid(z) > 0.5 and y == 1) or (self.sigmoid(z) < 0.5 and y == 0):
                total_error -= 1
        training_error = total_error / len(self.train_samples)
        return training_error

    def calculate_testing_error(self):
        weight, b = self.train()
        total_error = 0
        for (x, y) in self.test_samples:
            z = np.dot(x, weight)
            if (self.sigmoid(z) > 0.5 and y == 1) or (self.sigmoid(z) < 0.5 and y == 0):
                total_error += 1
        test_error = total_error / len(self.test_samples)
        return test_error


class MultiClassLog(object):
    def __init__(self, batch_size, learning_rate, lambda_set, epoch, k_fold):
        self.batch_size = batch_size
        self.lambda_set = lambda_set
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.k_fold = k_fold

    def classification(self, x_train, x_test, y_test):
        k = 0
        param = []
        x_test_inter = []
        for i in range(10):
            y = np.zeros(500)
            for j in range(k, k + 50):
                y[j] = 1
            k += 50
            log_train = LogisticReg(x_train, y, x_test, y_test, self.batch_size,
                                    self.learning_rate, self.lambda_set, self.epoch, self.k_fold)
            w, x_test_inter = log_train.train()
            param.append(w)

        return param, x_test_inter

    def test(self, param, x_test, y_test):
        wrong_class = 0
        for i in range(len(y_test)):
            pred = []
            for j in range(10):
                temp_pred = np.dot(x_test[i], param[j])
                pred.append(temp_pred)
            # print(np.argmax(pred))
            if np.argmax(pred) != y_test[i]:
                wrong_class += 1

        print("Classification Error :", wrong_class / len(y_test))


