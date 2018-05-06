import numpy as np
import random


class SVM(object):
    def __init__(self, x_train, y_train, x_test, y_test, max_pass, reg, tol):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.max_pass = max_pass
        self.reg = reg
        self.tol = tol
        self.train_samples = list(zip(self.x_train, self.y_train))
        self.test_samples = list(zip(self.x_test, self.y_test))

    def calculate_error(self, x, y, alphas, b):
        f = 0
        for (x_temp, y_temp), alpha in zip(self.train_samples, alphas):
            f += alpha * y_temp * np.dot(x_temp, x)
        f += b
        return f - y

    def lower_bound(self, i, j, alpha_i, alpha_j):
        if self.y_train[i] == self.y_train[j]:
            return max(0, alpha_i + alpha_j - self.reg)
        else:
            return max(0, alpha_j - alpha_i)

    def higher_bound(self, i, j, alpha_i, alpha_j):
        if self.y_train[i] == self.y_train[j]:
            return min(self.reg, alpha_j + alpha_i)
        else:
            return min(self.reg, self.reg + alpha_j - alpha_i)

    def eta(self, i, j):
        return 2 * np.dot(self.x_train[i], self.x_train[j]) - \
                np.dot(self.x_train[i], self.x_train[i]) - \
                np.dot(self.x_train[j], self.x_train[j])

    def calculate_clip_alpha_j(self, j, alpha_j, Ei, Ej, H, L, eta):
        # print("dfnsjfn")
        # print(self.y_train[j])
        # print(Ej, Ei)
        # print(eta)
        # print(H)
        # print(L)
        # print("dnfjnes")
        alpha_j = alpha_j - ((self.y_train[j] * (Ei - Ej)) / eta)
        # print(alpha_j)
        if alpha_j > H:
            return H
        elif alpha_j < L:
            return L
        else:
            return alpha_j

    def calculate_alpha_i(self, alpha_j_old, alpha_j, alpha_i, i, j):
        alpha_i = alpha_i + self.y_train[i] * self.y_train[j] * (alpha_j_old - alpha_j)
        return alpha_i

    def calculate_b1(self, b, i, j, Ei, alpha_i_old, alpha_j_old, alphas):
        return b - Ei - \
                    self.y_train[i] * (alphas[i] - alpha_i_old) * np.dot(self.x_train[i], self.x_train[i]) - \
                    self.y_train[j] * (alphas[j] - alpha_j_old) * np.dot(self.x_train[i], self.x_train[j])

    def calculate_b2(self, b, i, j, Ej, alpha_i_old, alpha_j_old, alphas):
        return b - Ej - \
                    self.y_train[i] * (alphas[i] - alpha_i_old) * np.dot(self.x_train[i], self.x_train[j]) - \
                    self.y_train[j] * (alphas[j] - alpha_j_old) * np.dot(self.x_train[j], self.x_train[j])

    def calculate_b(self, b1, b2, i, j, alphas):
        if 0 < alphas[i] < self.reg:
            return b1
        elif 0 < alphas[j] < self.reg:
            return b2
        else:
            return (b1 + b2)/2
    @staticmethod
    def calculate_hypothesis(x, weight, b):
        return np.dot(x, weight) + b

    def svm(self):
        alphas = np.zeros(len(self.y_train))
        b = 0
        passes = 0
        while passes < self.max_pass:
            print("pass : ", passes)
            num_changed_alphas = 0
            for i in range(len(self.y_train)):
                Ei = self.calculate_error(self.x_train[i], self.y_train[i], alphas, b)
                # print(Ei)
                if ((self.y_train[i] * Ei < - self.tol) and (alphas[i] < self.reg)) or \
                    ((self.y_train[i] * Ei > self.tol) and (alphas[i] > 0)):

                    j = i
                    while j == i:
                        j = random.randint(0, len(self.y_train) - 1)
                    # print(j)
                    Ej = self.calculate_error(self.x_train[j], self.y_train[j], alphas, b)
                    # print(Ej)
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    lower_bound = self.lower_bound(i, j, alpha_i_old, alpha_j_old)
                    higher_bound = self.higher_bound(i, j, alpha_i_old, alpha_j_old)
                    if lower_bound == higher_bound:
                        continue
                    eta = self.eta(i, j)
                    if eta >= 0:
                        continue
                    alphas[j] = self.calculate_clip_alpha_j(j, alpha_j_old, Ei, Ej, higher_bound, lower_bound, eta)
                    # print(alphas[j])
                    if np.abs(alpha_j_old - alphas[j]) < 0.00001:
                        continue
                    alphas[i] = self.calculate_alpha_i(alpha_j_old, alphas[j], alpha_i_old, i, j)
                    # print(alphas[i])
                    b1 = self.calculate_b1(b, i, j, Ei, alpha_i_old, alpha_j_old, alphas)
                    b2 = self.calculate_b2(b, i, j, Ej, alpha_i_old, alpha_j_old, alphas)
                    b = self.calculate_b(b1, b2, i, j, alphas)
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        weight = 0
        # print(alphas)
        for i in range(len(self.y_train)):
            weight += alphas[i] * self.y_train[i] * self.x_train[i]

        return weight, b

    def calculate_training_error(self, weight, b):
        # weight, b = self.svm()
        total_error = 0
        for (x, y) in self.train_samples:
            if self.calculate_hypothesis(x, weight, b) > 0 and y != 1 or \
                                    self.calculate_hypothesis(x, weight, b) < 0 and y != -1:
                total_error += 1
        train_error = total_error/len(self.train_samples)
        print(train_error)
        return train_error

    def calculate_testing_error(self, weight, b):
        # weight, b = self.svm()
        total_error = 0
        for (x, y) in self.test_samples:
            if self.calculate_hypothesis(x, weight, b) > 0 and y != 1 or \
                                    self.calculate_hypothesis(x, weight, b) < 0 and y != -1:
                total_error += 1
        test_error = total_error/len(self.test_samples)
        return test_error


class MultiClassSVM(object):
    def __init__(self, max_pass, reg, tol):
        self.ma_pass = max_pass
        self.reg = reg
        self.tol = tol

    def classification(self, x_train, x_test, y_test):
        k = 0
        param = []
        y = np.ones(500)
        #         print(y)
        for i in range(10):
            for f in range(len(y)):
                y[f] = -1
            print("classifier :", i)
            for j in range(k, k + 50):
                y[j] = 1
            k += 50
            svm_train = SVM(x_train, y, x_test, y_test, self.ma_pass, self.reg, self.tol)
            w, b = svm_train.svm()
            training_error = svm_train.calculate_training_error(w, b)
            print(training_error)
            param.append((w, b))
        return param

    def test(self, param, x_test, y_test):
        wrong_class = 0
        for i in range(len(y_test)):
            pred = []
            for j in range(10):
                #                 print(param[j][0].shape)
                #                 print(param[j][1])
                temp_pred = np.dot(x_test[i], param[j][0]) + param[j][1]
                pred.append(temp_pred)
            print(np.argmax(pred))
            if np.argmax(pred) != y_test[i]:
                wrong_class += 1

        print("Classification Error :", wrong_class / len(y_test))


