import numpy as np
import PCA
import scipy.io as io
import matplotlib.pyplot as plt
import cv2

data = io.loadmat("ExtYaleB10.mat")

train = data['train']
test = data['test']
train = np.ndarray.tolist(train)[0]
test = np.ndarray.tolist(test)[0]
train = np.array(train)
test = np.array(test)

x_train = []
for i in range(len(train)):
    train_temp = train[i].T
    for j in range(len(train_temp)):
        x_train.append(train_temp[j].T)
x_train = np.array(x_train)

x_test = []
k = 0
y_test_log_svm = np.zeros((140,1))
for i in range(len(test)):
    test_temp = test[i].T
    for j in range(len(test_temp)):
        y_test_log_svm[k][0] = i
        x_test.append(test_temp[j].T)
        k += 1
x_test = np.array(x_test)

x_combined = []
k = 0
h = 0
while k + h < 640:
    for j in range(k, k+50):
        x_combined.append(x_train[j])
    for f in range(h, h+14):
        x_combined.append((x_test[f]))
    k += 50
    h += 14
x_combined = np.array(x_combined)
# plt.imshow(x_combined[120])
# plt.show()
# print(x_combined.shape)
x_combined_vec = []
for i in range(len(x_combined)):
    x_temp = cv2.resize(x_combined[i], (20, 17), interpolation=cv2.INTER_AREA)
    x_combined_vec.append(x_temp.flatten())

x_combined_vec = np.array(x_combined_vec)
print(x_combined_vec.shape)
print("apply PCA")
test = PCA.PCA(d=2)
mean, basis, new_x_data = test.pca(x_combined_vec.T)
# B=[1.29,1.27,1.46,0.91,0.56,0.99,1.00,0.37,1.24,1.23]
print(new_x_data.shape)
new_x_data = new_x_data.T
print("done")
fig, ax = plt.subplots()
plt.title("Data in 2-dim after applying PCA on original dataset")
ax.scatter(new_x_data[0:64, 0], new_x_data[0:64, 1], c='red', marker='o', label='class 1')
ax.scatter(new_x_data[64:128, 0], new_x_data[64:128, 1], c='blue', marker='o', label='class 2')
ax.scatter(new_x_data[128:192, 0], new_x_data[128:192, 1], c='green', marker='o', label='class 3')
ax.scatter(new_x_data[192:256, 0], new_x_data[192:256, 1], c='brown', marker='o', label='class 4')
ax.scatter(new_x_data[256:320, 0], new_x_data[256:320, 1], c='yellow', marker='o', label='class 5')
ax.scatter(new_x_data[320:384, 0], new_x_data[320:384, 1], c='orange', marker='o', label='class 6')
ax.scatter(new_x_data[384:448, 0], new_x_data[384:448, 1], c='gray', marker='o', label='class 7')
ax.scatter(new_x_data[448:512, 0], new_x_data[448:512, 1], c='pink', marker='o', label='class 8')
ax.scatter(new_x_data[512:576, 0], new_x_data[512:576, 1], c='purple', marker='o', label='class 9')
ax.scatter(new_x_data[576:640, 0], new_x_data[576:640, 1], c='magenta', marker='o', label='class 10')


ax.legend()
plt.show()

