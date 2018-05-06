import numpy as np
from KMeans import KMeans as KMeans
import scipy.io as io
import cv2
import PCA
import matplotlib.pyplot as plt
import SpectralClustering as sp


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
y = np.zeros(640)
a = 0
t = 0
for i in range(10):
    for j in range(t, t+64):
        y[j] = a
    t += 64
    a += 1

while k + h < 640:
    for j in range(k, k+50):
        x_combined.append(x_train[j])
    for f in range(h, h+14):
        x_combined.append((x_test[f]))
    k += 50
    h += 14
x_combined = np.array(x_combined)

x_combined_vec = []
for i in range(len(x_combined)):
    x_temp = cv2.resize(x_combined[i], (20, 17), interpolation=cv2.INTER_AREA)
    x_combined_vec.append(x_temp.flatten())

x_combined_vec = np.array(x_combined_vec)

test = PCA.PCA(d=2)
mean, basis, new_x_data = test.pca(x_combined_vec.T)
print(new_x_data.shape)
#
# # apply kmeans to original dataset
Kmeans = KMeans(L=10, K=10)
centroids, num_of_points_group, grouping, final_grouping = Kmeans.clustering(x_combined_vec.T)
print(final_grouping.shape)



test1 = PCA.PCA(d=100)
mean1, basis1, new_x_data1 = test1.pca(x_combined_vec.T)
Kmeans = KMeans(L=10, K=10)
centroids1, num_of_points_group1, grouping1, final_grouping1 = Kmeans.clustering(new_x_data1)
print(final_grouping.shape)

# # print(new_x_data1.shape)


# apply spectral clustering to original dataset

test3 = sp.SpectralClustering(K=10)
clusters, final_grouping = test3.clustering(x_combined_vec)
# Kmeans = KMeans(L=10, K=10)
# centroids, num_of_points_group, grouping, final_grouping = Kmeans.clustering(x_combined_vec.T)
# print(final_grouping.shape)
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
cluster5 = []
cluster6 = []
cluster7 = []
cluster8 = []
cluster9 = []
cluster10 = []
for i in range(len(final_grouping[:, 0])):
    if final_grouping[i, 0] == 1:
        cluster1.append(new_x_data[:, i])
    elif final_grouping[i, 1] == 1:
        cluster2.append(new_x_data[:, i])
    elif final_grouping[i, 2] == 1:
        cluster3.append(new_x_data[:, i])
    elif final_grouping[i, 3] == 1:
        cluster4.append(new_x_data[:, i])
    elif final_grouping[i, 4] == 1:
        cluster5.append(new_x_data[:, i])
    elif final_grouping[i, 5] == 1:
        cluster6.append(new_x_data[:, i])
    elif final_grouping[i, 6] == 1:
        cluster7.append(new_x_data[:, i])
    elif final_grouping[i, 7] == 1:
        cluster8.append(new_x_data[:, i])
    elif final_grouping[i, 8] == 1:
        cluster9.append(new_x_data[:, i])
    else:
        cluster10.append(new_x_data[:, i])

n = len(x_combined_vec)
total_number_of_pairs = n * (n - 2) / 2

# number_righ_clusters = 0
# for i in range(len(labels)):
#     for j in range(len(labels)):
#         if labels[i] == labels[j] and y[i] == y[j]:
#             number_righ_clusters += 1

# clustering_error = (total_number_of_pairs - number_righ_clusters)/total_number_of_pairs
# print(clustering_error)
#
# for i in range(len(labels)):
#     if labels[i] == 0:
#         cluster1.append(new_x_data[:, i])
#     elif labels[i] == 1:
#         cluster2.append(new_x_data[:, i])
#     elif labels[i] == 2:
#         cluster3.append(new_x_data[:, i])
#     elif labels[i] == 3:
#         cluster4.append(new_x_data[:, i])
#     elif labels[i] == 4:
#         cluster5.append(new_x_data[:, i])
#     elif labels[i] == 5:
#         cluster6.append(new_x_data[:, i])
#     elif labels[i] == 6:
#         cluster7.append(new_x_data[:, i])
#     elif labels[i] == 7:
#         cluster8.append(new_x_data[:, i])
#     elif labels[i] == 8:
#         cluster9.append(new_x_data[:, i])
#     else:
#         cluster10.append(new_x_data[:, i])


cluster1 = np.array(cluster1).T
cluster2 = np.array(cluster2).T
Y1_cluster1 = cluster1[0, :]
Y2_cluster1 = cluster1[1, :]
Y1_cluster2 = cluster2[0, :]
Y2_cluster2 = cluster2[1, :]
cluster3 = np.array(cluster3).T
cluster4 = np.array(cluster4).T
Y1_cluster3 = cluster3[0, :]
Y2_cluster3 = cluster3[1, :]
Y1_cluster4 = cluster4[0, :]
Y2_cluster4 = cluster4[1, :]
cluster5 = np.array(cluster5).T
cluster6 = np.array(cluster6).T
Y1_cluster5 = cluster5[0, :]
Y2_cluster5 = cluster5[1, :]
Y1_cluster6 = cluster6[0, :]
Y2_cluster6 = cluster6[1, :]
cluster7 = np.array(cluster7).T
cluster8 = np.array(cluster8).T
Y1_cluster7 = cluster7[0, :]
Y2_cluster7 = cluster7[1, :]
Y1_cluster8 = cluster8[0, :]
Y2_cluster8 = cluster8[1, :]
cluster9 = np.array(cluster9).T
cluster10 = np.array(cluster10).T
Y1_cluster9 = cluster9[0, :]
Y2_cluster9 = cluster9[1, :]
Y1_cluster10 = cluster10[0, :]
Y2_cluster10 = cluster10[1, :]
#
plt.title("Data in 2-dim after applying Spectral Clustering(KNN=5, 1/2sigma=15) on original dataset")
plt.scatter(Y1_cluster1, Y2_cluster1, color='red')
plt.scatter(Y1_cluster2, Y2_cluster2, color='blue')
plt.scatter(Y1_cluster3, Y2_cluster3, color='green')
plt.scatter(Y1_cluster4, Y2_cluster4, color='brown')
plt.scatter(Y1_cluster5, Y2_cluster5, color='yellow')
plt.scatter(Y1_cluster6, Y2_cluster6, color='pink')
plt.scatter(Y1_cluster7, Y2_cluster7, color='magenta')
plt.scatter(Y1_cluster8, Y2_cluster8, color='gray')
plt.scatter(Y1_cluster9, Y2_cluster9, color='purple')
plt.scatter(Y1_cluster10, Y2_cluster10, color='orange')

plt.show()