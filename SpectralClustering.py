import numpy as np
from KMeans import KMeans
import matplotlib.pyplot as plt


class SpectralClustering(object):
    def __init__(self, sigma=1, K=2, KNN=10):
        self.sigma = sigma
        self.K = K
        self.KNN = KNN

    def similarity_matrix(self, x_data):
        similarity_matrix = np.zeros((len(x_data), len(x_data)))
        gamma = 1 / (2 * np.square(self.sigma))
        for i in range(len(x_data)):
            distance_vector = []
            for j in range(len(x_data)):
                distance = np.linalg.norm((x_data[j, :] - x_data[i, :]), 2)
                distance_vector.append(distance)
            similarity_matrix[i, :] = distance_vector
            distance_vector.sort()
            max_distance = distance_vector[self.KNN]
            similarity_row = similarity_matrix[i, :]
            for k in range(len(similarity_row)):
                if similarity_row[k] <= max_distance and similarity_row[k] != 0:
                    weight = np.exp(-np.square(np.linalg.norm((x_data[i, :] - x_data[k, :]), 2)) * gamma)
                    similarity_matrix[i, k] = weight
                else:
                    similarity_matrix[i, k] = 0
        similarity_matrix = np.array(similarity_matrix)
        similarity_matrix = np.array(np.maximum(similarity_matrix, similarity_matrix.T))
        return similarity_matrix

    def clustering(self, x_data):
        similarity_matrix = self.similarity_matrix(x_data.T)
        degree_vector = []
        for weight_row in similarity_matrix:
            degree = np.sum(weight_row)
            degree_vector.append(degree)
        degree_matrix = np.diag(np.array(degree_vector))
        laplacian_matrix = degree_matrix - similarity_matrix
        eig_values, eig_vectors = np.linalg.eig(laplacian_matrix)
        idx = eig_values.argsort()
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
        eig_vectors_smallest = eig_vectors[:, 0:self.K]
        # print(eig_vectors_smallest)
        # cluster1 = []
        # cluster2 = []
        # for i in range(len(eig_vectors_smallest[:, 0])):
        #     if eig_vectors_smallest[i, 0] == 0:
        #         print("dsjfndfj")
        #         cluster1.append(eig_vectors_smallest[i])
        #     else:
        #         cluster2.append(eig_vectors_smallest[i])
        # cluster1 = np.array(cluster1)
        # cluster2 = np.array(cluster2)
        # X11 = cluster1[:, 0]
        # X21 = cluster1[:, 1]
        # X12 = cluster2[:, 0]
        # X22 = cluster2[:, 1]
        # plt.title("Clustering of 2-dim(axes found using Spectral Clustering) data using Spectral Clustering on dataset2")
        # plt.scatter(X11, X21, color="r")
        # plt.scatter(X12, X22, color="b")
        # plt.show()
        kmeans = KMeans(K=self.K, L=1)
        centroids, num_of_points_group, grouping, final_grouping = kmeans.clustering(np.real(eig_vectors_smallest.T))
        clusters = []
        for j in range(len(final_grouping[0, :])):
            cluster_points = []
            for i in range(len(final_grouping[:, j])):
                if final_grouping[i, j] == 1:
                    cluster_points.append(i)
            clusters.append(cluster_points)
        return clusters, final_grouping
