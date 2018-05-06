import numpy as np


class KMeans(object):
    def __init__(self, L=100, K=2):
        self.L = L
        self.K = K

    def cost_function(self, centroids_group, X, centroids):
        cost = 0
        for i in range(len(X)):
            for j in range(self.K):
                cost += centroids_group[i, j] * np.linalg.norm(X[i] - centroids[j], 2)

        return cost

    def clustering(self, x_data):
        X = np.array(x_data).T
        X_copy = np.copy(X)
        centroids_group = np.zeros((len(X), self.K))
        min_cost = 10000000
        final_centroids_group = []
        final_centroids = []
        for l in range(self.L):
            print("interation : ", l)
            np.random.shuffle(X_copy)
            centroids = X_copy[0:self.K, :]
            centroids = np.array(centroids)
            prev_cost = 0
            while True:
                for i in range(len(X)):
                    min_distance = 1000000
                    centroid = 0
                    for j in range(len(centroids)):
                        distance = np.linalg.norm(X[i] - centroids[j], 2)

                        if distance < min_distance:
                            min_distance = distance
                            centroid = j

                    for k in range(len(centroids_group[i, :])):
                        if k == centroid:
                            centroids_group[i, k] = 1
                        else:
                            centroids_group[i, k] = 0

                change = 0

                for c in range(len(centroids_group[0, :])):
                    points_in_cluster = 0
                    sum_of_points = 0
                    a = 0
                    for z in range(len(centroids_group[:, c])):
                        a = z
                        if centroids_group[z, c] == 1:
                            sum_of_points += X[z, :]
                            points_in_cluster += 1
                    # print(points_in_cluster)
                    # print(sum_of_points)
                    if points_in_cluster == 0:
                        sum_of_points = X[a, :]
                        points_in_cluster = 1
                    new_centroid = sum_of_points/points_in_cluster

                    if not np.array_equal(centroids[c], new_centroid):
                        change += 1
                        centroids[c] = new_centroid
                total_cost = self.cost_function(centroids_group, X, centroids)

                if prev_cost == total_cost and change == 0:
                    break
                else:
                    prev_cost = total_cost
            if total_cost < min_cost:
                min_cost = total_cost
                final_centroids = centroids
                final_centroids_group = centroids_group
        num_of_points_in_clusters = np.sum(final_centroids_group, axis=0)
        points_in_cluster = []
        for j in range(len(final_centroids_group[0, :])):
            points_in_a_cluster = []
            for i in range(len(final_centroids_group[:, j])):
                if final_centroids_group[i, j] == 1:
                    points_in_a_cluster.append(X[i, :])
            points_in_cluster.append(points_in_a_cluster)
        points_in_cluster = np.array(points_in_cluster)
        return final_centroids, num_of_points_in_clusters, points_in_cluster, final_centroids_group