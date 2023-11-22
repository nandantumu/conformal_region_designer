import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from .core import Clustering


class MeanShiftClustering(Clustering):
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        if self.bandwidth is None:
            self.bandwidth = estimate_bandwidth(X)
            if self.bandwidth == 0:
                self.bandwidth = 1e-10
        ms = MeanShift(bandwidth=self.bandwidth, n_jobs=8)
        ms.fit(X)
        self.cluster_centers_ = ms.cluster_centers_
        self.labels_ = ms.labels_

    def generate_clustered_points(self, X):
        clusters = [[] for _ in range(len(set(self.labels_)))]
        for i, x in enumerate(X):
            clusters[self.labels_[i]].append(x)
        for i, cluster in enumerate(clusters):
            clusters[i] = np.array(cluster)
        return clusters
