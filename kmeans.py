import numpy as np
import pandas as pd


def make_clusters(X:np.ndarray, centroids:np.ndarray, k:int):
    dist_to_centroids = np.array([np.linalg.norm(X-c, axis=1) for c in centroids])

    clusters = [[] for _ in range(k)]
    for X_i, c_j in enumerate(np.argmin(dist_to_centroids, axis=0)):
        clusters[c_j].append(X_i)

    return clusters


def kmeanspp_init(X:np.ndarray, k:int):
    centroids = []
    np.random.seed(0)
    centroids.append(X[np.random.choice(np.arange(len(X)), 1, replace=False)].squeeze())

    for c_j in range(1, k):
        dist_to_centroids = np.array([np.linalg.norm(X-c, axis=1) for c in centroids])
        next_centroid_index = np.argmax(np.min(dist_to_centroids, axis=0), axis=0)
        centroids.append(X[next_centroid_index])

    return np.array(centroids)


def kmeans(X:np.ndarray, k:int, centroids=None, tolerance=1e-2):
    # Initialize centroids if not provided
    if centroids is None:
        np.random.seed(0)
        centroids = X[np.random.choice(np.arange(len(X)), k, replace=False)]
    elif centroids=="kmeans++":
        centroids = kmeanspp_init(X, k)
    # Initialize clusters
    clusters = make_clusters(X, centroids, k)

    old_centroids = None
    while (old_centroids is None) or (np.linalg.norm(centroids - old_centroids) > tolerance):
        old_centroids = centroids.copy()
        centroids = np.array([np.mean(X[c], axis=0) for c in clusters])
        clusters = make_clusters(X, centroids, k)

    return centroids, clusters
