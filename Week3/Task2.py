import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Generate synthetic dataset
np.random.seed(42)
X1 = np.random.randn(50, 2) + np.array([2, 2])
X2 = np.random.randn(50, 2) + np.array([-2, -2])
X3 = np.random.randn(50, 2) + np.array([2, -2])
X = np.vstack((X1, X2, X3))

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Decide number of clusters (e.g., 3) by inspecting dendrogram
num_clusters = 3
labels = fcluster(Z, num_clusters, criterion='maxclust')

# Visualize clusters
plt.figure(figsize=(8, 6))
for i in range(1, num_clusters + 1):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
plt.title('Clusters determined by Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()