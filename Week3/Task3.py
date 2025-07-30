import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs

# Generate a noisy dataset
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
# Add random noise
noise = np.random.uniform(low=-6, high=6, size=(30, 2))
X_noisy = np.vstack([X, noise])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_noisy)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
km_labels = kmeans.fit_predict(X_noisy)

# Plot DBSCAN results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
unique_labels = set(db_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise
        col = [0, 0, 0, 1]
    class_member_mask = (db_labels == k)
    plt.scatter(X_noisy[class_member_mask, 0], X_noisy[class_member_mask, 1], c=[col], label=f'Cluster {k}')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot KMeans results
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X_noisy[km_labels == i, 0], X_noisy[km_labels == i, 1], label=f'Cluster {i}')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()