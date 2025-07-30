import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs

# Generate synthetic dataset with outliers
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_outliers = np.vstack([X, outliers])

# Isolation Forest
iso_forest = IsolationForest(contamination=0.06, random_state=42)
iso_pred = iso_forest.fit_predict(X_with_outliers)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.06)
lof_pred = lof.fit_predict(X_with_outliers)

# Plot results
plt.figure(figsize=(12, 5))

# Isolation Forest plot
plt.subplot(1, 2, 1)
plt.scatter(X_with_outliers[iso_pred == 1, 0], X_with_outliers[iso_pred == 1, 1], c='b', label='Normal')
plt.scatter(X_with_outliers[iso_pred == -1, 0], X_with_outliers[iso_pred == -1, 1], c='r', label='Anomaly')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# LOF plot
plt.subplot(1, 2, 2)
plt.scatter(X_with_outliers[lof_pred == 1, 0], X_with_outliers[lof_pred == 1, 1], c='b', label='Normal')
plt.scatter(X_with_outliers[lof_pred == -1, 0], X_with_outliers[lof_pred == -1, 1], c='r', label='Anomaly')
plt.title('Local Outlier Factor (LOF) Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()