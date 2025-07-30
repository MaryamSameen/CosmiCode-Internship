import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
means = gmm.means_
covariances = gmm.covariances_

# Visualize clusters and probability ellipses
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
from matplotlib.patches import Ellipse

for i, color in enumerate(colors):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s=20, color=color, label=f'Cluster {i+1}')
    # Draw ellipse for each Gaussian component
    mean = means[i]
    cov = covariances[i]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', linewidth=2)
    plt.gca().add_patch(ellipse)

plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()