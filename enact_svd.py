import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Assume 'adjective_matrix' is your data matrix with AN vectors
adjective_matrix = np.random.rand(100, 5)  # Example data

# Find the optimal number of clusters using silhouette score
silhouette_scores = []
K_range = range(2, 11)  # Assuming you want to test between 2 and 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(adjective_matrix)
    score = silhouette_score(adjective_matrix, cluster_labels)
    silhouette_scores.append(score)

# Plot the silhouette scores to find the optimal k
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Determine the optimal k (where silhouette score is highest)
optimal_k = K_range[np.argmax(silhouette_scores)]
print("Optimal number of clusters:", optimal_k)
