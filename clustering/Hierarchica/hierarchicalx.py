# Import NumPy for numerical operations
import numpy as np

# Import Matplotlib for data visualization
import matplotlib.pyplot as plt

# Import make_blobs to generate artificial clustered data
from sklearn.datasets import make_blobs

# Import AgglomerativeClustering for hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

# Import dendrogram and linkage to create the hierarchical tree
from scipy.cluster.hierarchy import dendrogram, linkage


# --------------------------------------------------
# 1. Generate sample data
# --------------------------------------------------

# Generate 200 data points divided into 4 groups
# cluster_std controls how spread out the points are
# random_state makes the generated data reproducible
X, _ = make_blobs(
    n_samples=200,
    centers=4,
    cluster_std=1.0,
    random_state=42
)


# --------------------------------------------------
# 2. Visualize the original data
# --------------------------------------------------

# Plot the data points
# X[:, 0] represents the first feature (X-axis)
# X[:, 1] represents the second feature (Y-axis)
plt.scatter(
    X[:, 0],
    X[:, 1],
    s=30,
    c='gray'
)

# Add a title to the plot
plt.title("Sample Data")

# Display the plot
plt.show()


# --------------------------------------------------
# 3. Perform Agglomerative Clustering
# --------------------------------------------------

# Create the Agglomerative Clustering model
# We want to divide the data into 4 clusters
# Ward linkage minimizes the variance within each cluster
agglo = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward'
)

# Fit the model to the data
# Then assign each data point to a cluster
labels = agglo.fit_predict(X)


# --------------------------------------------------
# 4. Visualize the clusters
# --------------------------------------------------

# Plot the data points and color them according to their cluster
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='viridis',
    s=30
)

# Add a title
plt.title("Agglomerative Clustering")

# Display the clustered data
plt.show()


# --------------------------------------------------
# 5. Create the hierarchical linkage
# --------------------------------------------------

# Calculate the hierarchical clustering structure
# Ward linkage is used to determine which clusters
# should be merged at each step
linked = linkage(
    X,
    method='ward'
)


# --------------------------------------------------
# 6. Create the Dendrogram
# --------------------------------------------------

# Create a figure with a width of 10 and height of 7
plt.figure(figsize=(10, 7))

# Draw the dendrogram
# truncate_mode='level' limits the displayed tree depth
# p=5 displays only the last 5 levels
# show_leaf_counts=True shows how many data points
# are contained in each displayed branch
dendrogram(
    linked,
    truncate_mode='level',
    p=5,
    show_leaf_counts=True
)

# Add a title
plt.title("Dendrogram")

# Display the dendrogram
plt.show()