
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler


from sklearn.cluster import AgglomerativeClustering

# Import linkage and dendrogram for creating the hierarchical tree
from scipy.cluster.hierarchy import dendrogram, linkage


# --------------------------------------------------
# 1. Create customer data
# --------------------------------------------------

# Create a dictionary containing customer information
data = {
    "Age": [22, 24, 23, 45, 48, 46, 30, 32, 29, 60],
    "Income": [1000, 1200, 1100, 5000, 5500, 5200, 2500, 2700, 2400, 7000],
    "Spending": [90, 85, 95, 30, 25, 35, 70, 75, 80, 20]
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Display the original customer data
print("Original Data:")
print(df)


# --------------------------------------------------
# 2. Select the features for clustering
# --------------------------------------------------

# Select the features that will be used to group customers
X = df[["Age", "Income", "Spending"]]


# --------------------------------------------------
# 3. Scale the features
# --------------------------------------------------

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the features so that all features have a similar impact
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 4. Create the Dendrogram
# --------------------------------------------------

# Calculate the hierarchical clustering using Ward linkage
Z = linkage(X_scaled, method="ward")

# Create a figure for the dendrogram
plt.figure(figsize=(10, 6))

# Plot the dendrogram
dendrogram(Z)

# Add a title to the plot
plt.title("Customer Hierarchical Clustering")

# Label the x-axis
plt.xlabel("Customers")

# Label the y-axis
plt.ylabel("Distance")

# Display the dendrogram
plt.show()


# --------------------------------------------------
# 5. Apply Agglomerative Hierarchical Clustering
# --------------------------------------------------

# Create the Agglomerative Clustering model
# We choose 3 clusters based on the dendrogram
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

# Fit the model and assign each customer to a cluster
labels = model.fit_predict(X_scaled)


# --------------------------------------------------
# 6. Add the cluster labels to the DataFrame
# --------------------------------------------------

# Add the cluster number to the original dataset
df["Cluster"] = labels

# Display the customers with their cluster assignments
print("\nCustomers with Cluster Labels:")
print(df)


# --------------------------------------------------
# 7. Analyze the customer segments
# --------------------------------------------------

# Calculate the average Age, Income, and Spending for each cluster
cluster_analysis = df.groupby("Cluster")[[
    "Age",
    "Income",
    "Spending"
]].mean()

# Display the average characteristics of each cluster
print("\nCluster Analysis:")
print(cluster_analysis)