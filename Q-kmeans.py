# 0. Adjust parameters
NUM_CLUSTERS = 4       # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = 20           # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)
from sklearn.datasets import load_iris

# 2. Load the Iris dataset using scikit-learn's load_iris() function
data = load_iris()
X = data.data

# 3. Implement K-Means Clustering
# 3.1. Import KMeans from scikit-learn
from sklearn.cluster import KMeans
# 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
kmeans = KMeans(n_clusters = NUM_CLUSTERS, max_iter = MAX_ITER)
# 3.3. Fit the KMeans model to the data X
kmeans.fit(X)
# 3.4. Obtain the cluster labels
labels = kmeans.labels_

# 4. Visualize the Results
import matplotlib.pyplot as plot
# 4.1. Extract the features for visualization
plot.figure(figsize = (10, 6))
plot.scatter(X[:, FEATURE_X_INDEX],
            X[:, FEATURE_Y_INDEX],
            c = labels,
            cmap = 'viridis')
# 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
centers = kmeans.cluster_centers_
plot.scatter(centers[:, FEATURE_X_INDEX],
            centers[:, FEATURE_Y_INDEX],
            c = 'red',
            marker = 'x',
            s = 200,)
# 4.3. Use different colors to represent different clusters
plot.xlabel(data.feature_names[FEATURE_X_INDEX])
plot.ylabel(data.feature_names[FEATURE_Y_INDEX])
plot.title(f'K-Means Clustering (n_clusters = {NUM_CLUSTERS}, max_iter = {MAX_ITER})', fontsize = 14)
plot.show()