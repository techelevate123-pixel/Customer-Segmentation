import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA

# --------------------------------------
# LOAD DATA
# --------------------------------------

df = pd.read_csv("data/customers.csv")
print("data loaded successfully!")
print(df.head())

# --------------------------------------
# SCALING
# --------------------------------------

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# --------------------------------------
# ELBOW METHOD (find best k)
# --------------------------------------

wcss = []

for k in range(1,10):
    Kmeans = KMeans(n_clusters=k, random_state=42)
    Kmeans.fit(x_scaled)
    wcss.append(Kmeans.inertia_)

plt.plot(range(1,10), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# ----------------------------------------
# KMEANS MODEL
# ----------------------------------------

Kmeans = KMeans(n_clusters=5, random_state=42)
df["kmeans_cluster"] = Kmeans.fit_predict(x_scaled)

# ----------------------------------------
# DBSCAN MODEL
# ----------------------------------------

dbscan = DBSCAN(eps=0.8, min_samples=5)
df["dbscan_cluster "] = dbscan.fit_predict(x_scaled)

# ----------------------------------------
# PCA (for Visualization)
# ----------------------------------------

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# ----------------------------------------
# VISUALIZE KMEANS CLUSTERS
# ----------------------------------------

plt.scatter(x_pca[:,0], x_pca[:,1], c=df["kmeans_cluster"])
plt.title("Kmeans Clusters")
plt.show()

# ----------------------------------------
# INSIGHTS
# ----------------------------------------

print("\nClusters Insights:\n")
print(df.groupby("kmeans_cluster").mean())