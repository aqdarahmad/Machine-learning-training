import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib


data = {
    'hours_listened': [2, 3, 4, 5, 6, 18, 20, 22, 24, 25],
    'songs_played':   [20, 25, 35, 40, 45, 150, 170, 190, 210, 220]
}
#CONVERT adata into a pandas dataframe
df=pd.DataFrame(data)

#plot data
plt.scatter(
    df['hours_listened'],
    df['songs_played']
)

plt.xlabel('Hours Listened')
plt.ylabel('Songs Played')
plt.title('Spotify Users')

plt.show()


kmeans = KMeans(
    n_clusters=2,
    random_state=0,
    n_init=10
)

kmeans.fit(df)

labels = kmeans.labels_

print(labels)

centroids=kmeans.cluster_centers_

plt.scatter(df['hours_listened'],df['songs_played'],c=labels,cmap='viridis',s=50)
plt.scatter(centroids[:,0],centroids[:,1],c='red',s=200,marker='*')
plt.title('Data plot after apply k-means')
plt.xlabel('Hours Listened')
plt.ylabel('Songs Played')
plt.show()


print("Clusters Labels:")
print(centroids)

print("\nlabels:")
print(labels)

joblib.dump(kmeans, 'kmeans.pkl')
print("The Model Saved !")