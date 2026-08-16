import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib


data={
    'x':[12,20,18,29,33,45,52,51,52,55,53,55,61,64,69,72],
    'y':[39,36,52,54,46,55,59,63,70,66,58,23,14,8,19,24]
}
#CONVERT adata into a pandas dataframe
df=pd.DataFrame(data)

#plot data
plt.scatter(df['x'],df['y'],c='blue',s=50,alpha=0.6)
plt.title('Data plot before apply k-means')
plt.xlabel('X-Data')
plt.ylabel('Y-Data')
plt.show()


#apply k-means clustering use k means k=3 cluster
kmeans=KMeans(n_clusters=3,random_state=0)
# learn k means from data in dataframe df
kmeans.fit(df)
labels=kmeans.predict(df)
centroids=kmeans.cluster_centers_

plt.scatter(df['x'],df['y'],c=labels,cmap='viridis',s=50)
plt.scatter(centroids[:,0],centroids[:,1],c='red',s=200,marker='*')
plt.title('Data plot after apply k-means')
plt.xlabel('X-Data')
plt.ylabel('Y-Data')
plt.show()


print("Clusters Labels:")
print(centroids)

print("\nlabels:")
print(labels)

joblib.dump(kmeans, 'kmeans.pkl')
print("The Model Saved !")