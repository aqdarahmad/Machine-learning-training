import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X = np.array([

[20,20],
[21,21],
[22, 19],
[23, 22],

[30, 30],
[31, 31],
[32, 29],
[33, 32]


]);

gmm = GaussianMixture(n_components=2, random_state=42)

gmm.fit(X)

labels = gmm.predict(X)
print(" Cluster labels:")
print(labels)


probabilities = gmm.predict_proba(X)

print(" Cluster probabilities:")
print(probabilities)
