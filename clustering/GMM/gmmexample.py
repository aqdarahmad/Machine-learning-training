import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pickle

data_path = 'customer_data.csv'
data=pd.read_csv(data_path)


categorical_columns = ['gender','education','country']
numerical_columns = ['age','income', 'purchase_frequency','spending']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le



scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

X = data[categorical_columns + numerical_columns]

gmm = GaussianMixture(n_components=8, random_state=42)
gmm.fit(X)



labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

print("\nCluster assignments:")
print(labels)

print("\nProbabilities:")
print(probabilities)

print("\nMeans:")
print(gmm.means_)

print("\nWeights:")
print(gmm.weights_)

with open('gmm_model.pkl', 'wb') as model_file:
    pickle.dump(gmm, model_file)


with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)


print("Model training complete. Files saved: gmm_model.pkl, scaler.pkl, label_encoders.pkl")