import joblib

kmeans = joblib.load('kmeans.pkl')
input_data=[[1,0]]
prediction=kmeans.predict(input_data)
print(prediction)