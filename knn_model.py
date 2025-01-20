from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris_dataset = load_iris() 

#print(iris_dataset.keys())

#assigning the values for the tests and train variables 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#checking the shape of our X-train ad=nd y_train data
print(f"X_train {X_train.shape}")
print(f"y_train {y_train.shape}")

#checking the shape of the test data for both x and y
print(f"X_test {X_test.shape}")
print(f"y_test {y_test.shape}")

#defining your model
knn_model = KNeighborsClassifier(n_neighbors=1)

#fitting our model with the y_train and x_train data
knn_model.fit(X_train, y_train)

#making predictions 
#X_new = np.array([[5, 2.9, 1, 0.3]])
#print(f"x_new shape {X_new.shape}")

#prediction = knn_model.predict(X_new)
#print(f"Prediction: {prediction}")
#print(f"predicted target name: {iris_dataset['target_names'][prediction]}")\

y_pred = knn_model.predict(X_test)
print(f"Test set predictions: {y_pred}")

#We can also use the score method of the knn object, which will compute the test set
#accuracy for us:
print(f"test set score: {knn_model.score(X_test, y_test):.2f}")

