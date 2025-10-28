from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [1, 2],
    [3, 2],
    [4, 2],
    [1, 4],
    [2, 6],
    [2, 7],
])
y = np.array([1, 0, 1, 0, 0, 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Actual:", y_train)
print("Predicted:", y_pred)
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train,s= 80, label =  'Actual')
plt.scatter(X_test[:, 0], X_test[:, 1], c  = y_pred, s =  100, marker = '*', label = 'Predicted')
plt.title("KNN")
plt.grid(True)
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


