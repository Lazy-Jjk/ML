import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_train[:, 0], y_train, color='blue', label='Train Data')
plt.scatter(X_test[:, 0], y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred, color='red', label='Prediction Line')
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)