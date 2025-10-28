import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.DataFrame({
    'area': [1000,1500,2000,2500,3000],
    'bedrooms': [2,3,3,4,5],
    'age': [10,5,8,4,2],
    'price': [50,60,70,80,90]
})

X = df[['area','bedrooms','age']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted price [2500,4,5]:", model.predict([[2500,4,5]])[0])
plt.scatter(y, model.predict(X), color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (Multiple Linear Regression)')
plt.grid(True)
plt.show()