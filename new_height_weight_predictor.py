import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

mydata = pd.read_csv("new_data.csv")
x = mydata[["height"]]
y = mydata[["weight"]]

model = LinearRegression()
model.fit(x,y)

cf = model.coef_
print("Coefficient = ", cf)
intercept_value = model.intercept_
print("Intercept = ", intercept_value)

y_pred = model.predict(x)
mse = mean_squared_error(y,y_pred)
print("MSE = ", mse)
rmse = np.sqrt(mse)
print("RMSE = ", rmse)

new_weight = model.predict([[160]])
print("Predict weight = ", new_weight)

plt.scatter(x,y)
plt.plot(x,y_pred, color = 'red')
plt.show()