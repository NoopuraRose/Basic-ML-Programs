import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

mydata = pd.read_csv("experience.csv")
x = mydata[["experience"]]
y = mydata[["salary"]]

plt.scatter(x,y)
plt.show()

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

new_salary = model.predict([[7]])
print("Predict salary = ", new_salary)