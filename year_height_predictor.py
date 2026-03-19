# library imports
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# data set reading
mydata = pd.read_csv("year.csv")
x = mydata[["year"]]
y = mydata[["height"]]

# training and model creation
model = LinearRegression()
model.fit(x,y)

cf = model.coef_
print("Coefficient = ", cf)
intercept_value = model.intercept_
print("Intercept = ", intercept_value)

# model evaluation
y_pred = model.predict(x)
mse = mean_squared_error(y,y_pred)
print("MSE = ", mse)
rmse = np.sqrt(mse)
print("RMSE = ", rmse)

# predicting new value
new_height = model.predict([[8]])
print("Predicted height = ", new_height)

# visualization
plt.scatter(x,y)
plt.plot(x,y_pred, color = 'red')
plt.show()