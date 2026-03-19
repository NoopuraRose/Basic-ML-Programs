# library imports
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np 

# data set reading
mydata = pd.read_csv("data.csv")
x = mydata[["height"]]
y = mydata[["weight"]]

# visualization
plt.scatter(x,y)
plt.show()

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

# predicting a new value
new_weight = model.predict([[160]])
print("Predict weight = ", new_weight)