import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

mydata = pd.read_csv("data.csv")
x = mydata[["height"]]
y = mydata[["weight"]]

model = KNeighborsRegressor(n_neighbors = 2)
model.fit(x,y)

new_weight = model.predict([[160]])
print("Predicted weight = ", new_weight)

y_pred = model.predict(x)
mse = mean_squared_error(y,y_pred)
print("MSE = ", mse)
rmse = np.sqrt(mse)
print("RMSE = ", rmse)