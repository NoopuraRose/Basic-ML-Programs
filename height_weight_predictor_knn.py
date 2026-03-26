import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

mydata = pd.read_csv("data.csv")
x = mydata[["height"]]
y = mydata[["weight"]]

rmse_values = []

for i in range(2,7):
    model = KNeighborsRegressor(n_neighbors = i)
    model.fit(x,y)
    y_pred = model.predict(x)
    mse = mean_squared_error(y,y_pred)
    # print("MSE = ", mse)
    rmse = np.sqrt(mse)
    # print("RMSE = ", rmse)
    rmse_values.append(rmse)

print(rmse_values)

best_k = range(2,7) [np.argmin(rmse_values)]
print(best_k)

model = KNeighborsRegressor(n_neighbors = best_k)
model.fit(x,y)

new_weight = model.predict([[160]])
print("Predicted weight = ", new_weight)