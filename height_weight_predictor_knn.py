import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor

mydata = pd.read_csv("data.csv")
x = mydata[["height"]]
y = mydata[["weight"]]

model = KNeighborsRegressor(n_neighbors = 2)
model.fit(x,y)

new_weight = model.predict([[160]])
print("Predicted weight = ", new_weight)