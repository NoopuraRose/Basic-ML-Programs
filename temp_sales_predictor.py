import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

mydata = pd.read_csv("temperature.csv")
x = mydata[["temperature"]]
y = mydata[["sales"]]

plt.scatter(x,y)
plt.show()

model = LinearRegression()
model.fit(x,y)

cf = model.coef_
print("Coefficient = ", cf)
intercept_value = model.intercept_
print("Intercept = ", intercept_value)

new_sales = model.predict([[32]])
print("Predicted sales = ", new_sales)
