import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

mydata = pd.read_csv("time.csv")
x = mydata[["time"]]
y = mydata[["weight_loss"]]

plt.scatter(x,y)
plt.show()

model = LinearRegression()
model.fit(x,y)

cf = model.coef_
print("Coefficient = ", cf)
intercept_value = model.intercept_
print("Intercept = ", intercept_value)

new_weightloss = model.predict([[12]])
print("Predict weight loss = ", new_weightloss)