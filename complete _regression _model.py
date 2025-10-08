import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\SENAPATI SIR FSDS NOTE\OCTOBER MONTH DS NOTE\6th - poly\6th - poly\1.POLYNOMIAL REGRESSION\emp_sal.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Linear regression visualization
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# polynomal regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

# Again linear model build with degree 2

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# poly model visualization

plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

# svr model

from sklearn.svm import SVR
svr_model = SVR(kernel='poly',degree=4,gamma='auto',C = 10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print(svr_model_pred)

svr_model = SVR(kernel='rbf',degree=4,gamma='scale',C = 10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print(svr_model_pred)

svr_model = SVR(kernel='rbf',degree=4,gamma='scale',C = 8.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print(svr_model_pred)

# knn regression model

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors= 5,weights = 'distance',algorithm = 'brute',p = 2)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

# decission tree model

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(x,y)

dt_model_pred = dt_model.predict([[6.5]])
print(dt_model_pred)

# Random forest model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators= 23, random_state = 0)
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)
