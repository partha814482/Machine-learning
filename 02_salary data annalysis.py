# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.feature_selection import VarianceThreshold
# Load file
dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\SENAPATI SIR FSDS NOTE\SEPTMBER MONTH DS NOTE\22nd, 23rd- slr\22nd, 23rd- slr\SIMPLE LINEAR REGRESSION\Salary_Data.csv")
dataset.head()
dataset.columns
dataset.isnull().sum()
dataset.dtypes
x = dataset.drop('YearsExperience',axis = 1)
y = dataset['Salary']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1234,test_size=0.3)
# x = Input data,y = output data
x_train.ndim
# 1 dimension means 1 column only
# 2 dimension means 2 column only
# when you have only one column , the shape will not show the column
# (21,) it is only one column data having 21 observation
# (9,) it is one column data having 9 observation 
# (30,2) it is 2 column data having 30 observation
# Reshape the data if you have only one column

# LinearRegression
LR = LinearRegression()
LR.fit(x_train,y_train)
# Model predections happens x_test
y_prediction = LR.predict(x_test)
print(y_prediction)

x_test.iloc[0] # serise
# In order to pass a test sample to a model
# we need to pass a list of values 
# on array of values
# tuple of values
x_test.iloc[0].vlaues

LR.predict([x_test.iloc[0].values,
            x_test.iloc[1].values])
ip1 = [5]
LR.predict([ip1])

x_test.shape,y_test.shape,y_prediction.shape

test_data = x_test
test_data['y_actual'] = y_test
test_data['y_prediction'] = y_prediction 
# y_test n serise 
# y_pridiction is numpy array value 
print(y_test[:5]) # float 5. mean 5.0
print(y_prediction[:5])

# RMSE
# MSE
# MAE
# R-square

R2 = r2_score(y_test,y_prediction)
MSE = mean_squared_error(y_test,y_prediction)
# MSE**(1/2)
RMSE = np.sqrt(MSE)
# accuracy_score(y_test,y_predection) it is a regression tech
print('R-square:',R2)
print('MSE:',MSE)
print('RMSE:',RMSE)

s = 0
for i in range(len(y_test)):
    v1 = y_test.values[i]-y_prediction[i]
    v2 = v1**R2
    s = s+v2
print(s/len(y_test))

LR.coef_
print('The coefficent of years of exprience is :',LR.coef_)

LR.intercept_
x_train.columns

# Regression_equation = LR.intercept_ + LR.coef_ * col name
# Regression equation

#y = -1.45 + 1 * Salary

vt = VarianceThreshold(threshold = 0)
# Threshold variance value 
# we want to drop the feature based on threshold
vt.fit(dataset) 

dir(vt)
vt.variances_
# 300 is the first column variance (T)
# 1.25 is second column variance (T)
# 30 is column variance (T)
# 0 is fourth column variance (F)

vt.get_support()
vt.get_params()
# Hyper parameter 
# that we are providing insisd the function

vt.threshold 

cols = vt.get_feature_names_out()
# the above syntax gives the column name
# these feature only we want include
dataset[cols]

dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\SENAPATI SIR FSDS NOTE\SEPTMBER MONTH DS NOTE\22nd, 23rd- slr\22nd, 23rd- slr\SIMPLE LINEAR REGRESSION\Salary_Data.csv")
dataset.head()
vt = VarianceThreshold(threshold=0)
# make sure before fitting the dataframe ,do not including output column
x = dataset.drop('YearsExperience',axis = 1)
# x it self a data frame
vt.fit(x)
vt.variances_
vt.get_support()
cols = vt.get_feature_names_out()
x[cols]

from statsmodels.api import sm
#Add constant to x for the intercept them
x_train_const = sm.add_constant(x_train)
# fit OLS model
model = sm.OLS(y_train, x_train_const).fit()
# print regression summery
print(model.summery())

 
 

