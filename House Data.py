# import library
import numpy as np # linear algebra
import pandas as pd # Data processing
import matplotlib.pyplot as plt
from subprocess import check_output
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# load dataset
dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\SENAPATI SIR FSDS NOTE\SEPTMBER MONTH DS NOTE\22nd, 23rd- slr\22nd, 23rd- slr\SLR - House price prediction\House_data.csv")
space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1,1)
y = np.array(price)

# Spliting the data into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state = 0)

# fitting simple linear regression model to the trraining set
regressior = LinearRegression()
regressior.fit(x_train,y_train) 

# predecting the prices
pred = regressior.predict(x_test)

# visuallize the traning test result
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressior.predict(x_train),color='blue')
plt.title('Visualize the Traning dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

# Visualizing the test result
plt.scatter(x_test, y_test, color='red')  # scatter plot for test data
plt.plot(x_train, regressor.predict(x_train), color='blue')  # regression line
plt.title('Visualization for Test Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()
