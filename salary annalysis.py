import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\SENAPATI SIR FSDS NOTE\SEPTMBER MONTH DS NOTE\22th- slr\Salary_Data.csv")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)

y_pred = regression.predict(x_test)
comparision = pd.DataFrame({'Actual': y_test,'predicted':y_pred})
print(comparision)
# scatter plot
plt.scatter(x_test,y_test,color='red')# Real salary data (testing)
plt.plot(x_train,regression.predict(x_train),color='blue') # Regression lies from testing set
plt.title('Salary vs Exprience (Test set)')
plt.xlabel('Years of exprence')
plt.ylabel('Salary')
plt.show()


# slope
m = regression.coef_
print(m)
# constand
c = regression.intercept_
print(c)

y_12 = m* 12+c 
print(y_12)


y_23 = m* 23+c 
print(y_23)


y_30 = m * 30 +c
print(y_30)


