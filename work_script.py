# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:54:32 2019

@author: Gaurav
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("weatherHistory.csv")

categorical=df.select_dtypes(include=["object"]).keys()
print(categorical)
quantitative=df.select_dtypes(include=["float64"]).keys()
print(quantitative)

#checking if any quantative has zero effect on temperature
df[quantitative].hist()

#Dropping Loud Cover as zero effect
df=df.drop('Loud Cover', axis=1)

#SimpleImputer to replace 0 in pressure(millibars)
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=0, strategy='median')
df.iloc[:, 9:10]=imp.fit_transform(df.iloc[:, 9:10])

imp=SimpleImputer(missing_values=np.nan, strategy="constant")
df.iloc[:, 2:3]=imp.fit_transform(df.iloc[:,2:3])

X=df.iloc[:, [0,1,2,4,5,6,7,8,9,10]].values
Y=df.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:, 2]=labelencoder_X.fit_transform(X[:, 2])
X[:, 9]=labelencoder_X.fit_transform(X[:, 9])
ohe=OneHotEncoder(categorical_features=[0,1,2,9])
X=ohe.fit_transform(X).toarray()  # --> This is throwing memory error sometimes

#Before Transformation, X had 11 features
#After one hot transformation, X has 373 features

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, random_state=40)

#Training the model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)

y_pred=reg.predict(x_test)

#Checking accuracy
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

#Error comes out to be 0.8846

######################################################################################

#Now will trying finding p-values and doing backward elimination
#Function for elimination takenn from Udemy
import statsmodels.api as sm
X=np.append(arr=np.ones((96453,1)).astype(int), values=X, axis=1)
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x 
SL = 0.05
cols=np.arange(start=0, stop=373, step=1)
X_opt = X[:, cols]
X_Modeled = backwardElimination(X_opt, SL)

#After backward elimination, features reduced form 373 to 90 for X_Modeled
#Generating training and testing datasets
#Fitting the training_optimised datasets to Linear Regressor
from sklearn.model_selection import train_test_split
x_train_opt, x_test_opt, y_train_opt, y_test_opt=train_test_split(X_Modeled, Y, test_size=0.25, random_state=40)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train_opt, y_train_opt)

y_pred_opt=reg.predict(x_test_opt)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test_opt, y_pred_opt)

#Error comes out to be 0.8802

########################################################################################

#Difference in errors ar 0.0044 after removing dummy variables
#Now for plotting the graph using X_Modeled
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = x_test[0:100,86] #Index for humidity is 86
z = y_pred[0:100]
y = x_test[0:100,88] #Index for pressure is 88
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('Humidity')
ax.set_zlabel('Temperature')
ax.set_ylabel('Pressure(millibars)')
plt.show()

  