# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:19:02 2019

@author: KH.Engvik
"""

# Import packages.
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd

t0 = time()
plots_path = r"e:\Data\Plots\Plot"
text_path = r"e:\Data\Text\Result"

def make_data(datapoints):
    #This function  makes a "coordinate system"
    #that the Frankefunction can be evaluated on. 
    #It takes an integer valued argument that defines the coarseness of
    #the grid.
    a = np.linspace(0, 1, datapoints)
    b = np.linspace(0, 1, datapoints)
    x, y = np.meshgrid(a,b)
    return x, y

def franke_function(x,y, noiceLevel=0):
    #This function calculates Franke's function on the meshgrids x, y
    #It then adds noice, drawn from a normal distribution, with mean 
    #0, std 1 and scaled by noiceLevel, to each calculated datapoint of 
    # the Franke function.
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    if noiceLevel != 0:
        np.random.seed()
        noice = np.random.randn(x.shape[0],x.shape[1])*noiceLevel
        return term1 + term2 + term3 + term4 , noice
    else:
        noice = np.zeros(x.shape)
        return term1 + term2 + term3 + term4 , noice

def design_matrix(toorder,v1,v2):
    #This functions sets up a design matrix under the assumption that
    #we are looking for a function in two variables of degree n, n beeing 
    #specified in the argument toorder
    DM = np.zeros((len(v1),int(((toorder+1)*(toorder+2))/2)))
    DM[:,0] = 1
    teller = 1
    for n in range(toorder):
        for i in range(n+2):
            DM[:,teller] = (v1**i)*(v2**(n+1-i))
            teller = teller+1
    return DM


def R2(y_data, y_model):
    #This function calculates and returns the R2 value between for a model 
    #with prediction y_model and and the target of the predictions y_data
    return 1-np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)



def MSE(y_data,y_model):
    #This function calculates and returns the mean squared error 
    #between y_data and y_model
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

k = 3
linear_regression = LinearRegression()
lasso_regression = Lasso()
ridge_regression = Ridge(alpha=10, solver="svd")
kfold = KFold(n_splits = k, shuffle=True)
    
polynomial_order = np.arange(1,5)
lamdas = np.logspace(-3,1,5)
alphas = np.logspace(-3,1,5)
ols_mse = list()
ridge_mse = list()
lasso_mse = list()

#Setting the number of datapoints, the amount of noise 
#in the Franke function and the order of the polinomial
#used as a model.
number_of_datapoints = 40
level_of_noise = 0.1

#Making the input and output vektors of the dataset
x, y = make_data(number_of_datapoints)
z , noise = franke_function(x, y, level_of_noise)
#Flattening matrices for easier handling
noiceDim1 = np.ravel(noise)
xDim1 = np.ravel(x)
yDim1 = np.ravel(y)
x = xDim1
y = yDim1
#Frankes function withou noice
true = np.ravel(z)
#Frankes function with noice
noicy = true + np.ravel(noise)
#Sets the rang of polynomial orders to run through
o = 15
order = np.arange(o) + 1
#Creates a list to store the results of each iteration in
dta = list()   
for order in polynomial_order:
    A = design_matrix(order,x,y)
    ols_avg_mse = 0
    lamda_mse = np.zeros(len(lamdas))
    alpha_mse = np.zeros(len(lamdas))
        
    #Initiating kfold cv
    for train_index, test_index in kfold.split(noicy):
        X_train, X_test = A[train_index], A[test_index]
        y_train, y_test = noicy[train_index], noicy[test_index]
            
        #Solve using OLS
        linear_regression.fit(X_train,y_train)
        ytilde = linear_regression.predict(X_test)
        ols_avg_mse = ols_avg_mse + MSE(y_test,ytilde)
            
        #Run through the lamda parameters for the ridge regression and
        #the current polynomial order of the model
        for i in range(len(lamdas)):
            #Solve using Ridge and current lamda
            ridge_regression.alpha = lamdas[i]
            ridge_regression.fit(X_train,y_train)
            ytilde = ridge_regression.predict(X_test)
            lamda_mse[i] = lamda_mse[i]+MSE(y_test,ytilde)
                
        for i in range(len(alphas)):
            lasso_regression.alpha = alphas[i]
            lasso_regression.fit(X_train,y_train)
            ytilde = lasso_regression.predict(X_test)
            alpha_mse[i] = alpha_mse[i]+MSE(y_test,ytilde)
        #Store the results for the current polynomial order
        ridge_mse.append(lamda_mse/k)
        ols_mse.append(ols_avg_mse/k)
        lasso_mse.append(alpha_mse/k)
    print("OLS",ols_mse)
    print("RIDGE",ridge_mse)
    print("LASSO",lasso_mse)