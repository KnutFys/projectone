# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:26:39 2019

"""

# Import packages.
# Numpy DESCRIUPTION
import numpy as np
from time import time
t0 = time()
#Set up functions
def MakeData(datapoints):
    #This function  makes a "coordinate system"
    #that the Frankefunction can be evaluated on. 
    #It takes an integer valued argument that defines the coarseness of
    #the grid.
    a = np.arange(0, 1, 1/datapoints)
    b = np.arange(0, 1, 1/datapoints)
    x, y = np.meshgrid(a,b)
    return x, y

def FrankeFunction(x,y, noiceLevel=0):
    #This function calculates the Franke function on the meshgrids x, y
    #It then adds noice, drawn from a normal distribution, with mean 
    #0, std  noiceLevel, to each calculated datapoint of 
    # the Franke function.
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    if noiceLevel != 0:
        noice = np.random.normal(0,noiceLevel,x.shape)#*noiceLevel        
        return term1 + term2 + term3 + term4 , noice
    else:
        noice = np.zeros(x.shape)
        return term1 + term2 + term3 + term4 , noice

def DefineTheDesignMatrix(toorder,v1,v2):
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
    #This function calculates and returns the R2 value between a 
    #
    return 1-np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)



def MSE(y_data,y_model):
    #This function calculates and returns the mean squared error 
    #between y_data and y_model
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def confint(beta, V, sigma):
    #Returns an array of 95% confidence intervals around the betas.
    #The argument beta is a vector of estimators to build thel
    #intervals around. V is the covariance matrix of beta, divided
    #by the estimated sigma squared. Sigma is the estimator for the
    #std of the noice
    interval = np.zeros((2,len(beta)))
    for i in range(len(beta)):
        interval[0,i] = beta[i]-(1.645*np.sqrt(V[i,i])*sigma)
        interval[1,i] = beta[i]+(1.645*np.sqrt(V[i,i])*sigma)
    return interval
        
#Setting the number of datapoints, the amount of noise 
#in the Franke function and the order of the polinomial
#used as a model.
number_of_datapoints = 40
polinomial_order = 8
level_of_noice = 0

#Making the input and output vektors of the dataset
x, y = MakeData(number_of_datapoints)
z , noice = FrankeFunction(x, y, level_of_noice)
noiceDim1 = np.ravel(noice)
xDim1 = np.ravel(x)
yDim1 = np.ravel(y)
true_function_values = np.ravel(z)
noicy_function_values = true_function_values + np.ravel(noice)

BigX = DefineTheDesignMatrix(polinomial_order,xDim1,yDim1)        
BigX

#Fitting using linalg operations and beta=(X^TX)^-1Xy
beta = (np.linalg.inv(BigX.T@BigX)@BigX.T).dot(noicy_function_values)
ytilde = BigX @ beta
print("MSE value, no training split:", MSE(true_function_values,ytilde))
print("R2 value, no training split:", R2(true_function_values,ytilde))
delta = true_function_values-ytilde
sigma_squared = delta.dot(delta)/(len(true_function_values)-len(beta)-1)
var_beta = sigma_squared**2*np.linalg.inv(BigX.T@BigX)
print("Var(beta)= ",np.diag(var_beta))
#print(np.linalg.lstsq(BigX,zDim1))





from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection='3d')


# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(BigX,noicy_function_values, test_size=0.33, random_state=42)

X_traintrue, X_testtrue, y_traintrue, y_testtrue = \
train_test_split(BigX,true_function_values, test_size=0.33, random_state=42)

beta2 = (np.linalg.inv(X_train.T@X_train)@X_train.T).dot(y_train)
ytilde = X_train @ beta2
ypredict = X_test @ beta2
print("MSE value on training set:", MSE(y_traintrue,ytilde))
print("R2 value on training set:", R2(y_traintrue,ytilde))
print("MSE value on testing set:", MSE(y_testtrue,ypredict))
print("R2 value on testing set:", R2(y_testtrue,ypredict))
print(beta-beta2)
covar_beta = np.linalg.inv(BigX.T@BigX)
covar_beta2 = np.cov(BigX)

print("Runtime: %g sec" % (time()-t0))
delta =y_traintrue-ytilde
sigma_squared = delta.dot(delta)/(len(true_function_values)-len(beta)-1)
var_beta = sigma_squared**2*np.linalg.inv(BigX.T@BigX)
print("Var(beta) for training set ",np.diag(var_beta))
delta = y_testtrue-ypredict
sigma_squared = delta.dot(delta)/(len(true_function_values)-len(beta)-1)
var_beta = sigma_squared**2*np.linalg.inv(BigX.T@BigX)
print("Var(beta) for test set ",np.diag(var_beta))

conf_beta = (confint(beta2,np.linalg.inv(BigX.T@BigX),np.sqrt(sigma_squared)))

def fivefold(tfv,nfv,A,k=5, it=1600):
    iterations = len(A[:,0])
    tfv_split = np.zeros(iterations)
    nfv_split = np.zeros(iterations)
    A_split = np.zeros((iterations,len(A[0,:])))
    r2 = 0
    mse = 0
    #pick out, without replacement random elements of the true function, 
    #the function with noice
    #and the corresponding row of the vandermonde matrix
    for j in range(k):
        for i in range(320):
            if iterations > 1:
                index =320*j+i
                random_element = np.random.randint(low=0, high=iterations-1)
                A_split[index,:] = A[random_element,:]
                tfv_split[index]= tfv[random_element]
                nfv_split[index]= nfv[random_element]
                A = np.delete(A,random_element,0)
                tfv = np.delete(tfv,random_element,0)
                nfv = np.delete(nfv,random_element,0)
                iterations = iterations -1
    #adds the final remaining elements
    print("delta", tfv_split-nfv_split)
    print(tfv.shape, A_split[1598])
    print(tfv_split.shape, nfv_split.shape, A_split.shape)
    A_split[iterations-1,:] = A[0,:]
    tfv_split[iterations-1] = tfv[0]
    nfv_split[iterations-1] = nfv[0]
    for j in range(k):
        teststart = j*320
        teststop = (j+1)*320
        A_train = np.delete(A_split, slice(teststart,teststop),0)
        tfv_train = np.delete(tfv_split,slice(teststart,teststop))
        nfv_train = np.delete(nfv_split,slice(teststart,teststop))
        A_test = A_split[teststart:teststop,:]
        tfv_test = tfv_split[teststart:teststop]
        nfv_test = nfv_split[teststart:teststop]
        beta = (np.linalg.inv(A_train.T@A_train)@A_train.T).dot(nfv_train)

        ypredict = A_test @ beta
        mse = mse + MSE(tfv_test,ypredict)
        r2 = r2 + R2(tfv_test,ypredict)
    mse_av, r2_av = mse/k, r2/k
        
    print("The average MSE was: " ,mse_av)
    print("The average R2 score was: ", r2_av)
    return mse_av, r2_av
fivefold(true_function_values,noicy_function_values,BigX)

#test to see if its added in git
