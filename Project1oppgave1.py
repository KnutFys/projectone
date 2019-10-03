# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:22:18 2019

@author: Knut Hauge Engvik
"""

# Import packages.
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
t0 = time()
plots_path = r"e:\Data\Plots\Plot"
text_path = r"e:\Data\Text\Result"

class Problem:
    #This class is basicly an instance of the  linear equation
    #Ab = y. It is initiated with a design_matrix and a data_set corresponding
    #to A and y. An optional argument true_data can be given if one knows the 
    #true underlying function  that is to be approximated.
    #
    #This is perhaps a rather convoluted way to aproach this project, but
    #as I am rather new to Python I wanted to explore different ways to
    #structure my coding.
    #
    def __init__(self,data_set,design_matrix,true_data=None):
        #Sets the design matrix and data set to solve
        self.data = data_set
        self.M = design_matrix
        #Checks for availability of noise free data
        if true_data is None:
            self.truedata = self.data
        else:
            self.truedata = true_data
        #Determines whether the class should use sklearn variants or
        #selfwritten code
        self.use_skl = False
        self.seed = 1
        self.silent = True
        self.tts=False
        
    def solve(self):
        #Uses self written function  predict() to solve itself.
        #If tts = True it will use sklearn train_test_split algorithm
        #

        results = predict(
                self.M,self.data,self.truedata, split=self.tts,
                silent=self.silent,seed=self.seed)
        self.beta = results.get("beta")
        self.msetest = results.get("msetest")
        self.r2 = results.get("R2")
        self.var = results.get("var")
        self.msetrain = results.get("msetrain")
        self.r2train = results.get("R2train")
        self.prediction = self.M @ self.beta

def make_data(datapoints):
    #This function  makes a "coordinate system"
    #that the Frankefunction can be evaluated on. 
    #It takes an integer valued argument that defines the coarseness of
    #the grid.
    a = np.linspace(0, 1, datapoints)
    b = np.linspace(0, 1, datapoints)
    x, y = np.meshgrid(a,b)
    return x, y

def franke_function(x,y, noise_level=0):
    #This function calculates Franke's function on the meshgrids x, y
    #It then adds noice, drawn from a normal distribution, with mean 
    #0, std 1 and scaled by noiceLevel, to each calculated datapoint of 
    # the Franke function.
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    if noise_level != 0:
        np.random.seed(1)
        noise = np.random.randn(x.shape[0],x.shape[1])*noise_level
        return term1 + term2 + term3 + term4 , noise
    else:
        noise = np.zeros(x.shape)
        return term1 + term2 + term3 + term4 , noise

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




def predict(A,data,true_data=None,split=False,seed=1, silent=True): 
    #This is a basic OLS solver using matrix inversion to solve
    #the normal equations A^TAb = A^Ty. The argument A is the problems design
    #matrix, data is the vector corresponding to y.
    #If available true_data is noice free data tha can bes used for testing
    #purposes. 
    #The split argument determens if the function should call train_test_split
    #from sklearn package to split the data into testing and training sets.
    #The silent argument just determines if the function should print to
    #screen wha it does. Implemented for debugging
    
    #Determine whether or no to split data into training/testing sets
    if split:
        #Splits data using train_test _split from sklearn
        X_train, X_test, y_train, y_test = train_test_split(
                A,data,test_size=0.20,random_state=seed,shuffle=True)
        #Find betas using matrix (pseudo)inversion to solve normal
        #equations
        beta = (np.linalg.pinv(X_train.T@X_train)@X_train.T).dot(y_train)
        ytilde = X_train @ beta

        #Overwrites the y_test vector conaining noicy data
        #with the real function values/noiceless data
        tmp = train_test_split(A,true_data, test_size=0.20,
                               random_state=seed,shuffle=True)
        y_train_true = tmp[2]
        y_test_true = tmp[3]
        ypredict = X_test @ beta
        msetrain = MSE(y_train_true,ytilde)
        msetest = MSE(y_test_true,ypredict)
        r2train = R2(y_train_true,ytilde)
        r2test = R2(y_test_true,ypredict)
        if not silent:
            #If one wants to the metrics as they are calculated
            #set silent=False
            print("MSE value on training set:", msetrain)
            print("R2 value on training set:", r2train)
            print("MSE value on testing set:", msetest)
            print("R2 value on testing set:", r2test)
        
        #Calculates unbiased estimator of sigma^2, the variance
        #of the noice
        #Based on training predictions
        delta =y_train_true-ytilde
        sigma_squared = delta.dot(delta)/(len(y_train_true)-len(beta)-1)
        #Estimate variance of betas in training set
        M = np.diag(np.linalg.pinv(X_train.T@X_train))
        var_beta = sigma_squared*np.sqrt(M)
        
        #Retun a dictionary of metrics for training and
        #testing predictions
        return {"beta":beta, "msetest":msetest,"R2":r2test,"var":var_beta
                ,"R2train":r2train,"msetrain":msetrain}
    else:
        #If split argument is not true then train_test_split is not employed
        #and all data is used both as testing and training data.
        
        #Find betas using matrix (pseudo)inversion to solve normal
        #equations
        beta = (np.linalg.pinv(A.T@A)@A.T).dot(data)
        ytilde = A @ beta
        msetest = MSE(true_data,ytilde)
        r2 = R2(true_data,ytilde)
        #Calculates unbiased estimator of sigma^2, the variance
        #of the noice
        delta = true_data-ytilde
        sigma_squared = delta.dot(delta)/(len(true_data)-len(beta)-1)
        #Estimate variance of betas
        M = np.diag(np.linalg.pinv(A.T@A))
        var_beta = sigma_squared*np.sqrt(M)
        if not silent:
            print("MSE value, no training split:", msetest)
            print("R2 value, no training split:", r2)
        return {"beta":beta, "msetest":msetest,"R2":r2,"var":var_beta}
            
def confint(V):
    #Returns an array of 95% confidence bounds given
    #a variance vector V
    n = len(V[0,:])
    interval = np.zeros(n)
    for i in range(n):
        interval[i] = 1.645*np.sqrt(V[i])
    return interval


#Part 1: Fitting a polynomial of order up to 5 to Frankes function

def oppgave_1(o=5,level_of_noise=0):
    print("Oppgave 2")
    #Setting the number of datapoints(pr axis), the amount of noise 
    #in the Franke function and the order of the polinomial
    #used as a model.
    number_of_datapoints = 40
    #Making the input and output vektors of the dataset
    x, y = make_data(number_of_datapoints)
    z , noise = franke_function(x, y, level_of_noise)
    #Flattening matrices for easier handling
    x = np.ravel(x)
    y = np.ravel(y)

    #Frankes function withou noice
    true = np.ravel(z)
    #Frankes function with noice
    noicy = true + np.ravel(noise)
    #Sets the range of polynomial orders to run through
    order = np.arange(o) + 1
    #Creates a list to store the results of each iteration in
    dta = list()
    #And an array to store the confidence intervals
    conf_int = np.zeros((int(((o+1)*(o+2))/2),o))
    for i in order:
        #Sets up teh design matrix
        X = design_matrix(i,x,y)
        #Creates an instance of the Prorblem class.
        prob = Problem(noicy,X,true)
        prob.tts=False
        prob.solve()
        dta.append(["{}".format(i),prob.msetest,prob.r2])
        for j in range(len(prob.var)):
            conf_int[j,i-1] = prob.var[j]
    #Make a dataframe for plotting 
    #MSE and R2 values and to familiarize myself with pandas
    df = pd.DataFrame(dta,columns=["Polynomial","MSE","R2"])
    
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Polynomial order")
    ax1.set_ylabel("MSE")
    ax1.plot(df["Polynomial"],df["MSE"], color = "g",label="MSE")
    ax2 = ax1.twinx()
    ax2.set_ylabel("R2")
    ax2.plot(df["Polynomial"],df["R2"], color = "r",label="R2")
    fig.legend(bbox_to_anchor=(0.5, 0.5, 0.4, 0.2))
    #This part is structured specificaly for part one of the project
    #to make table of confidence intervals. It will not work
    #for polynomila orders other than 5
    if o == 5:
        #Create a dataframe for easier vieving of confidence intervals
        df2 = pd.DataFrame(conf_int, columns=["Order 1","Order 2",
                                              "Order 3", "Order 4", "Order 5"])
        df2.index.name = "Beta"
        print(df2)
    #For saving figures and tables for report
    #--------------------------------------------------
    #fig.savefig("{}Oppgave1MSER2.png".format(plots_path))
    #df2.to_csv("{}oppgave1.csv".format(text_path))
  


#Oppgave_1 function can be called using an integer argument  giving the 
#maximum order of polynomial to fit.
oppgave_1(20,0.1)

print("Running time: {} seconds".format(time()-t0))