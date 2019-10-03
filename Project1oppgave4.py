# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:24:30 2019

@author: NerdusMaximus
"""

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
        #added lamda parameter for ridge regression
        self.lamda = 0
        
    def solve(self):
        #Uses self written function  predict() to solve itself.
        #If tts = True it will use sklearn train_test_split algorithm
        #

        results = predict(
                self.M,self.data,self.truedata, split=self.tts,
                silent=self.silent,seed=self.seed)
        self.beta = results.get("beta")
        self.msetest = results.get("msetest")
        self.r2test = results.get("R2")
        self.var = results.get("var")
        self.msetrain = results.get("msetrain")
        self.r2train = results.get("R2train")
        self.prediction = self.M @ self.beta
    
    def split(self,folds=5):            
        print("Splitting data into {} folds.".format(folds))
        self.data_split, self.M_split, self.truedata_split = k_fold_split(
                self.data,self.M,self.truedata, k=folds)
        print("Splitting completed.")
    
        #Performs kfold cross validation
        n = len(self.data_split)
        #Arrays to store metrics for each fold
        mse_test = np.zeros(n)
        r2_test = np.zeros(n)
        mse_train = np.zeros(n)
        r2_train = np.zeros(n)
        #Creates an array to be used with a boolean mask to pick out 
        #training sets. 
        mas = np.arange(n)
        #Iterate over each fold and calculate MSE and R2
        for i in range(n):
            #Find the size of
            rows = len(self.M[:,0]) - len(self.M_split[i][:,0])
            cols = len(self.M[0,:])
            A = np.zeros((rows,cols))
            training_data = np.zeros(rows)
            test_data = np.array(self.truedata_split)
            #Create a boolean mask to hide the ith index
            mask = np.ones(n,bool)
            mask[i] = False                
            
            #iterate over elements to be added to training matrix and
            #training data
            for ind in mas[mask]:
                length_of_addition = len(self.M_split[ind])
                count = 0
                for ind2 in range(length_of_addition):
                    A[count,:] = self.M_split[ind][ind2]
                    training_data[count] = self.data_split[ind][ind2]
                    count = count + 1
            
            #solve for beta using normal equation and pseudoinverse
            if self.lamda != 0:
                AtA = A.T@A+self.lamda*np.eye(len(A[0,:]),len(A[0,:]))
                beta = (np.linalg.pinv(AtA)@A.T).dot(training_data)
            else:
                #If the ridge parameter is 0
                #solve for beta using normal equation and pseudoinverse
                #Not realy necessary to check
                #as lamda = 0 leads to OLS
                beta = (np.linalg.pinv(A.T@A)@A.T).dot(training_data)
            ytilde = A @ beta
            #predict data for the ith fold wich was left out of
            #the training data
            ypredict = self.M_split[i] @ beta
            
            #store metrics
            mse_test[i] =  MSE(test_data[i],ypredict)
            r2_test[i] = R2(test_data[i],ypredict)
            mse_train[i] =  MSE(training_data,ytilde)
            r2_train[i] = R2(training_data,ytilde)
        #Sets the MSE and R2 values to the average of the values calculated 
        #for each fold
        self.msetest = mse_test.mean()
        self.r2test = r2_test.mean()
        self.msetrain = mse_train.mean()
        self.r2train = r2_train.mean()

def k_fold_split(data,A,true_data=None, k=5):
    #This function partitions the design matrix and the datasets into
    #k sets/folds
    #
    # If noiseless data is not available this sets true_data = data
    #data is noicy function values i.e. frankes fuction with
    #noice, measurement data etc
    if true_data is None:
        true_data = data
    #Finds the number of data points that needs to be sorted into folds 
    iterations = len(A[:,0])
    #Calculates the base size of the folds. If the number ofdatapoints
    #is not divisible by the number of folds the remainder will be handled
    #later
    fold_size = iterations//k
    remainder = iterations - k*fold_size
    #Creates empty arrays to be filled
    true_data_split = []
    data_split = []
    A_split = []
    #pick out, without replacement random elements of the true function, 
    #the function with noice
    #and the corresponding row of the design matrix
    for j in range(k):
        A_fold = np.zeros((fold_size,len(A[0,:])))
        true_data_fold = np.zeros(fold_size)
        data_fold = np.zeros(fold_size)
        for i in range(fold_size):
            if iterations > 1:                
                random_element = np.random.randint(low=0, high=iterations)
                A_fold[i,:] = A[random_element,:]
                true_data_fold[i]= true_data[random_element]
                data_fold[i]= data[random_element]
                A = np.delete(A,random_element,0)
                true_data = np.delete(true_data,random_element,0)
                data = np.delete(data,random_element,0)
                iterations = iterations -1
        #Adds the jth fold to 
        A_split.append(A_fold)
        data_split.append(data_fold)
        true_data_split.append(true_data_fold)

        
    #adds leftover elements if sample size is not divisible by fold number 
    if remainder > 0:
        for i in range(0,remainder):
            random_element = np.random.randint(low=0, high=iterations)
            A_split[i] = np.vstack([A_split[i],A[random_element,:]])
            true_data_split[i] = np.append(true_data_split[i],
                           true_data[random_element])
            data_split[i] = np.append(data_split[i],data[random_element])
            A = np.delete(A,random_element,0)
            true_data = np.delete(true_data,random_element,0)
            data = np.delete(data,random_element,0)
            iterations = iterations -1 
    return data_split, A_split, true_data_split
    
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
        interval[i] = 1.96*np.sqrt(V[i])
    return interval


#Part 4: 
def oppgave_4(o=5,level_of_noise=0):
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

    lambdas = np.logspace(-5,0,10)
    dataframe_dic = dict()
    for lmd in lambdas:
        dta = list()
        print(lmd)
        for i in order:
            #Sets up teh design matrix
            X = design_matrix(i,x,y)
            #Creates an instance of the Prorblem class. 
            prob = Problem(noicy,X,true)
            #Sets the random seed
            prob.seed = 10
            #Uses own code to perform kfoldcv
            prob.lamda = lmd
            prob.split(5)
            print(lmd)
            print("Running time: {} seconds".format(time()-t0))
            dta.append(["{}".format(i),prob.msetest,prob.msetrain])
        df = pd.DataFrame(dta,columns=["Polynomial","MSE test set",
                                   "MSE training set"])
        dataframe_dic[lmd] = df
        
    cmap = plt.get_cmap('jet_r')    
    plt.figure()
    fig1 = plt.figure(figsize=(8,4))
    ax1 = fig1.add_subplot(111)
    ax1.set_position([0.1,0.1,0.6,0.8])
    ax1.set_xlabel("Polynomial order")
    ax1.set_ylabel("Training MSE")
    fig2 = plt.figure(figsize=(8,4))
    ax2 = fig2.add_subplot(111)
    ax2.set_position([0.1,0.1,0.6,0.8])
    ax2.set_xlabel("Polynomial order")
    ax2.set_ylabel("Testing MSE")
    n=0
    for df in dataframe_dic:        
        ax1.plot(dataframe_dic[df]["Polynomial"],
                 dataframe_dic[df]["MSE training set"],
                 color = cmap(float(n)/len(lambdas)),
                 label="Lambda=%10.2E" %(df))
        ax2.plot(dataframe_dic[df]["Polynomial"],
                 dataframe_dic[df]["MSE test set"],
                 color = cmap(float(n)/len(lambdas)),
                 label="Lambda=%10.2E" %(df))
        print(df, dataframe_dic[df])
        n = n+1
    fig1.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig2.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig1.savefig("{}Oppgave4ridgetrain.png".format(plots_path))
    fig2.savefig("{}Oppgave4ridgetest.png".format(plots_path))
    
#Oppgave_2 function can be called using an integer argument  giving the 
#maximum order of polynomial to fit. Second argument determines lvl of noise
oppgave_4(15,0.1)

print("Running time: {} seconds".format(time()-t0))