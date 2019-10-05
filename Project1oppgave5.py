# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:19:02 2019

@author: KH.Engvik
"""

# Import packages.
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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

def franke_function(x,y, noiceLevel=0,seed=1):
    #This function calculates Franke's function on the meshgrids x, y
    #It then adds noice, drawn from a normal distribution, with mean 
    #0, std 1 and scaled by noiceLevel, to each calculated datapoint of 
    # the Franke function.
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    if noiceLevel != 0:
        np.random.seed(seed)
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




def oppgave_5(o=5,level_of_noise=0, seed=1):
    #Setting the number of datapoints, the amount of noise 
    #in the Franke function and the order of the polinomial
    #used as a model.
    number_of_datapoints = 40
    np.random.seed(seed)
    #Making the input and output vektors of the dataset
    x, y = make_data(number_of_datapoints)
    z , noise = franke_function(x, y, level_of_noise,seed)
    #Flattening matrices for easier handling
    xDim1 = np.ravel(x)
    yDim1 = np.ravel(y)
    x = xDim1
    y = yDim1
    #Frankes function withou noise
    true = np.ravel(z)
    #Frankes function with noise
    noicy = true + np.ravel(noise)

    #Create instances of sklearn kFold klass to split data for kfoldcv
    lasso_regression = Lasso()
    #Create instances of sklearn kFold klass to split data for kfoldcv
    splits = 5
    kfold = KFold(n_splits = splits, shuffle=True)
    #Sets a range of polynomial orders to fit to the data
    polynomial_order = np.arange(o)+1
    #Set a range og shrinkage factors for the LASSO regression
    alphas = np.logspace(-5,-2,10)
    #Creates a dictionary to store dataframes for each LASSO parameter
    dataframe_dic = dict()
    
    for alph in alphas:
        print("Calculating LASSO, alpha: {}".format(alph))
        #Creates a list to store the results of each iteration in
        dta = list()  
        for order in polynomial_order:
            print("Using polynomial order {}".format(order))
            #Creating designmatrix
            A = design_matrix(order,x,y)
            #Remove intecept
            A = A[:,1:] 
            alpha_mse_test = np.zeros(splits)
            alpha_mse_train = np.zeros(splits)
            counter = 0
            #Initiating kfold cv
            for train_index, test_index in kfold.split(noicy):
                print("Calculating fold {} of {}".format(counter+1,splits))
                X_train, X_test = A[train_index], A[test_index]
                y_train = noicy[train_index]
                y_train_true, y_test_true = true[train_index], true[test_index]
                #Using current aplha and polynomial order solve using Lasso
                lasso_regression.alpha = alph
                lasso_regression.fit(X_train,y_train)
                #Estimate testing and training data
                ypredict = lasso_regression.predict(X_test)
                ytilde = lasso_regression.predict(X_train)
                #Get MSE metric for training and testing data    
                alpha_mse_test[counter] = MSE(y_test_true,ypredict)
                alpha_mse_train[counter] = MSE(y_train_true,ytilde)
                counter = counter + 1                
                print("Running time: {} seconds".format(time()-t0))
            dta.append(["{}".format(order),alpha_mse_test.mean(),
                        alpha_mse_train.mean()])
        df = pd.DataFrame(dta,columns=[
                "Polynomial","MSE test set","MSE training set"])
        dataframe_dic[alph] = df
        
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
                 color = cmap(float(n)/len(alphas)),
                 label="Alpha=%10.2E" %(df))
        ax2.plot(dataframe_dic[df]["Polynomial"],
                 dataframe_dic[df]["MSE test set"],
                 color = cmap(float(n)/len(alphas)),
                 label="Alpha=%10.2E" %(df))
        print("alpha:",df)
        print(dataframe_dic[df])
        n = n+1
    fig1.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig2.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    if level_of_noise < 0.21:
        lvl = "low"
    elif level_of_noise < 0.41:
        lvl = "med"
    else:
        lvl = "high"
    fig1.savefig("{}Oppg4LATrainSeed{}{}.png".format(plots_path,seed,lvl))
    fig2.savefig("{}Oppg4LATestSeed{}{}.png".format(plots_path,seed,lvl))

#Oppgave_5 function can be called using an integer argument  giving the 
#maximum order of polynomial to fit. Second argument determines lvl of noise
#and third argument set random seed
for i in range(1,4):
    oppgave_5(15,0.6,i*5)

print("Running time: {} seconds".format(time()-t0))