# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:11:48 2019

@author: KnuHEngvik
"""

# Import packages.
import numpy as np
from time import time
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from imageio import imread
import matplotlib.pyplot as plt
import pandas as pd

t0 = time()
plots_path = r"e:\Data\Plots\Plot"
text_path = r"e:\Data\Text\Result"
image_path = "e:\\Data\\Data\\"


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

def MSE(y_data,y_model):
    #This function calculates and returns the mean squared error 
    #between y_data and y_model
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def oppgave_6(o=15,seed=4,test=True):
    
    # Load the terrain
    terrain = imread("{}SRTM_data_Norway_1.tif".format(image_path))
    # Show the terrain
    plt.figure()
    plt.title('Terrain Norway 1, Original')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    #Pick out  small square to analyze if test is set to True
    if test:
        #Pick out  small square to analyze
        square_size = 100
        x_shift = np.random.randint(0,1801-square_size)
        y_shift = np.random.randint(0,3601-square_size)
        terrain = terrain[y_shift:y_shift+square_size,
                                  x_shift:x_shift+square_size]
        plt.figure()
        plt.title('Terrain part 1, Original {} pt box'.format(square_size))
        plt.imshow(terrain, cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    else:
        #Use settings determined  by analysing small squares for analysis
        #on entire dataset. Attemting to rebuild the image from 
        #a model based on evenly spaced datapoints 
        
        #Set model parameters
        order = 15
        #Ridge parameter
        lmd = 0.0001
        #Lasso parameter
        alph = 0.0001
        #Set the coarseness of the sample grid
        coarseness = 5
        x_dimension_original = len(terrain[0,:])
        y_dimension_original = len(terrain[:,0])        
        x_dimension = x_dimension_original//coarseness
        y_dimension = y_dimension_original//coarseness
        terrain_points = np.zeros((y_dimension,x_dimension))
        for x_axis in range(x_dimension):
            for y_axis in range(y_dimension):
                terrain_points[y_axis,x_axis] = terrain[y_axis*coarseness,
                              x_axis*coarseness]
        #Create mesh grid for training data, selected points
        x = np.linspace(0, 1, x_dimension)
        y = np.linspace(0, 1, y_dimension)
        x_grid, y_grid = np.meshgrid(x,y)
        #Create meshgrid for original data
        x_original = np.linspace(0, 1, x_dimension_original)
        y_original = np.linspace(0, 1, y_dimension_original)
        x_grid_original, y_grid_original = np.meshgrid(x_original,y_original)
        #Flatten grids
        data = np.ravel(terrain_points)
        data_original = np.ravel(terrain)
        x = np.ravel(x_grid)
        y = np.ravel(y_grid)
        x_original = np.ravel(x_grid_original)
        y_original = np.ravel(y_grid_original)
        #Creates a scaler to normalize data
        scaler = MinMaxScaler()
        print("Running time: {} seconds".format(time()-t0))
        #Normalizing data
        scaler.fit(data.reshape(-1, 1))
        #Normalizing training data
        normalized_data = scaler.transform(data.reshape(-1, 1))
        normalized_data = normalized_data[:,0]
        #Normalizing original data --------not used?
        normalized_data_original = scaler.transform(data_original.reshape(
                -1, 1))
        normalized_data_original = normalized_data_original[:,0]
        
        #Initiate instances of the regressors
        linear_regression = LinearRegression()
        ridge_regression = Ridge(solver="svd",alpha=lmd)
        lasso_regression = Lasso(alpha=alph)
        print("Running time: {} seconds".format(time()-t0))
        #Create training matrix
        A = design_matrix(order,x,y)
        #Remove intercept
        A = A[:,1:]
        print("Running time: {} seconds".format(time()-t0))
        #Create prediction matrix
        X_test = design_matrix(order,x_original,y_original)
        X_test = X_test[:,1:]
        print("Running time: {} seconds".format(time()-t0))
        #Make prediction using OLS model
        linear_regression.fit(A,normalized_data)
        rebuilt =  linear_regression.predict(X_test)
        print("OLS MSE: ", MSE(normalized_data_original,rebuilt))
        rebuilt = scaler.inverse_transform(rebuilt.reshape(-1,1))
        rebuilt = np.reshape(rebuilt,y_grid_original.shape)
        fig_rebuild = plt.figure(figsize=(9,5))
        ax1 = fig_rebuild.add_subplot(131)
        ax2 = fig_rebuild.add_subplot(132)
        ax3 = fig_rebuild.add_subplot(133)
        plt.title('Terrain Norway 1, rebuild')
        ax1.imshow(rebuilt, cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        #Make prediction using Ridge model
        ridge_regression.fit(A,normalized_data)
        rebuilt =  ridge_regression.predict(X_test)
        print("Ridge MSE: ", MSE(normalized_data_original,rebuilt))
        rebuilt = scaler.inverse_transform(rebuilt.reshape(-1,1))
        print("Running time: {} seconds".format(time()-t0))
        rebuilt = np.reshape(rebuilt,y_grid_original.shape)
        ax2.imshow(rebuilt, cmap='gray')

        #Make prediction using LASSO model
        lasso_regression.fit(A,normalized_data)
        rebuilt =  lasso_regression.predict(X_test)
        print("LASSO MSE: ", MSE(normalized_data_original,rebuilt))
        rebuilt = scaler.inverse_transform(rebuilt.reshape(-1,1))
        print("Running time: {} seconds".format(time()-t0))
        rebuilt = np.reshape(rebuilt,y_grid_original.shape)
        ax3.imshow(rebuilt, cmap='gray')
        
        fig_rebuild.savefig("{}TerrainRebuilOrder{}P4.png".format(
                plots_path,order))

        return()
        
    #Get dimensions of data set and make a grid to base the model on
    y_dimension = len(terrain[:,0])
    x_dimension = len(terrain[0,:])

    x = np.linspace(0, 1, x_dimension)
    y = np.linspace(0, 1, y_dimension)
    x_grid, y_grid = np.meshgrid(x,y)
    #Flatten grid
    data = np.ravel(terrain)
    x = np.ravel(x_grid)
    y = np.ravel(y_grid)
    #set random seed
    np.random.seed(seed)
    
    #Creates a scaler to normalize data
    scaler = MinMaxScaler()
    
    #Normalizing data
    scaler.fit(data.reshape(-1, 1))
    normalized_data = scaler.transform(data.reshape(-1, 1))
    normalized_data = normalized_data[:,0]
    
    
    #Create instances of sklearn kFold klass to split data for kfoldcv
    splits = 5
    kfold = KFold(n_splits = splits, shuffle=True)
    #Sets a range of polynomial orders to fit to the data
    polynomial_order = np.arange(o)+1
    
    
    #---------OLS------------------------------
    #------------------------------------------
    #Solve using OLS

    linear_regression = LinearRegression()

    dta = list()  
    for order in polynomial_order:
        print("Using polynomial order {}".format(order))
        #Creating designmatrix
        A = design_matrix(order,x,y)
        mse_test = np.zeros(splits)
        mse_train = np.zeros(splits)
        counter = 0
        #Initiating kfold cv
        for train_index, test_index in kfold.split(normalized_data):
            print("Calculating fold {} of {}".format(counter+1,splits))
            X_train, X_test = A[train_index], A[test_index]
            y_train, y_test = normalized_data[
                    train_index],normalized_data[test_index]
            #Using current  polynomial order and fold to solve using OLS
            linear_regression.fit(X_train,y_train)
            ytilde = linear_regression.predict(X_train)
            ypredict = linear_regression.predict(X_test)
            #Get MSE metric for training and testing data
            mse_test[counter] = MSE(y_test,ypredict)
            mse_train[counter] = MSE(y_train,ytilde)
            counter = counter + 1 
            print(counter)               
            print("Running time: {} seconds".format(time()-t0))
            
        dta.append(["{}".format(order),mse_test.mean(),mse_train.mean()])
            
        '''     
        rebuilt =  linear_regression.predict(A)
        rebuilt = scaler.inverse_transform(rebuilt.reshape(-1,1))
        rebuilt = np.reshape(rebuilt,y_grid.shape)
        plt.figure()
        plt.title('Terrain Norway 1, rebuild')
        plt.imshow(rebuilt, cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        '''    
    df = pd.DataFrame(dta,columns=["Polynomial",
                                   "MSE test set","MSE training set"])
    
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
    ax1.plot(df["Polynomial"],df["MSE training set"],label="Training OLS")
    ax2.plot(df["Polynomial"],df["MSE test set"],label="Test OLS")
    fig1.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig2.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig1.savefig("{}TerrainOLStrainSeed{}.png".format(plots_path,seed))
    fig2.savefig("{}TerrainOLStestSeed{}.png".format(plots_path,seed))
            

    #---------RIDGE----------------------------
    #------------------------------------------
    #Creates a dictionary to store dataframes for each Ridge parameter
    dataframe_dic = dict()
    ridge_regression = Ridge(solver="svd")
    #Set a range og shrinkage factors for the Ridge regression
    lambdas = np.logspace(-5,-1,10)
    for lmd in lambdas:
        print("Calculating Ridge, lambda: {}".format(lmd))
        #Creates a list to store the results of each iteration in
        dta = list()  
        for order in polynomial_order:
            print("Using polynomial order {}".format(order))
            #Creating designmatrix
            A = design_matrix(order,x,y)
            #Removing intercept
            A = A[:,1:]
            lambda_mse_test = np.zeros(splits)
            lambda_mse_train = np.zeros(splits)
            counter = 0
            #Initiating kfold cv
            for train_index, test_index in kfold.split(normalized_data):
                X_train, X_test = A[train_index], A[test_index]
                y_train, y_test = normalized_data[
                        train_index],normalized_data[test_index]
                #Using current lambda and polynomial order solve using Ridge
                ridge_regression.alpha = lmd
                ridge_regression.fit(X_train,y_train)
                #Estimate testing and training data
                ypredict = ridge_regression.predict(X_test)
                ytilde = ridge_regression.predict(X_train)
                #Get MSE metric for training and testing data    
                lambda_mse_test[counter] = MSE(y_test,ypredict)
                lambda_mse_train[counter] = MSE(y_train,ytilde)
                print("Calculating fold {} of {}".format(counter+1,splits))
                counter = counter + 1                
                print("Running time: {} seconds".format(time()-t0))
            dta.append(["{}".format(order),lambda_mse_test.mean(),
                        lambda_mse_train.mean()])
            
            '''
            rebuilt =  ridge_regression.predict(A)
            rebuilt = scaler.inverse_transform(rebuilt.reshape(-1,1))
            rebuilt = np.reshape(rebuilt,y_grid.shape)
            plt.figure()
            plt.title('Terrain Norway 1, rebuild')
            plt.imshow(rebuilt, cmap='gray')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
            '''
        df = pd.DataFrame(dta,columns=[
                "Polynomial","MSE test set","MSE training set"])
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
                 label="Alpha=%10.2E" %(df))
        ax2.plot(dataframe_dic[df]["Polynomial"],
                 dataframe_dic[df]["MSE test set"],
                 color = cmap(float(n)/len(lambdas)),
                 label="Alpha=%10.2E" %(df))
        n = n+1
    fig1.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig2.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig1.savefig("{}TerrainRidgetrainSeed{}.png".format(plots_path,seed))
    fig2.savefig("{}TerrainRidgetestSeed{}.png".format(plots_path,seed))
    
    #---------LASSO----------------------------
    #------------------------------------------

    
    #Create an instance of the Lasso class from sklearn
    lasso_regression = Lasso()
    #Set a range og shrinkage factors for the LASSO regression
    alphas = np.logspace(-5,-2,10)
    dataframe_dic = dict()
    for alph in alphas:
        print("Calculating LASSO, alpha: {}".format(alph))
        #Creates a list to store the results of each iteration in
        dta = list()  
        for order in polynomial_order:
            print("Using polynomial order {}".format(order))
            #Creating designmatrix
            A = design_matrix(order,x,y)
            #Removing intercept
            A = A[:,1:]
            alpha_mse_test = np.zeros(splits)
            alpha_mse_train = np.zeros(splits)
            counter = 0
            #Initiating kfold cv
            for train_index, test_index in kfold.split(normalized_data):
                X_train, X_test = A[train_index], A[test_index]
                y_train, y_test = normalized_data[
                        train_index],normalized_data[test_index]
                #Using current aplha and polynomial order solve using Lasso
                lasso_regression.alpha = alph
                lasso_regression.fit(X_train,y_train)
                #Estimate testing and training data
                ypredict = lasso_regression.predict(X_test)
                ytilde = lasso_regression.predict(X_train)
                #Get MSE metric for training and testing data    
                alpha_mse_test[counter] = MSE(y_test,ypredict)
                alpha_mse_train[counter] = MSE(y_train,ytilde)
                print("Calculating fold {} of {}".format(counter+1,splits))
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
        n = n+1
    fig1.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig2.legend(bbox_to_anchor=(0.71, 0.5),loc="center left", borderaxespad=0)
    fig1.savefig("{}TerrainLASSOtrainSeed{}.png".format(plots_path,seed))
    fig2.savefig("{}TerrainLASSOtestSeed{}.png".format(plots_path,seed))
    
    
#Oppgave_6 function can be called using an integer argument  giving the 
#maximum order of polynomial to fit. Second argument  random seed
oppgave_6(o=15,test=True)

print("Running time: {} seconds".format(time()-t0))