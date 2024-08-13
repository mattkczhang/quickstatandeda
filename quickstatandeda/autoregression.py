import pandas as pd
import numpy as np
import statsmodels.api as sm
from htmlpreparation import getModelResults, allCombinations

def forwardSelection(x, y):
    """Forward selection of linear regression

    Args:
        x (pd.DataFrame): features
        y (pd.Series): target

    Returns:
        pd.DataFrame: table that showcases the steps of forward selection
    """
    if len(x) != len(y):
        print('The number of rows of features and target is not matched. Check out their length!')
        return None

    predictors = {}
    for i in x.columns.tolist():
        predictors['+'+i] = x[i]
    
    trainData = pd.DataFrame(y)
    trainData.insert(0, 'intercept', 1.0)
    trainData = trainData[['intercept']]

    model = sm.OLS(y, trainData[['intercept']], missing='drop').fit()
    results0 = getModelResults(model)

    step_summary = pd.DataFrame([[
        'Intercept', results0['AIC'], results0['BIC'], results0['R-squared'], 
        results0['Adjusted R-sqaured'], results0['Log-likelihood'],results0['P-value'], np.nan]])
    
    min_p = 1
    model0 = model
    name = ''

    while predictors != {}:
        for key in predictors:
            trainData_temp = trainData.join(predictors[key])
            model = sm.OLS(y, trainData_temp, missing='drop').fit()
            results1 = getModelResults(model)

            if np.isnan(results1['Adjusted R-sqaured']) or np.isnan(results1['P-value']):
                step_summary.reset_index(inplace=True)
                step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance'] 
                return step_summary

            f_test_p = model.compare_f_test(model0)[1]
            # print(key, f_test_p)       
            if f_test_p < min_p:
                min_p = f_test_p
                name = key
                results = results1
        if min_p > 0.05 or min_p == 1:
            break

        step_summary = pd.concat([step_summary, pd.DataFrame([[name, results['AIC'], results['BIC'], results['R-squared'], 
                                                                   results['Adjusted R-sqaured'], results['Log-likelihood'],results['P-value'], min_p]])], ignore_index = True)
        trainData = trainData.join(predictors[name])
        predictors.pop(name)
        min_p = 1
        model0 = model

    step_summary.reset_index(inplace=True)
    step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance']    
    return step_summary


def backwardSelection(x, y):
    """Backward selection of linear regression

    Args:
        x (pd.DataFrame): features
        y (pd.Series): target

    Returns:
        pd.DataFrame: table that showcases the steps of backward selection
    """
    if len(x) != len(y):
        print('The number of rows of features and target is not matched. Check out their length!')
        return None

    predictors = {}
    for i in x.columns.tolist():
        predictors[i] = x[i]
    
    trainData = pd.DataFrame(x)
    trainData.insert(0, 'intercept', 1.0)

    model = sm.OLS(y, trainData, missing='drop').fit()
    results0 = getModelResults(model)

    step_summary = pd.DataFrame([[
        'Full Model', results0['AIC'], results0['BIC'], results0['R-squared'], 
        results0['Adjusted R-sqaured'], results0['Log-likelihood'],results0['P-value'], np.nan]])
    
    max_p = 0
    model0 = model
    name = ''

    while predictors != {}:
        for key in predictors:
            trainData_temp = trainData.drop(columns=[key])
            model = sm.OLS(y, trainData_temp, missing='drop').fit()
            results1 = getModelResults(model)

            if np.isnan(results1['Adjusted R-sqaured']) or np.isnan(results1['P-value']):
                step_summary.reset_index(inplace=True)
                step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance']    
                return step_summary

            f_test_p = model0.compare_f_test(model)[1]
            # print(key, f_test_p)       
            if f_test_p > max_p:
                max_p = f_test_p
                name = key
                results = results1
        
        if max_p > 0.05 or max_p == 0:
            break            

        step_summary = pd.concat([step_summary, pd.DataFrame([['-'+name, results['AIC'], results['BIC'], results['R-squared'], 
                                                                   results['Adjusted R-sqaured'], results['Log-likelihood'],results['P-value'], max_p]])], ignore_index = True)
        trainData = trainData.drop(columns=[name])
        predictors.pop(name)
        max_p = 0
        model0 = model

    step_summary.reset_index(inplace=True)
    step_summary.columns = ['Step', 'Predictor Entered', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                       'Log-likelihood', 'P-value', 'F-test significance']    
    return step_summary


def allPossibleSelection(x, y):
    """All possible selection of linear regression

    Args:
        x (pd.DataFrame): features
        y (pd.Series): target

    Returns:
        pd.DataFrame: table that showcases all the possible combinations of features and the corresponding model metrics
    """
    if len(x) != len(y):
        print('The number of rows of features and target is not matched. Check out their length!')
        return None

    if len(x.columns) > 0:
        combs = allCombinations(x.columns)
        step_summary = pd.DataFrame()
        for i in combs:
            trainData = pd.DataFrame(x[list(i)])
            trainData.insert(0, 'intercept', 1.0)
            model = sm.OLS(y, trainData, missing='drop').fit()
            results = getModelResults(model)
            step_summary = pd.concat([step_summary, 
                                    pd.DataFrame([[str(list(i)).replace('[','').replace(']','').replace(', ', ' + ').replace("'",""), 
                                                    results['AIC'], results['BIC'], results['R-squared'], 
                                                    results['Adjusted R-sqaured'], results['Log-likelihood'],
                                                    results['P-value']]])], ignore_index=True)

        step_summary.reset_index(inplace=True)
        step_summary.columns = ['Index', 'Predictors', 'AIC', 'BIC', 'R-squared', 'Adjusted R-sqaured',
                        'Log-likelihood', 'P-value']    
        
        return step_summary
    else:
        return None
