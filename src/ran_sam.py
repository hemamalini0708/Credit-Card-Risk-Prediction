import numpy as np
import pandas as pd
import sklearn



def mis_data_ran(X_train,X_test):
    try:
        var = ["MonthlyIncome","NumberOfDependents"]
        for i in var:
            X_train[i+'_ran_sam'] = X_train[i].copy() # shallow copy
            s = X_train[i].dropna().sample(X_train[i].isnull().sum(), random_state=42)
            s.index = X_train[X_train[i].isnull()].index
            X_train.loc[X_train[i].isnull(), i+'_ran_sam'] = s

            X_test[i + '_ran_sam'] = X_test[i].copy()  # shallow copy
            s = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
            s.index = X_test[X_test[i].isnull()].index
            X_test.loc[X_test[i].isnull(), i+'_ran_sam'] = s
            X_train = X_train.drop([i],axis=1)
            X_test = X_test.drop([i], axis=1)
        return X_train,X_test
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")




