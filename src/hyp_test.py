import sys
import numpy as np
import pandas as pd
from log_file import phase_1
import matplotlib.pyplot as plt
logger = phase_1("hyp_test")
from scipy.stats import pearsonr

def hypothesis(X_train,y_train,X_test,y_test):
    try:
        cor_p_value = []
        waste_cols = []
        for i in X_train.columns:
            sol = pearsonr(X_train[i] , y_train)
            cor_p_value.append(sol)
        cor_p_value = np.array(cor_p_value)
        p_value = pd.Series(cor_p_value[:, 1], index=X_train.columns)
        for j in p_value.index:
            if p_value[j] > 0.05:
                waste_cols.append(j)
                logger.info(f"Targeted cols : {j} : {p_value[j]}")
        X_train = X_train.drop(waste_cols,axis=1)
        X_test = X_test.drop(waste_cols,axis=1)
        return X_train,X_test

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")