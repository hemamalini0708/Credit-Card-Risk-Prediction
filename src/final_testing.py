import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from log_file import phase_1
import sys
logger = phase_1("Final_testing")

def testing_():
    try:
        model_ = pickle.load(open("credit_card.pkl",'rb'))
        print(type(model_))
        temp = np.random.random((5,2))
        temp = temp.ravel()
        if model_.predict([temp])[0] == 0:
            return 'Bad Customer'
        else:
            return 'Good Customer'
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

