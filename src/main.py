'''
in this file we load data and call functions which are needed for model development
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import time
import sys
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')
from log_file import phase_1
logger = phase_1("main")
from sklearn.model_selection import train_test_split
from ran_sam import mis_data_ran
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder()
from one_h_enc import conversion_cat_num
from odinal_enc import cat_num_odinal_enc
from feature_selection_area import constant
from hyp_test import hypothesis
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from models import multi_models
from best_model import selecting_best_model
from final_model import f_m
from final_testing import testing_
class SERVICE:
    def __init__(self,path):
        try:
            self.path=path
            self.df = pd.read_csv(self.path)
            logger.info("The Data Loaded successfully")
            logger.info(f"The number of rows : {self.df.shape[0]} and columns : {self.df.shape[1]}")
            self.col_with_null_values = []
            for i in self.df.columns:
                if self.df[i].isnull().sum() > 0:
                    self.col_with_null_values.append(i)
            logger.warning(f"The Features with Null values : {self.col_with_null_values}")

            # removing last 2 rows
            self.df = self.df.drop([150000,150001],axis=0)
            self.col_with_null_values_ = [i for i in self.df if self.df[i].isnull().sum() > 0]
            logger.warning(f"The Features with Null values : {self.col_with_null_values_}")

            # only MonthlyIncome | MonthlyIncome.1 | NumberOfDependents we have null values
            self.c = 0
            for j in self.df.index:
                if np.isnan(self.df['MonthlyIncome'][j]) == np.isnan(self.df['MonthlyIncome.1'][j]):
                    pass
                elif self.df['MonthlyIncome'][j] == self.df['MonthlyIncome.1'][j]:
                    pass
                else:
                    self.c = self.c + 1

            if self.c == 0:
                logger.warning(f"Both Features has same data : MonthlyIncome : MonthlyIncome.1")
            else:
                logger.warning(f"Both Features are not same  : MonthlyIncome : MonthlyIncome.1")

            self.df = self.df.drop(['MonthlyIncome.1'],axis=1)
            self.X = self.df.iloc[: , :-1] # independent
            self.y = self.df.iloc[: , -1] # dependent
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f"Training data size : {len(self.X_train) , len(self.y_train)}")
            logger.info(f"Testing data size : {len(self.X_test), len(self.y_test)}")

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def missing_data(self):
        try:
            logger.info(f'Dtype of Monthly Income : {self.df["MonthlyIncome"].dtype}')
            logger.info(f'Dtype of NumberofDependents : {self.df["NumberOfDependents"].dtype}')
            cols_with_null_values = ["MonthlyIncome","NumberOfDependents"]
            for i in cols_with_null_values:
                if self.df[i].dtype == np.float64:
                    pass
                elif self.df[i].dtype == object:
                    self.X_train[i] = pd.to_numeric(self.X_train[i])
                    self.X_test[i] = pd.to_numeric(self.X_test[i])
            logger.info(f'Dtype of NumberofDependents : {self.X_train["NumberOfDependents"].dtype}')
            self.X_train,self.X_test = mis_data_ran(self.X_train,self.X_test)
            logger.info(f'Null values in training data : {self.X_train.isnull().sum()}')
            logger.info(f'Null values in training data : {self.X_test.isnull().sum()}')

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def variable_tf(self,train_num,test_num):
        try:
            for k in train_num.columns:
                train_num[k] = np.log(train_num[k]+1)
                test_num[k] = np.log(test_num[k]+1)
            logger.info(f"Converted to Log Level :")
            return train_num,test_num
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def outlier_handling(self,train_num,test_num):
        try:
            for i in train_num.columns:
                upper = train_num[i].quantile(0.95)
                lower = train_num[i].quantile(0.05)
                train_num[i] = np.where(train_num[i] > upper,upper,
                                        np.where(train_num[i] < lower,lower,train_num[i]))
                test_num[i] = np.where(test_num[i] > upper, upper,
                                        np.where(test_num[i] < lower, lower, test_num[i]))
            return train_num,test_num
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def cat_to_num(self):
        try:
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')

            self.X_train_cat = self.X_train.select_dtypes(include ='object')
            self.X_test_cat = self.X_test.select_dtypes(include ='object')

            logger.info(f"categorical columns from the data : {self.X_train_cat.columns}")

            # since we have both nominal features and odinal features
            # for nominal features we are going to apply One_hot_encoding encoding
            # for odinal features we are going to apply Odinal_encoding

            self.X_train_cat,self.X_test_cat=conversion_cat_num(self.X_train_cat,self.X_test_cat)
            self.X_train_cat, self.X_test_cat = cat_num_odinal_enc(self.X_train_cat, self.X_test_cat)
            self.X_train_num,self.X_test_num = self.variable_tf(self.X_train_num,self.X_test_num)
            self.X_train_num,self.X_test_num = self.outlier_handling(self.X_train_num,self.X_test_num)
            logger.info(f"outliers from Training data and Test data handled succesfully")
            logger.info(f"Feature Engineering completed : Know we will combine num and cat columns")
            self.training_ind_data = pd.concat([self.X_train_num,self.X_train_cat],axis=1)
            self.testing_ind_data = pd.concat([self.X_test_num,self.X_test_cat],axis=1)
            logger.info("Concatination Successfull")
            logger.info(f"Training data size : {self.training_ind_data.shape}")
            logger.info(f"Testing data size : {self.testing_ind_data.shape}")
            cols = constant(self.training_ind_data,self.testing_ind_data)
            logger.info(f"Columns with 0.1 variance where removed : {cols}")
            self.training_ind_data = self.training_ind_data.drop(cols,axis=1)
            self.testing_ind_data = self.testing_ind_data.drop(cols, axis=1)
            logger.info(f"After removing unwanted columns : {self.training_ind_data.shape}")
            self.y_train = self.y_train.map({'Good':1 , 'Bad':0}).astype(int)
            self.y_test = self.y_test.map({'Good': 1, 'Bad': 0}).astype(int)
            self.training_ind_data,self.testing_ind_data = hypothesis(self.training_ind_data,self.y_train,self.testing_ind_data,self.y_test)
            logger.info(f"After removing unwanted columns using Hypothesis Testing: {self.training_ind_data.shape}")
            logger.info(f"After removing unwanted columns using Hypothesis Testing : {self.testing_ind_data.shape}")
            return self.training_ind_data,self.testing_ind_data,self.y_train,self.y_test

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")
    def data_balancing(self):
        try:
            self.training_ind_data,self.testing_ind_data,self.y_train,self.y_test = self.cat_to_num()
            logger.info(f"Before Upsampling Technique")

            logger.info(f"Number of rows for 0 -> : {sum(self.y_train == 0)}")
            logger.info(f"Number of rows for 1 -> : {sum(self.y_train == 1)}")

            sm = SMOTE(random_state=2)
            self.training_ind_data_up, self.y_train_up = sm.fit_resample(self.training_ind_data, self.y_train)

            logger.info(f"After Upsampling Technique")

            logger.info(f"Number of rows for 0 -> : {sum(self.y_train_up == 0)}")
            logger.info(f"Number of rows for 1 -> : {sum(self.y_train_up == 1)}")

            return self.training_ind_data_up,self.y_train_up,self.testing_ind_data,self.y_test

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def scaling_data(self):
        try:
            self.training_ind_data_up,self.y_train_up,self.testing_ind_data,self.y_test = self.data_balancing()
            logger.info(f"info : {self.training_ind_data_up.head(3)}")
            sc = StandardScaler()
            sc.fit(self.training_ind_data_up)
            self.scaled_training_inde_cols = sc.transform(self.training_ind_data_up)
            self.scaled_test_inde_cols = sc.transform(self.testing_ind_data)
            logger.info(f"info : {self.scaled_training_inde_cols[:]}")
            multi_models(self.scaled_training_inde_cols,self.y_train_up,self.scaled_test_inde_cols,self.y_test)
            #After training we got to know RF has high Test Accuracy:
            selecting_best_model(self.scaled_training_inde_cols,self.y_train_up,self.scaled_test_inde_cols,self.y_test)
            f_m(self.scaled_training_inde_cols,self.y_train_up,self.scaled_test_inde_cols,self.y_test)
            outcome = testing_()
            logger.info(f"Model prediction : {outcome}")
            logger.info(f'column names : {self.training_ind_data_up.columns}')

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

if __name__ == "__main__":
    try:
        path = 'data/creditcard.csv'
        obj = SERVICE(path) # constructor will be called
        obj.missing_data()
        obj.scaling_data()
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

