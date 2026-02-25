import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from log_file import phase_1
logger = phase_1("models")
def knn(X_train,y_train,X_test,y_test):
    try:
        # value_acc = []
        # k_values = np.arange(3,49,2)
        # for i in k_values:
        #     knn_reg = KNeighborsClassifier(n_neighbors=i)
        #     knn_reg.fit(X_train,y_train)
        #     value_acc.append(accuracy_score(y_test,knn_reg.predict(X_test)))
        # logger.info(f"All k_value accuracy : {value_acc}")
        # logger.info(f"Best k_value : {k_values[value_acc.index(max(value_acc))]} with Acc : {max(value_acc)}")
        knn_reg = KNeighborsClassifier(n_neighbors=3)
        knn_reg.fit(X_train, y_train)
        logger.info(f"KNN Test Accuracy : {accuracy_score(y_test,knn_reg.predict(X_test))}")
        logger.info(f"KNN Confusion Matrix : {confusion_matrix(y_test,knn_reg.predict(X_test))}")
        logger.info(f"KNN classification Report : {classification_report(y_test,knn_reg.predict(X_test))}")
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


def NB(X_train,y_train,X_test,y_test):
    try:
        NB_reg = GaussianNB()
        NB_reg.fit(X_train, y_train)
        logger.info(f"NB Test Accuracy : {accuracy_score(y_test, NB_reg.predict(X_test))}")
        logger.info(f"NB Confusion Matrix : {confusion_matrix(y_test, NB_reg.predict(X_test))}")
        logger.info(f"NB classification Report : {classification_report(y_test, NB_reg.predict(X_test))}")
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def LR(X_train, y_train, X_test, y_test):
    try:
        LR_reg = LogisticRegression()
        LR_reg.fit(X_train, y_train)
        logger.info(f"LR Test Accuracy : {accuracy_score(y_test, LR_reg.predict(X_test))}")
        logger.info(f"LR Confusion Matrix : {confusion_matrix(y_test, LR_reg.predict(X_test))}")
        logger.info(f"LR classification Report : {classification_report(y_test, LR_reg.predict(X_test))}")
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def DT(X_train,y_train,X_test,y_test):
    try:
        DT_reg = DecisionTreeClassifier(criterion='entropy')
        DT_reg.fit(X_train, y_train)
        logger.info(f"DT Test Accuracy : {accuracy_score(y_test, DT_reg.predict(X_test))}")
        logger.info(f"DT Confusion Matrix : {confusion_matrix(y_test, DT_reg.predict(X_test))}")
        logger.info(f"DT classification Report : {classification_report(y_test, DT_reg.predict(X_test))}")
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def RF(X_train, y_train, X_test, y_test):
    try:
        # value_acc = []
        # trees = np.random.randint(1,100,10)
        # for j in trees:
        #     RF_reg = RandomForestClassifier(criterion='entropy',n_estimators=j)
        #     RF_reg.fit(X_train, y_train)
        #     value_acc.append(accuracy_score(y_test,RF_reg.predict(X_test)))
        # logger.info(f"All Tree Value accuracy : {value_acc}")
        # logger.info(f"Best Tree : {trees[value_acc.index(max(value_acc))]} with Acc : {max(value_acc)}")
        RF_reg = RandomForestClassifier(criterion='entropy', n_estimators=99)
        RF_reg.fit(X_train, y_train)
        logger.info(f"RF Test Accuracy : {accuracy_score(y_test, RF_reg.predict(X_test))}")
        logger.info(f"RF Confusion Matrix : {confusion_matrix(y_test, RF_reg.predict(X_test))}")
        logger.info(f"RF classification Report : {classification_report(y_test, RF_reg.predict(X_test))}")
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


def multi_models(X_train,y_train,X_test,y_test):
    try:
        logger.info(f"-----KNN----------")
        knn(X_train,y_train,X_test,y_test)
        logger.info(f"-----Naive Bayes----------")
        NB(X_train,y_train,X_test,y_test)
        logger.info(f"-----Logistic Regression----------")
        LR(X_train, y_train, X_test, y_test)
        logger.info(f"-----Decision Tree----------")
        DT(X_train, y_train, X_test, y_test)
        logger.info(f"-----Random Forest----------")
        RF(X_train, y_train, X_test, y_test)
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")