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
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
logger = phase_1("best_model")

def selecting_best_model(final_X_train,final_y_train,final_X_test,final_y_test):
    try:
        # knn
        knn_reg = KNeighborsClassifier(n_neighbors=3)
        knn_reg.fit(final_X_train, final_y_train)
        knn_pred = knn_reg.predict(final_X_test)

        # naive bayes
        naive_reg = GaussianNB()
        naive_reg.fit(final_X_train, final_y_train)
        naive_pred = naive_reg.predict(final_X_test)

        # logistic regression
        log_reg = LogisticRegression()
        log_reg.fit(final_X_train, final_y_train)
        log_pred = log_reg.predict(final_X_test)

        # decision_tree
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(final_X_train, final_y_train)
        dt_pred = dt_reg.predict(final_X_test)

        # random_forest
        rf_reg = RandomForestClassifier(criterion='entropy', n_estimators=99)
        rf_reg.fit(final_X_train, final_y_train)
        rf_pred = rf_reg.predict(final_X_test)

        fpr_knn, tpr_knn, threshold = roc_curve(final_y_test, knn_pred)
        fpr_lr, tpr_lr, threshold = roc_curve(final_y_test, log_pred)
        fpr_nb, tpr_nb, threshold = roc_curve(final_y_test, naive_pred)
        fpr_dt, tpr_dt, threshold = roc_curve(final_y_test, dt_pred)
        fpr_rf, tpr_rf, threshold = roc_curve(final_y_test, rf_pred)

        plt.figure(figsize=(5, 2))
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr_knn, tpr_knn, color='r', label='KNN')
        plt.plot(fpr_lr, tpr_lr, color='b', label='LR')
        plt.plot(fpr_nb, tpr_nb, color='y', label='NB')
        plt.plot(fpr_dt, tpr_dt, color='g', label='DT')
        plt.plot(fpr_rf, tpr_rf, color='black', label='RF')

        plt.legend(loc=0)
        plt.show()
    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")