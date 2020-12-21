# Classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from sklearn.gaussian_process.kernels import RBF

from train_test import train_models, save_models
from data_preparation import read_data

# ML models
models = []
models.append(('LR', LogisticRegression(C=0.1, tol=0.1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('DT', DecisionTreeClassifier(min_samples_leaf=4, min_samples_split=13,splitter='best')))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(C=10, gamma='scale', kernel='rbf',probability=True)))
models.append(('ADB', AdaBoostClassifier(n_estimators=20)))
models.append(('RF', RandomForestClassifier(criterion='gini', max_depth=6, min_samples_leaf=1, min_samples_split=2,
                       n_estimators=100)))
models.append(('GPC', GaussianProcessClassifier(kernel=RBF(length_scale=1))))
models.append(('XGB', XGBClassifier(booster='gbtree', colsample_bylevel=1,
                    learning_rate=0.001, max_depth=6,
                    min_child_weight=5,n_estimators=700,
                    objective='binary:logistic')))

path = "E:/project/models/"
data_path = "E:/project/training_data.csv"

data,labels = read_data(data_path)
train_models(models, data, labels)
save_models(models, path)
