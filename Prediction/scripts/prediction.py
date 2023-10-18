# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:41:39 2022

@author: Jim
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
#matplotlib inline

import itertools
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
# from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, KFold
# from mlxtend.plotting import plot_learning_curves
# from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io as io 
import math

def build_model_BN(x_train,y_train):
    reg_model = GaussianNB()
    reg_model.fit(x_train,y_train)
    return reg_model

def build_model_SVR(x_train,y_train):
    estimator = SVR(kernel='linear')
    param_grid = {
        'C': np.linspace(1,15,15),
    }
    svr_model = GridSearchCV(estimator, param_grid)
    svr_model.fit(x_train, y_train)
    print(svr_model.best_params_)
    return svr_model

def build_model_lr(x_train,y_train):
    reg_model = linear_model.LinearRegression()
    reg_model.fit(x_train,y_train)
    return reg_model

def build_model_ridge(x_train,y_train):
    reg_model = linear_model.Ridge(alpha=0.8)#alphas=range(1,100,5)
    reg_model.fit(x_train,y_train)
    return reg_model

def build_model_lasso(x_train,y_train):
    reg_model = linear_model.LassoCV(cv = 10,n_jobs=-1)
    reg_model.fit(x_train,y_train)
    return reg_model

def build_model_gbdt(x_train,y_train):
    estimator =GradientBoostingRegressor(loss='ls',subsample= 0.85,max_depth= 5,n_estimators = 100)
    param_grid = { 
            'learning_rate': [0.05,0.08,0.1,0.2],
            }
    gbdt = GridSearchCV(estimator, param_grid,cv=5)
    gbdt.fit(x_train,y_train)
    print(gbdt.best_params_)
    # print(gbdt.best_estimator_ )
    return gbdt

def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=120, learning_rate=0.08, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=5) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=63,n_estimators = 100)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    print(gbm.best_params_)
    return gbm

def Weighted_method(test_pre1,test_pre2,test_pre3,test_pre4,w=[1/4,1/4,1/4,1/4]):
    Weighted_result = w[0]*pd.Series(test_pre1)+w[1]*pd.Series(test_pre2)+w[2]*pd.Series(test_pre3)+w[3]*pd.Series(test_pre4)
    return Weighted_result


#data_predictor = pd.read_excel('E:/ASD/cov.xlsx',sheet_name='Sheet3')
data_predictor = io.loadmat('G:/cocaine/prediciton/Feature/origin/GM/ACC/zmaps_roi3/begin.mat');
y = np.array(data_predictor['measure_begin']);
data = io.loadmat('G:/cocaine/prediciton/Feature/origin/GM/ACC/zmaps_roi3/FC_features_beg.mat');
X = data['finalfeature']
# pc = PCA(n_components=0.95)
# X = pc.fit_transform(x)
svr_trains, lr_trains, ridge_trains, lasso_trains, gbdt_trains, xgb_trains, lgb_trains = [],[],[],[],[],[],[]
svr_RMSEs, lr_RMSEs, ridge_RMSEs, lasso_RMSEs, gbdt_RMSEs, xgb_RMSEs, lgb_RMSEs = [],[],[],[],[],[],[]
svr_maes, lr_maes, ridge_maes, lasso_maes, gbdt_maes, xgb_maes, lgb_maes = [],[],[],[],[],[],[]
#fusion_maes,fusion_RMSEs = [],[]

loo = LeaveOneOut()
loo.get_n_splits(X)
count = 0 
# kf = KFold(n_splits=5)
# kf.get_n_splits(X)
## 5折交叉验证方式
# sk=StratifiedKFold(n_splits=5,shuffle=False)  
for train_ind, test_ind in loo.split(X):
# for train_ind, test_ind in kf.split(X):
# for train_ind,test_ind in sk.split(X,y):
    
    x_train, y_train, x_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
  
    # x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

    # Train and Predict
    print('Predict SVR...')
    model_SVR = build_model_SVR(x_train,y_train)
    Con_SVR = model_SVR.score(x_train, y_train)
    pred_SVR = model_SVR.predict(x_test)
    pred_SVR_train = model_SVR.predict(x_train)
    
    print('Predict LR...')
    model_lr = build_model_lr(x_train,y_train)
    Con_lr = model_lr.score(x_train,y_train)
    pred_lr = model_lr.predict(x_test)
    pred_lr_train = model_lr.predict(x_train)

    print('Predict Ridge...')
    model_ridge = build_model_ridge(x_train,y_train)
    Con_ridge = model_ridge.score(x_train,y_train)
    pred_ridge = model_ridge.predict(x_test)
    pred_ridge_train = model_ridge.predict(x_train)
    count += 1
    print('Predict Lasso...# {}th'.format(count))
    model_lasso = build_model_lasso(x_train,y_train)
    Con_lasso = model_lasso.score(x_train,y_train)
    pred_lasso = model_lasso.predict(x_test)
    pred_lasso_train = model_lasso.predict(x_train)

    print('Predict GBDT...')
    model_gbdt = build_model_gbdt(x_train,y_train)
    Con_gbdt = model_gbdt.score(x_train,y_train)
    pred_gbdt = model_gbdt.predict(x_test)
    pred_gbdt_train = model_gbdt.predict(x_train)

    print('predict XGB...')
    model_xgb = build_model_xgb(x_train,y_train)
    Con_xgb = model_xgb.score(x_train,y_train)
    pred_xgb = model_xgb.predict(x_test)
    pred_xgb_train = model_xgb.predict(x_train)

    print('predict lgb...')
    model_lgb = build_model_lgb(x_train,y_train)
    Con_lgb = model_lgb.score(x_train,y_train)
    pred_lgb = model_lgb.predict(x_test)
    pred_lgb_train = model_lgb.predict(x_train)
    
    svr_train = mean_absolute_error(y_train,pred_SVR_train)
    svr_trains.append(svr_train)
    svr_mae = mean_absolute_error(y_test,pred_SVR)
    svr_maes.append(svr_mae)
    svr_MSE = np.square(np.subtract(y_test,pred_SVR)).mean() 
    svr_RMSE = math.sqrt(svr_MSE)
    svr_RMSEs.append(svr_RMSE)
    
    lr_train = mean_absolute_error(y_train,pred_lr_train)
    lr_trains.append(lr_train)
    lr_mae = mean_absolute_error(y_test,pred_lr)
    lr_maes.append(lr_mae)
    lr_MSE = np.square(np.subtract(y_test,pred_lr)).mean() 
    lr_RMSE = math.sqrt(lr_MSE)
    lr_RMSEs.append(lr_RMSE)
    
    ridge_train = mean_absolute_error(y_train,pred_ridge_train)
    ridge_trains.append(ridge_train)
    ridge_mae = mean_absolute_error(y_test,pred_ridge)
    ridge_maes.append(ridge_mae)
    ridge_MSE = np.square(np.subtract(y_test,pred_ridge)).mean() 
    ridge_RMSE = math.sqrt(ridge_MSE)
    ridge_RMSEs.append(ridge_RMSE)
    
    lasso_train = mean_absolute_error(y_train,pred_lasso_train)
    lasso_trains.append(lasso_train)
    lasso_mae = mean_absolute_error(y_test,pred_lasso)
    lasso_maes.append(lasso_mae)
    lasso_MSE = np.square(np.subtract(y_test,pred_lasso)).mean() 
    lasso_RMSE = math.sqrt(lasso_MSE)
    lasso_RMSEs.append(lasso_RMSE)
    
    gbdt_train = mean_absolute_error(y_train,pred_gbdt_train)
    gbdt_trains.append(gbdt_train)
    gbdt_mae = mean_absolute_error(y_test,pred_gbdt)
    gbdt_maes.append(gbdt_mae)
    gbdt_MSE = np.square(np.subtract(y_test,pred_gbdt)).mean() 
    gbdt_RMSE = math.sqrt(gbdt_MSE)
    gbdt_RMSEs.append(gbdt_RMSE)
    
    xgb_train = mean_absolute_error(y_train,pred_xgb_train)
    xgb_trains.append(xgb_train)
    xgb_mae = mean_absolute_error(y_test,pred_xgb)
    xgb_maes.append(xgb_mae)
    xgb_MSE = np.square(np.subtract(y_test,pred_xgb)).mean() 
    xgb_RMSE = math.sqrt(xgb_MSE)
    xgb_RMSEs.append(xgb_RMSE)
    
    lgb_train = mean_absolute_error(y_train,pred_lgb_train)
    lgb_trains.append(lgb_train)
    lgb_mae = mean_absolute_error(y_test,pred_lgb)
    lgb_maes.append(lgb_mae)
    lgb_MSE = np.square(np.subtract(y_test,pred_lgb)).mean() 
    lgb_RMSE = math.sqrt(lgb_MSE)
    lgb_RMSEs.append(lgb_RMSE)
    
    # w = [0.4,0.2,0.2,0.2]
    # fusion_pred = Weighted_method(pred_lasso,pred_gbdt,pred_xgb,pred_lgb,w)
    # fusion_mae = mean_absolute_error(y_test,fusion_pred)
    # fusion_maes.append(fusion_mae)
    # fusion_MSE = np.square(np.subtract(y_test,fusion_pred)).mean() 
    # fusion_RMSE = math.sqrt(fusion_MSE)
    # fusion_RMSEs.append(fusion_RMSE)
    
print('train SVR mae:',np.mean(svr_trains))
print('train SVR R^2',np.mean(Con_SVR))
print('SVR mae:',np.mean(svr_maes))
print('SVR RMSE',np.mean(svr_RMSEs))

print('train LR mae:',np.mean(lr_trains))
print('train LR R^2',np.mean(Con_lr))
print('LR mae:',np.mean(lr_maes))
print('LR RMSE',np.mean(lr_RMSEs))

print('train Ridge mae:',np.mean(ridge_trains))
print('train Ridge R^2',np.mean(Con_ridge))
print('Ridge mae:',np.mean(ridge_maes))
print('Ridge RMSE',np.mean(ridge_RMSEs))

print('train Lasso mae:',np.mean(lasso_trains))
print('train Lasso R^2',np.mean(Con_lasso))
print('Lasso mae:',np.mean(lasso_maes))
print('Lasso RMSE',np.mean(lasso_RMSEs))

print('train GBDT mae:',np.mean(gbdt_trains))
print('train GBDT R^2',np.mean(Con_gbdt))
print('GBDT mae:',np.mean(gbdt_maes))
print('GBDT RMSE',np.mean(gbdt_RMSEs))

print('train XGB mae:',np.mean(xgb_trains))
print('train XGB R^2',np.mean(Con_xgb))
print('XGB mae:',np.mean(xgb_maes))
print('XGB RMSE',np.mean(xgb_RMSEs))

print('train lgb mae:',np.mean(lgb_trains))
print('train lgb R^2',np.mean(Con_lgb))
print('lgb mae:',np.mean(lgb_maes))
print('lgb RMSE',np.mean(lgb_RMSEs))


# print('fusion mae:',np.mean(fusion_maes))
# print('fusion RMSE',np.mean(fusion_RMSEs))
