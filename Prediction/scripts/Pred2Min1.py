# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:59:27 2022

@author: JIM
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_absolute_error
import scipy.io as io 
import math
import scipy.stats

def build_model_lasso(x_train,y_train):
    reg_model = linear_model.LassoCV(cv = 10,n_jobs = -1)
    reg_model.fit(x_train,y_train)
    return reg_model
    
data_label = pd.read_excel('E:/ASD/PostPrep/variables/SHENGMIN/ic_edit.xlsx',sheet_name='GIG')
 
# feature_label = ['Age','FIQ','VIQ','PIQ']
feature_label = ['FIQ']
for flabel in feature_label:
    # for ic in range(0,len(data_label['IC'])):
    for ic in range(14,15):
        data = io.loadmat(''.join(['E:/ASD/PostPrep/prediction/Newfeatures/pred_features/GIG_IC/',flabel,'/',flabel,'_'
                                    ,data_label['IC'][ic],'.mat']));
        X = data['pred_feature']
        print('{} is going...'.format(data_label['IC'][ic]))
        data_predictor = pd.read_excel('E:/ASD/cov.xlsx',sheet_name='Sheet3')
        y = np.array(data_predictor[flabel]);
        lasso_trains, lasso_RMSEs, lasso_maes, pred, target = [],[],[],[],[]
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        count = 0 
        # kf = KFold(n_splits=5)
        # kf.get_n_splits(X)
        ## 5折交叉验证方式
        for train_ind, test_ind in loo.split(X):
        # for train_ind, test_ind in kf.split(X):  
            x_train, y_train, x_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
          
            count += 1
            print('Predict Lasso...# {}th'.format(count))
            model_lasso = build_model_lasso(x_train,y_train)
            Con_lasso = model_lasso.score(x_train,y_train)
            pred_lasso = model_lasso.predict(x_test)
            pred_lasso_train = model_lasso.predict(x_train)
            
            pred.append(pred_lasso[0])
            target.append(y_test[0])
            
            lasso_train = mean_absolute_error(y_train,pred_lasso_train)
            lasso_trains.append(lasso_train)
            lasso_mae = mean_absolute_error(y_test,pred_lasso)
            lasso_maes.append(lasso_mae)
            lasso_MSE = np.square(np.subtract(y_test,pred_lasso)).mean() 
            lasso_RMSE = math.sqrt(lasso_MSE)
            lasso_RMSEs.append(lasso_RMSE)
            
        
        print('train Lasso mae:',np.mean(lasso_trains))
        print('train Lasso R^2',np.mean(Con_lasso))
        print('Lasso mae:',np.mean(lasso_maes))
        print('Lasso RMSE',np.mean(lasso_RMSEs))
        # r, p = scipy.stats.pearsonr(target, pred)
        # r, p = scipy.stats.spearmanr(target, pred)
        # print('r',r);print('p',p)
        # plt.hist(pred)
        def statistic(x, y):
            return scipy.stats.spearmanr(x, y).correlation
            # return scipy.stats.pearsonr(np.log(x), np.log(abs(y))).statistic
        # rng = np.random.default_rng()
        res = scipy.stats.permutation_test((target, pred), statistic,
                                           permutation_type = 'pairings',
                                           n_resamples=1000)
        rvalue, pvalue = res.statistic, res.pvalue
        print('r',rvalue);print('p',pvalue)
        data_label.loc[ic,(' '.join([flabel,'train Lasso mae']),
                           ' '.join([flabel,'train Lasso R^2']),
                           ' '.join([flabel,'Lasso mae']),
                           ' '.join([flabel,'Lasso RMSE']),
                           ' '.join([flabel,'r']),
                           ' '.join([flabel,'p']),
                          )] = [np.mean(lasso_trains),np.mean(Con_lasso),
                                        np.mean(lasso_maes),np.mean(lasso_RMSEs),rvalue,pvalue]
# data_label.to_excel('E:/ASD/PostPrep/variables/SHENGMIN/ic_results11.xlsx',sheet_name='GIGresults')

import seaborn as sns

plt.figure(figsize=(10,10))
sns.regplot(x=target, y=pred, fit_reg=True, scatter_kws={"s": 100})
plt.xlabel('FIQ_targ')
plt.ylabel('FIQ_pred')
plt.title('AUD_IVA prediction')
plt.show()