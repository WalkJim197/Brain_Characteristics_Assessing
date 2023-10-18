# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:07:16 2023

@author: Conles
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
import seaborn as sns
plt.style.use('ggplot')

def build_model_ridge(x_train,y_train):
    reg_model = linear_model.Ridge(alpha=0.8)#alphas=range(1,100,5)
    reg_model.fit(x_train,y_train)
    return reg_model


# data_predictor = pd.read_excel('E:/ASD/cov.xlsx',sheet_name='Sheet3')
# y = np.array(data_predictor['Age']);
# data = io.loadmat("C:/Users/to139/Desktop/pre.mat");
# X = data['pred_feature']
data_predictor = io.loadmat('G:/cocaine/prediciton/Feature/origin/WM/PUTA/zmaps_roi1/onset.mat');
y = np.array(data_predictor['measure_onset']);
data = io.loadmat('G:/cocaine/prediciton/Feature/origin/WM/PUTA/zmaps_roi1/FC_features_onset.mat');
X = data['finalfeature']

ridge_trains, ridge_RMSEs, ridge_maes, pred, target = [],[],[],[],[]

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
    print('Predict ridge...# {}th'.format(count))
    model_ridge = build_model_ridge(x_train,y_train)
    Con_ridge = model_ridge.score(x_train,y_train)
    pred_ridge = model_ridge.predict(x_test)
    pred_ridge_train = model_ridge.predict(x_train)
    
    pred.append(pred_ridge[0])
    target.append(y_test[0][0][0])
    
    ridge_train = mean_absolute_error(y_train,pred_ridge_train)
    ridge_trains.append(ridge_train)
    ridge_mae = mean_absolute_error(y_test,pred_ridge)
    ridge_maes.append(ridge_mae)
    ridge_MSE = np.square(np.subtract(y_test,pred_ridge)).mean() 
    ridge_RMSE = math.sqrt(ridge_MSE)
    ridge_RMSEs.append(ridge_RMSE)
    
print('train ridge mae:',np.mean(ridge_trains))
print('train ridge R^2',np.mean(Con_ridge))
print('ridge mae:',np.mean(ridge_maes))
print('ridge RMSE',np.mean(ridge_RMSEs))
# r, p = scipy.stats.pearsonr(target, pred)

# print('r',r);print('p',p)
#aresutSOM50=[np.mean(ridge_trains),np.mean(Con_lasso),np.mean(lasso_maes),np.mean(lasso_RMSEs),r,p];
def statistic(x, y):
    return scipy.stats.spearmanr(x, y).correlation
    # return scipy.stats.pearsonr(np.log(x), np.log(abs(y))).statistic
# rng = np.random.default_rng()
res = scipy.stats.permutation_test((target, pred), statistic,
                                   permutation_type = 'pairings',
                                   n_resamples=1000)
rvalue, pvalue = res.statistic, res.pvalue
print('r',rvalue);print('p',pvalue)



plt.figure(figsize=(10,10))
sns.regplot(x=target, y=pred, fit_reg=True, scatter_kws={"s": 100})
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.xlabel('beg_targ')
plt.ylabel('beg_pred')
plt.title('VAN_IVA prediction')
plt.show()
