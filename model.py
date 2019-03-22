import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import xgboost as xgb

import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

###########################################
#### making the training & testing set ####
###########################################

prop = pd.read_csv('prop1004.csv', engine='c')
train = pd.read_csv('train_2016_v2.csv', engine='c')

#### dealing the training set
train_df = train.merge(prop, how='left', on='parcelid')
train_df.fillna(train_df.median(), inplace=True)

# drop out ouliers
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.419]

#### create time ft
train_df['transactiondate'] = pd.to_datetime(train_df.transactiondate)
train_df['month'] = train_df.transactiondate.dt.month

x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df['logerror'].values
print(x_train.shape, y_train.shape)

x_test = prop.drop(['parcelid'], axis=1)
# change!!
x_test['month'] = 10

#########################
#### First Pediction ####
#########################

# prepare the empty frame for first step predicting
T = x_test.values
X = x_train.values
y = y_train
kf = KFold(n_splits=6)
kf_list = list(kf.split(X))

S_train = np.zeros((X.shape[0], 8))
S_test = np.zeros((T.shape[0], 8))

#########
## SVR ##
#########

print('SVR......')

scaler = StandardScaler()
scaler.fit(X)
xtrain = scaler.transform(X)
xtest = scaler.transform(T)

S_test_i = np.zeros((T.shape[0], 6))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = xtrain[train_idx]
    y_train = y[train_idx]
    X_holdout = xtrain[test_idx]
    y_holdout = y[test_idx]
    svr = SVR(C=.001, gamma=1)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_holdout)[:]
    S_train[test_idx, 0] = y_pred
    S_test_i[:, j] = svr.predict(xtest)[:]

S_test[:, 0] = S_test_i.mean(1)

del xtrain, xtest

###############################
## GradientBoostingRegressor ##
###############################

print('Gradient Boost......')

params = {'learning_rate': .1,
          'n_estimators': 70,
          'max_depth': 13,
          'min_samples_split':400,
          'min_samples_leaf': 60,
          'subsample': .8}

S_test_i = np.zeros((T.shape[0], 6))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]
    rf = GradientBoostingRegressor(params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_holdout)[:]
    S_train[test_idx, 1] = y_pred
    S_test_i[:, j] = rf.predict(T)[:]
S_test[:, 1] = S_test_i.mean(1)

###############
## extratree ##
###############

print('Extra Tree......')
S_test_i = np.zeros((T.shape[0], 6))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]
    rf = ExtraTreesRegressor(n_estimators=400, max_features=.9)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_holdout)[:]
    S_train[test_idx, 2] = y_pred
    S_test_i[:, j] = rf.predict(T)[:]
S_test[:, 2] = S_test_i.mean(1)

###############
## xgb_tree1 ##
###############

print('First time XGB tree_booster......')

xgb_params = {'eta': 0.033,
              'max_depth': 6,
              'subsample': 0.80,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': 1}

S_test_i = np.zeros((T.shape[0], 6))

for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_holdout, label=y_holdout)
    dtest = xgb.DMatrix(T)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(xgb_params,
                      dtrain,
                      600,
                      watchlist,
                      early_stopping_rounds=10,
                      maximize=False,
                      verbose_eval=10)
    y_pred = model.predict(dvalid)[:]
    S_train[test_idx, 3] = y_pred
    S_test_i[:, j] = model.predict(dtest)[:]
S_test[:, 3] = S_test_i.mean(1)

###############
## xgb_tree2 ##
###############

print('second time XGB tree_booster......')

xgb_params = {'eta': 0.037,
              'max_depth': 5,
              'subsample': 0.80,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'lambda': 0.8,
              'alpha': 0.4,
              'silent': 1}

S_test_i = np.zeros((T.shape[0], 6))

for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_holdout, label=y_holdout)
    dtest = xgb.DMatrix(T)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(xgb_params,
                      dtrain,
                      600,
                      watchlist,
                      early_stopping_rounds=10,
                      maximize=False,
                      verbose_eval=10)
    y_pred = model.predict(dvalid)[:]
    S_train[test_idx, 4] = y_pred
    S_test_i[:, j] = model.predict(dtest)[:]

S_test[:, 4] = S_test_i.mean(1)

################
## xgb_linear ##
################

print('XGB linear_booster......')
xgb_pars = {'n_estimators': 598.0,
            'alpha': 1.0,
            'lambda': 3.3,
            'eta': 0.475,
            'lambda_bias': 0.6,
            'nthread': 4,
            'booster' : 'gblinear',
            'silent': 1,
            'eval_metric': 'mae',
            'objective': 'reg:linear'}

S_test_i = np.zeros((T.shape[0], 6))

for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_holdout, label=y_holdout)
    dtest = xgb.DMatrix(T)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(xgb_pars, dtrain, 600, watchlist, early_stopping_rounds=10,
                      maximize=False, verbose_eval=10)
    y_pred = model.predict(dvalid)[:]
    S_train[test_idx, 5] = y_pred
    S_test_i[:, j] = model.predict(dtest)[:]
S_test[:, 5] = S_test_i.mean(1)

###################
## random forest ##
###################

S_test_i = np.zeros((T.shape[0], 6))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]
    rf = RandomForestRegressor(max_features= .15,
                               random_state=100,
                               n_estimators= 480,
                               n_jobs= 1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_holdout)[:]
    S_train[test_idx, 6] = y_pred
    S_test_i[:, j] = rf.predict(T)[:]
S_test[:, 6] = S_test_i.mean(1)

#########
## lgb ##
#########

params = {'max_bin': 10,
          'learning_rate': .0021,
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'mae',
          'sub_feature': .345,
          'bagging_fraction': .85,
          'bagging_freq': 40,
          'num_leaves': 512,
          'min_data': 500,
          'min_hessian': .05,
          'verbose': 0,
          'feature_fraction_seed': 2,
          'bagging_seed': 3
          }

S_test_i = np.zeros((T.shape[0], 6))

for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]
    d_train = lgb.Dataset(X_train, label=y_train)
    clf = lgb.train(params, d_train, 430)
    y_pred = clf.predict(X_holdout)[:]
    S_train[test_idx, 7] = y_pred
    S_test_i[:, j] = clf.predict(T)[:]
S_test[:, 7] = S_test_i.mean(1)

####################
## Neural Network ##
####################

imputer= Imputer()
imputer.fit(X)
XX = imputer.transform(X)
imputer.fit(T)
TT = imputer.transform(x_test)

sc = StandardScaler()
XX = sc.fit_transform(XX)
TT = sc.transform(TT)
len_x=int(XX.shape[1])

S_test_i = np.zeros((T.shape[0], 6))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = XX[train_idx]
    y_train = y[train_idx]
    X_holdout = XX[test_idx]
    y_holdout = y[test_idx]

    # Neural Network
    nn = Sequential()
    nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
    nn.add(PReLU())
    nn.add(Dropout(.4))
    nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.6))
    nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.5))
    nn.add(Dense(units = 26, kernel_initializer = 'normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.6))
    nn.add(Dense(1, kernel_initializer='normal'))
    nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

    nn.fit(np.array(X_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)
    y_pred = nn.predict(X_holdout)[:]
    nn_pred = y_pred.flatten()
    S_train[test_idx, 8] = nn_pred
    tt_pred = nn.predict(TT)[:]
    nntt_pred = tt_pred.flatten()
    S_test_i[:, j] = nntt_pred

S_test[:,8] = S_test_i.mean(1)

print(pd.DataFrame(S_test).head())

#################################
#### model ensemble with XGB ####
#################################

print('Second step predicting......')
num_boost_rounds = 500

dtrain = xgb.DMatrix(S_train, y)
dtest = xgb.DMatrix(S_test)

xgb_params = {  # best as of 2017-09-28 13:20 UTC
    'eta': 0.007,
    'max_depth': 7,
    'subsample': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 5.0,
    'alpha': 0.65,
    'colsample_bytree': 0.5,
    'silent': 1
}

model = xgb.train(xgb_params,
                  dtrain,
                  num_boost_round=2000,
                  evals=evals,
                  early_stopping_rounds=10,
                  verbose_eval=50)

xgb_pred = model.predict(dtest)
print(pd.DataFrame(xgb_pred).head())

## gathering the result
sub = prop[['parcelid']]
sub.rename(columns = {'parcelid':'ParcelId'}, inplace = True)
sub['201610'] = xgb_pred
sub['201611'] = xgb_pred
sub['201612'] = xgb_pred
sub['201710'] = xgb_pred
sub['201711'] = xgb_pred
sub['201712'] = xgb_pred
len(sub)
sub.to_csv('sub1003.csv', index=False)
