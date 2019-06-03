import numpy as np

import pandas as pd

from sklearn.preprocessing import RobustScaler


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBRegressor




from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


df = pd.read_csv('final_dataset/train.csv')

#Separazione delle colonne secondo categorie
ID_col = ['Unnamed: 0',"Unnamed: 0.1", "STATION_ID","STATION_ID_2","STATION_ID_3","STATION_ID_4", 'KEY', 'KEY_2_x', 'KEY_2_y']
target_col = ["SPEED_AVG", 'SPEED_MAX','SPEED_MIN','SPEED_SD']
cat_col = ["EVENT_TYPE","WEATHER", 'TIME_INTERVAL']
time_col = ['DATETIME_UTC','START_DATETIME_UTC','END_DATETIME_UTC', "DATETIME_UTC_WEATHER"]
other_col = ['KM_START', 'KM_END']
num_col= list(set(list(df.columns))-set(ID_col)-set(target_col)-set(time_col)-set(cat_col))

#caricare il dataset riducendo le colonne numeriche da 64 a 32 bit
df = pd.read_csv('final_dataset/train.csv', dtype={col: np.float32 for col in num_col})
test = pd.read_csv('final_dataset/validation.csv', dtype={col: np.float32 for col in num_col})


#Soluzione + semplice: droppare tutti i missing values
df.dropna(inplace = True)


#one-hot encoding delle var categoriche
catOneHot_col = []
for i in cat_col:
    one_hot = pd.get_dummies(df[i])
    one_hot2 = pd.get_dummies(test[i])
    df = df.drop(i,axis = 1)
    test = test.drop(i,axis = 1)
    catOneHot_col.extend(one_hot.columns)
    df = df.join(one_hot)
    test = test.join(one_hot2)
df.columns


features = list(num_col) + list(catOneHot_col)

# prova con xgboost e crossvalidation
x_train = df[list(features)].values
y_train = df["SPEED_AVG"].values

#si prendono le features e la target variable dal dataset di test
x_test = test[list(features)].values
y_test = test["SPEED_AVG"].values


#scaling
scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



gb = XGBRegressor(
    learning_rate=0.1,
    n_estimators=2000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:gamma',
    nthread=8,
    scale_pos_weight=1,
    seed=27)

# Nel caso si voglia usare la Cross Validation
# xgb_param = gb.get_xgb_params()
# xgtrain = xgb.DMatrix(df[features].values, label=df['SPEED_AVG'].values)
# cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=gb.get_params()['n_estimators'], nfold=10, metrics='mae', early_stopping_rounds=50)
# gb.set_params(n_estimators=cvresult.shape[0])


gb.fit(x_train, y_train, eval_metric='mae')


#funzione per visualizzare il MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100





#visualizza punteggi sul Training set, test set + media della Cross validation
predictions = gb.predict(x_train)
predictionsT = gb.predict(x_test)

print ("MAE Score (Train): %f" % mean_absolute_error(y_train,predictions))
print ('MAE Score (Test): %f' % mean_absolute_error(y_test, predictionsT))
#print("MAE Score (CV): %f" % cvresult['test-mae-mean'].tail(1))

print ("MAPE Score (Train): %f" % mean_absolute_percentage_error(y_train,predictions))
print ("MAPE Score (Test): %f" % mean_absolute_percentage_error(y_test,predictionsT))

print ("R2 Score (Train): %f" % r2_score(y_train,predictions))
print ("R2 Score (Test): %f" % r2_score(y_test,predictionsT))