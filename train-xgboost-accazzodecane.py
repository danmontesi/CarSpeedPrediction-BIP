import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.ensemble import ExtraTreesRegressor

import xgboost as xgb
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBRegressor



from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv('final_dataset/train.csv')

ID_col = ['Unnamed: 0',"Unnamed: 0.1", "STATION_ID","STATION_ID_2","STATION_ID_3","STATION_ID_4", 'KEY', 'KEY_2_x', 'KEY_2_y']
target_col = ["SPEED_AVG", 'SPEED_MAX','SPEED_MIN','SPEED_SD']
cat_col = ["EVENT_TYPE","WEATHER", 'TIME_INTERVAL']
time_col = ['DATETIME_UTC','START_DATETIME_UTC','END_DATETIME_UTC', "DATETIME_UTC_WEATHER"]
other_col = ['KM_START', 'KM_END']
num_col= list(set(list(df.columns))-set(ID_col)-set(target_col)-set(time_col)-set(cat_col))

df = pd.read_csv('final_dataset/train.csv', dtype={col: np.float32 for col in num_col})

df.dropna(inplace = True)



#ONE HOT ENCODING
catOneHot_col = []
for i in cat_col:
    one_hot = pd.get_dummies(df[i])
    df = df.drop(i,axis = 1)
    catOneHot_col.extend(one_hot.columns)
    df = df.join(one_hot)
df.columns


#RANDOM FOREST
features = list(num_col) + list(catOneHot_col)

X = df[features]
y = df['SPEED_AVG']
forest = ExtraTreesRegressor(n_estimators=750, random_state=0, bootstrap=True,criterion='mae', oob_score = True)
forest.fit(X, y)

forest.oob_score_

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(15):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")

plt.xlim([-1, X.shape[1]])
plt.show()




#XGBOOOOOOOOST
x_train = df[list(features)].values
y_train = df["SPEED_AVG"].values

gb = XGBRegressor(
    learning_rate=0.1,
    n_estimators=750,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:gamma',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

xgb_param = gb.get_xgb_params()
xgtrain = xgb.DMatrix(df[features].values, label=df['SPEED_AVG'].values)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=gb.get_params()['n_estimators'], nfold=10, metrics='mae',
                  early_stopping_rounds=50)
gb.set_params(n_estimators=cvresult.shape[0])

gb.fit(x_train, y_train, eval_metric='mae')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

predictions = gb.predict(x_train)

print ("MAE Score (Train): %f" % mean_absolute_error(y_train,predictions))

print("MAE Score (Test): %f" %cvresult['test-mae-mean'].tail(1))

print ("MAPE Score (Train): %f" % mean_absolute_percentage_error(y_train,predictions))
print ("R2 Score (Train): %f" % r2_score(y_train,predictions))

test = pd.read_csv('test.csv', dtype={col: np.float32 for col in num_col})
catOneHot_col = []
for i in cat_col:
    one_hot = test.get_dummies(test[i])
    test = test.drop(i,axis = 1)
    catOneHot_col.extend(one_hot.columns)
    test = test.join(one_hot)
#df.columns

#snippet per joinare la colonna dei risultati al dataset
df = df.join(predictions)


#snippet per generare il csv finale
cols=['','', '']


output = test[cols]
output = output.groupby('').sum()
output.to_csv('RS.csv')