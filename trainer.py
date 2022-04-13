import numpy as np
import pandas as pd
from helpers import Data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from DeepRegressionEnsembles import DeepRegressionEnsembles, RidgeRegression
from matplotlib import pyplot as plt

data = Data()

X,y = data.load_peer()

ind = np.array(X.index)
np.random.shuffle(ind)
nb_cval = 3
test_id = np.array_split(ind,nb_cval)

def split_val(df,val_perc=0.1):
    ind_val = np.array(df.index)
    np.random.shuffle(ind_val)
    val_tr = int(np.round(len(ind_val)*val_perc))
    return df.iloc[:val_tr,:], df.iloc[val_tr:,]

res = []
for cval in range(nb_cval):
    #
    X_test = X.loc[test_id[cval],:]
    y_test = y.loc[test_id[cval],:]
    res_cv = y_test.copy()
    X_train_full = X.loc[~X.index.isin(test_id[cval]),:]
    X_val, X_train = split_val(X_train_full)
    y_train_full = y.loc[~y.index.isin(test_id[cval]),:]
    y_val, y_train = split_val(y_train_full)

    ##################
    # predict in sample mean
    ##################
    res_cv['benchmark'] = y_train_full.mean().values[0]

    ##################
    # Linear regression
    ##################
    model = LinearRegression()
    model.fit(X_train_full,y_train_full)
    res_cv['ols_pred']=model.predict(X_test)


    ##################
    # ridge
    ##################
    model = RidgeRegression(input_dim=X_train.shape[1])
    model.train(X_train.values,y_train.values,X_val.values,y_val.values)
    print('Ridge selected best lambda', model.best_lbd)
    res_cv['ridge']=model.predict(X_test.values)

    ##################
    # Random forest
    ##################
    val_rf = {}
    NB_TREE = [10,50,100,500]
    for nb_tree in NB_TREE:
        model = RandomForestRegressor(n_estimators=nb_tree)
        model.fit(X_train,y_train)
        val_rf[nb_tree] = np.mean(np.square(model.predict(X_val)-y_val['y']))
    best_nb_tree = pd.Series(val_rf).idxmin()
    print('We select', best_nb_tree, 'nb of tree')

    model = RandomForestRegressor(n_estimators=best_nb_tree)
    model.fit(X_train_full,y_train_full)
    res_cv['RF']=model.predict(X_test)

    ##################
    # DRE
    ##################
    print('###############  start training DRE')
    model = DeepRegressionEnsembles(k=100,p=100,depth=2,verbose=True,demean_layers=False, normalize_layers=False,act='relu',perf_measure='MSE')
    model.train(X_train.values,y_train.values,X_val.values,y_val.values)
    res_cv['DRE']=model.predict(X_test.values)

    res.append(res_cv)

res = pd.concat(res,axis=0)

perf = {}
for c in res.columns[1:]:
    perf[c]=np.mean(np.square(res['y']-res[c]))
perf=pd.Series(perf)
perf.plot.bar()
plt.grid()
plt.ylabel('MSE')
plt.tight_layout()
plt.show()