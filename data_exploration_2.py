import numpy as np
import pandas as pd
from helpers import Data
from matplotlib import pyplot as plt
'''
Second idea, pear pressure among the Analysts
'''

##################
# the impact of a recommendation
##################

data = Data()

df = data.load_ibes()
df['ym'] = df['date'].dt.year*100+df['date'].dt.month
ret  =data.load_returns()
ret.index = ret['date']
t = ret.groupby('ticker')['ret'].rolling(12).mean().reset_index()
t['ret_future'] = t.groupby('ticker')['ret'].shift(-12)
t = t.dropna()
t['ym'] = t['date'].dt.year*100+t['date'].dt.month
del t['ret']

# merge with monthly return
ret = data.load_returns()
ret['ym'] = ret['date'].dt.year*100+ret['date'].dt.month
df=df.merge(t).merge(ret[['ym','ticker','ret']])
t = df.groupby('rec')[['ret_future','ret']].mean()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t.index, t['ret_future'],color='g', label='Ret long term')
ax2.plot(t.index, t['ret'],color='b', label='Ret short term')
ax1.set_xlabel('X data')
ax1.set_ylabel('Long term ret', color='g')
ax2.set_ylabel('Short term ret', color='b')
plt.tight_layout()
plt.show()

##################
# building features
##################
min_nb_analyst = 10
t=df.groupby('ticker')['analyst'].nunique().sort_values()
print(f'we focus on the {(t>min_nb_analyst).sum()} firms with more than {min_nb_analyst} ananylsts')
ticker_to_keep = list(t[t>min_nb_analyst].index)
df = df.loc[df['ticker'].isin(ticker_to_keep),:]

f = []
for ticker in ticker_to_keep:
    t = df.loc[df['ticker']==ticker,:].groupby(['date','analyst'])['rec'].mean().reset_index()
    t=t.sort_values('date').pivot(columns='analyst',values='rec',index='date')
    features = (~pd.isna(t)).sum(1).reset_index().rename(columns={0:'nb_actif'})
    features['perc_actif'] = (~pd.isna(t)).mean(1).values
    t = t.fillna(method='ffill')
    features['mean_pred'] = t.mean(1).values
    features['var_pred'] = t.std(1).values
    for l in range(10):
        features[f'd_mean_pred_{l+1}'] = (t.mean(1)-t.mean(1).shift(l+1)).values

    features = features.fillna(0.0)
    features['ticker'] = ticker
    f.append(features)
features = pd.concat(f,axis=0)

df[['date','analyst','rec','ret']].groupby()



