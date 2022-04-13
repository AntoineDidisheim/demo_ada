import numpy as np
import pandas as pd
import didipack as didi
from matplotlib import pyplot as plt
from helpers.data import Data

data = Data()

##################
# returns
##################

ret = data.load_returns()

print('Nb industry', ret['industry'].unique().shape[0])

t=ret.groupby(['date','industry'])['mkt_cap'].sum().reset_index()
t.groupby('industry')['mkt_cap'].mean().sort_values().plot.bar()
plt.xticks([], [])
plt.show()

t.pivot(columns='industry',index='date',values='mkt_cap').fillna(0.0).plot(legend=None)
plt.ylabel('Industry mkt cap')
plt.show()


ret['tot_mkt'] = ret.groupby(['date','industry'])['mkt_cap'].transform('sum')
ret['w_ret'] = ret['ret']*ret['mkt_cap']/ret['tot_mkt']
t=ret.groupby(['date','industry'])['w_ret'].sum().reset_index()
t = t.pivot(columns='industry',index='date',values='w_ret').fillna(0.0)

t.plot(legend=None)
plt.show()

t.cumsum().plot(legend=None)
plt.show()

# create some df for latter
ret_per_ind = t.copy()
ret_per_ind_monthly = t.copy()
ret_per_ind_monthly['ym'] = ret_per_ind_monthly.index.year*100+ret_per_ind_monthly.index.month
ret_per_ind_monthly=ret_per_ind_monthly.groupby('ym').mean()
ret_per_ind_monthly=ret_per_ind_monthly.melt(ignore_index=False).reset_index()

##################
# ananylsts
##################

an = data.load_ibes()

# add the industry
ind= ret[['ticker','industry']].drop_duplicates()
# arbitrairly drop the first when a ticker has two industry
ind = ind.loc[~ind['ticker'].duplicated(keep='first'),:]
# check that we don't have duplicated left
ind['ticker'].duplicated(keep='first').sum()

an = an.merge(ind)
an['ym'] = an['date'].dt.year*100+an['date'].dt.month

mean_rec = an.groupby(['ym','industry'])['rec'].mean().reset_index()
std_rec = an.groupby(['ym','industry'])['rec'].std().reset_index()

# time series trend
mean_rec.pivot(columns='industry',index='ym',values='rec').plot(legend=None)
plt.show()
std_rec.pivot(columns='industry',index='ym',values='rec').plot(legend=None)
plt.show()

# scatter plots
t = mean_rec.merge(ret_per_ind_monthly)
plt.scatter(t['rec'],t['value'],color='k',marker='+')
plt.xlabel('mean rec')
plt.ylabel('industry return')
plt.show()

t = std_rec.merge(ret_per_ind_monthly)
plt.scatter(t['rec'],t['value'],color='k',marker='+')
plt.xlabel('std rec')
plt.ylabel('industry return')
plt.show()

##################
# news trend
##################

news = data.load_news()

# check that it sums to 1
news.sum(1)

# look at some fun trend
list(news.columns)
for c in ['Internet','Elections','Russia', 'UK']:
    plt.plot(news['date'],news[c],label=c)
plt.legend()
plt.grid()
plt.xlabel('date')
plt.ylabel('topic')
plt.show()


