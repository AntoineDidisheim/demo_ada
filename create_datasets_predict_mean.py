import numpy as np
import pandas as pd
import didipack as didi
from matplotlib import pyplot as plt
from helpers.data import Data
'''
we will try to predict the industry specific level of disagreement among analysts (std monthly prediction across indu)

we will train one model per industry 

'''



data = Data()

##################
# define parameters
##################

predict_std = False
min_nb_analyst = 10

##################
# get the industry
##################
ret = data.load_returns()
# add the industry
ind= ret[['ticker','industry']].drop_duplicates()
# arbitrairly drop the first when a ticker has two industry
ind = ind.loc[~ind['ticker'].duplicated(keep='first'),:]
# check that we don't have duplicated left
ind['ticker'].duplicated(keep='first').sum()

an = data.load_ibes()

an = an.merge(ind)
an['ym'] = an['date'].dt.year*100+an['date'].dt.month

if predict_std:
    an = an.groupby(['ym','industry'])['rec'].aggregate(['std','count']).reset_index().rename(columns={'std':'y'})
else:
    an = an.groupby(['ym','industry'])['rec'].aggregate(['mean','count']).reset_index().rename(columns={'mean':'y'})
an = an.sort_values(['industry','ym']).reset_index(drop=True)
an['y_lag'] = an.groupby('industry')['y'].shift(1)

df = an[['ym','industry','y','y_lag']]
##################
# adding mean return per group
##################
ret['ym'] = ret['date'].dt.year*100+ret['date'].dt.month
ret['tot_mkt'] = ret.groupby(['ym','industry'])['mkt_cap'].transform('sum')
ret['w_ret'] = ret['ret']*ret['mkt_cap']/ret['tot_mkt']
ret=ret.groupby(['ym','industry'])['w_ret'].sum().reset_index()
ret = ret.sort_values(['industry','ym']).reset_index(drop=True)
ret['lag_ret'] = ret.groupby('industry')['w_ret'].shift(1)

df = df.merge(ret)
##################
# add news
##################

news = data.load_news()
news['ym'] = news['date'].dt.year*100+news['date'].dt.month
del news['date']
news = news.groupby('ym').mean().reset_index()
df = df.merge(news)
df.to_pickle(data.pickle_data_dir+'df.p')
