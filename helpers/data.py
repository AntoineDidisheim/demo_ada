import numpy as np
import pandas as pd
import didipack as didi


class Data:
    def __init__(self):
        self.data_dir = 'data/'
        self.raw_data_dir = f'{self.data_dir}raw_data/'
        self.pickle_data_dir = f'{self.data_dir}pickle/'

    def load_ibes(self,reload =False):
        if reload:
            df = pd.read_csv(f'{self.raw_data_dir}ibes.csv')
            df.columns = [x.lower() for x in df.columns]
            df['date']=pd.to_datetime(df['anndats'],format='%Y%m%d')
            df = df[['date','oftic','amaskcd','ireccd']]
            df.columns = ['date','ticker','analyst','rec']
            df.to_pickle(self.pickle_data_dir+'ibes.p')
        else:
            df = pd.read_pickle((self.pickle_data_dir+'ibes.p'))
        return df

    def load_news(self,reload =False):
        if reload:
            df = pd.read_csv(self.raw_data_dir+'news.csv')
            df['date'] = pd.to_datetime(df['date'])
            df.to_pickle(self.pickle_data_dir+'news.p')
        else:
            df = pd.read_pickle(self.pickle_data_dir+'news.p')
        return df

    def load_returns(self, reload =False):
        if reload:
            df = pd.read_csv(self.raw_data_dir+'ret.csv')
            df.columns = [x.lower() for x in df.columns]
            df.loc[:,'siccd'] = pd.to_numeric(df['siccd'],errors='coerce').values
            df = df.dropna()
            df['siccd'] = df['siccd'].astype(int)
            df['industry'] = df['siccd'].apply(lambda x: int(str(x)[:2]))
            df['mkt_cap'] = df['prc'].abs()*df['shrout']
            df = df[['date','ticker','ret','mkt_cap','industry']]
            df['date']=pd.to_datetime(df['date'],format='%Y%m%d')
            df.loc[:,'ret'] = pd.to_numeric(df['ret'],errors='coerce').values
            df = df.dropna()
            df.to_pickle(self.pickle_data_dir+'ret.p')
        else:
            df = pd.read_pickle(self.pickle_data_dir+'ret.p')
        return df

    def load_peer(self):
        df = pd.read_pickle(self.pickle_data_dir+'df_peer.p')
        for c in df.columns[5:]:
            df[c]= (df[c]-df[c].mean())/df[c].std()

        y = df[['y']]
        X = df.iloc[:,6:]

        return X, y

    def load_pred_mean(self):
        data_name = 'simple'
        df = pd.read_pickle(self.pickle_data_dir+'df.p')

        if data_name =='simple':
            df = df[['y','y_lag','lag_ret']].dropna().reset_index(drop=True)
        else:
            df=df.iloc[:,2:]
        for c in df.columns:
            df[c]= (df[c]-df[c].mean())/df[c].std()
        y = df[['y']]
        X = df.drop(columns='y')
        return X, y

    def load_fact(self,reload =False):
        if reload:
            df = pd.read_csv(self.raw_data_dir+'fact.csv')
            df['date'] = pd.to_datetime(df['date'])
            df = df.pivot(index='date',columns='name',values='ret')
            df = df.fillna(0.0)
            df = df.reset_index()
            df.to_pickle(self.pickle_data_dir+'fact.p')
        else:
            df = pd.read_pickle(self.pickle_data_dir+'fact.p')
        return df

    def produce_ticker_list(self):
        df= self.load_ibes(False)
        df[['ticker']].drop_duplicates().to_csv('data/ticker.txt',index=False,header=False)
self = Data()
# df= self.load_ibes(False)
#
