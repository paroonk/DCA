from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pickle
import tqdm
import xlsxwriter.utility
from matplotlib import style
from scipy.stats.mstats import gmean
from sklearn.utils import resample

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')
n_per_year = 12
col_Transaction = ['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                   'Inv.Asset Price', 'Capital Gain', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                   'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR']
col_Simulation = ['Year', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'IRR_LS', 'IRR_DCA', 'IRR_VA']
col_Summary = ['Iter', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'IRR_LS', 'IRR_DCA', 'IRR_VA']

### Simulation Config ###
forecast_year = 10
init_Cash = 120000.0


# def fund(forecast_year):
#     global n_per_year
#     df = pd.DataFrame(columns=['Month', 'RR', 'S'])
#
#     RR = df_Price.iloc[1:]['RR'].values
#     SETi = df_Price.iloc[1:]['SETi'].values
#     init_S = df_Price.iloc[0]['SETi']
#
#     for t in range(0, (forecast_year * n_per_year) + 1):
#         df = df.append({}, ignore_index=True)
#         df.loc[t]['Month'] = t
#
#         if t == 0:
#             df.loc[t]['S'] = init_S
#         elif t > 0 and (forecast_year * n_per_year) + 1:
#             df.loc[t]['RR'] = RR[t - 1]
#             df.loc[t]['S'] = SETi[t - 1]
#
#     df = df.fillna('')
#     df['Month'] = df['Month'].astype('int')
#     df = df.set_index('Month')
#     return df
#
#
# def simulation(method, df_Price, forecast_year, init_Cash, i):
#     ### Portfolio Simulation ###
#     global n_per_year
#     global col_Simulation
#     global col_Summary
#     df_Simulation = pd.DataFrame(columns=col_Simulation)
#     df_Summary_ = pd.DataFrame(columns=col_Summary)
#     df_LS = {}
#     df_DCA = {}
#     df_VA = {}
#
#     df_Stock = direct(df_Price, forecast_year)
#
#     for year in range(forecast_year):
#         df_LS[year] = LS(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash)
#         df_DCA[year] = DCA(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash)
#         df_VA[year] = VA(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash)
#         df_Simulation = df_Simulation.append({}, ignore_index=True)
#         df_Simulation.loc[year]['Year'] = year + 1
#         df_Simulation.loc[year]['IRR_LS'] = df_LS[year].loc[n_per_year]['IRR']
#         df_Simulation.loc[year]['IRR_DCA'] = df_DCA[year].loc[n_per_year]['IRR']
#         df_Simulation.loc[year]['IRR_VA'] = df_VA[year].loc[n_per_year]['IRR']

    # df_Simulation = df_Simulation.append({}, ignore_index=True)
    # df_Simulation.loc[forecast_year]['Year'] = 'Avg'
    # df_Simulation.loc[forecast_year]['SET_Final'] = df_Stock.iloc[-1]['S']
    # df_Simulation.loc[forecast_year]['RR_Mean'] = '{:.2%}'.format(df_Stock.iloc[1:]['RR'].mean() * n_per_year)
    # df_Simulation.loc[forecast_year]['RR_Std'] = '{:.2%}'.format(df_Stock.iloc[1:]['RR'].std() * np.sqrt(n_per_year))
    # df_Simulation.loc[forecast_year]['RR_Skew'] = df_Stock.iloc[1:]['RR'].skew()
    # df_Simulation.loc[forecast_year]['RR_Kurt'] = df_Stock.iloc[1:]['RR'].kurt()
    # df_Simulation.loc[forecast_year]['IRR_LS'] = '{:.2%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_LS'].str.rstrip('%').astype('float') / 100.0)) - 1)
    # df_Simulation.loc[forecast_year]['IRR_DCA'] = '{:.2%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_DCA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    # df_Simulation.loc[forecast_year]['IRR_VA'] = '{:.2%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_VA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    # df_Simulation = df_Simulation.fillna('')
    # df_Simulation = df_Simulation.set_index('Year')
    #

    ### Summary of IRR ###
    # df_Summary_ = df_Summary_.append({}, ignore_index=True)
    # df_Summary_['Iter'] = int(i + 1)
    # df_Summary_['SET_Final'] = df_Simulation.loc['Avg']['SET_Final']
    # df_Summary_['RR_Mean'] = df_Simulation.loc['Avg']['RR_Mean']
    # df_Summary_['RR_Std'] = df_Simulation.loc['Avg']['RR_Std']
    # df_Summary_['RR_Skew'] = df_Simulation.loc['Avg']['RR_Skew']
    # df_Summary_['RR_Kurt'] = df_Simulation.loc['Avg']['RR_Kurt']
    # df_Summary_['IRR_LS'] = df_Simulation.loc['Avg']['IRR_LS']
    # df_Summary_['IRR_DCA'] = df_Simulation.loc['Avg']['IRR_DCA']
    # df_Summary_['IRR_VA'] = df_Simulation.loc['Avg']['IRR_VA']

    # return df_Summary_.values.tolist()
    # return []


def get_col_widths(df, index=True):
    if index:
        idx_max = max([len(str(s)) for s in df.index.values] + [len(str(df.index.name))])
        col_widths = [idx_max] + [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    else:
        col_widths = [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    return col_widths


if __name__ == '__main__':

    results = []
    pool = Pool()
    # for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, method, df_SET, forecast_year, init_Cash), range(iter)), total=iter):
    #     pass
    #     results.extend(result)

    # df_Summary = pd.DataFrame(results, columns=col_Summary, dtype='object')
    # df_Summary.sort_values(by='Iter', inplace=True)
    #
    # print(df_Summary)

    pickle_in = open('data/NAV.pickle', 'rb')
    df_NAV = pickle.load(pickle_in)
    df_NAV = df_NAV.fillna(method='ffill').fillna(method='bfill')

    pickle_in = open('data/Div.pickle', 'rb')
    df_Div = pickle.load(pickle_in)
    # df_Div = df_Div.fillna(0).cumsum()

    print(df_NAV.head())

    df_first = df_NAV.resample('MS').first()
    df_last = df_NAV.resample('M').last()
    df_return = df_first.head(1).append(df_last)
    print(df_return.head())
