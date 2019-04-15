import time
import random
import pickle
import tqdm
from functools import partial
from multiprocessing import Pool
from scipy.stats.mstats import gmean
from sklearn.utils import resample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')
n_per_year = 12


def monte_carlo(forecast_year_):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S', 'RR'])

    ### Monte Carlo Config ###
    df_SET = pd.read_excel('SET.xlsx', sheet_name='Sheet1')
    init_S = df_SET.iloc[0]['SETi']  # 449.96
    u = df_SET.iloc[1:]['RR'].mean() * n_per_year  # 0.1376
    sigma = df_SET.iloc[1:]['RR'].std() * np.sqrt(n_per_year)  # 0.1580
    dt = 1 / n_per_year

    for t in range(0, (forecast_year_ * n_per_year) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year_ * n_per_year) + 1:
            df.loc[t]['u.dt'] = u * dt
            df.loc[t]['S(u.dt)'] = df.loc[t - 1]['S'] * df.loc[t]['u.dt']
            df.loc[t]['N'] = np.random.normal()
            df.loc[t]['N.sigma.sqrt(dt)'] = df.loc[t]['N'] * sigma * np.sqrt(dt)
            df.loc[t]['S(N.sigma.sqrt(dt))'] = df.loc[t - 1]['S'] * df.loc[t]['N.sigma.sqrt(dt)']
            df.loc[t]['dS'] = df.loc[t]['S(u.dt)'] + df.loc[t]['S(N.sigma.sqrt(dt))']
            df.loc[t]['S'] = df.loc[t - 1]['S'] + df.loc[t]['dS']
            df.loc[t]['RR'] = (df.loc[t]['S'] - df.loc[t - 1]['S']) / df.loc[t - 1]['S']

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def direct(forecast_year_):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RR', 'S'])

    df_SET = pd.read_excel('SET.xlsx', sheet_name='Sheet1')
    RR = df_SET.iloc[1:]['RR'].values
    SETi = df_SET.iloc[1:]['SETi'].values
    init_S = df_SET.iloc[0]['SETi']

    for t in range(0, (forecast_year_ * n_per_year) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year_ * n_per_year) + 1:
            df.loc[t]['RR'] = RR[t - 1]
            df.loc[t]['S'] = SETi[t - 1]

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def bootstrap(forecast_year_):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RR', 'dS', 'S'])

    df_SET = pd.read_excel('SET.xlsx', sheet_name='Sheet1')
    RR = df_SET.iloc[1:]['RR'].values
    RR = resample(RR, replace=True, n_samples=forecast_year_ * n_per_year, random_state=None)
    init_S = df_SET.iloc[0]['SETi']

    for t in range(0, (forecast_year_ * 12) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year_ * n_per_year) + 1:
            df.loc[t]['RR'] = RR[t - 1]
            df.loc[t]['dS'] = df.loc[t]['RR'] * df.loc[t - 1]['S']
            df.loc[t]['S'] = df.loc[t - 1]['S'] + df.loc[t]['dS']

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def LS(df_Stock_, init_Cash_):
    global n_per_year
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = 0.0
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def DCA(df_Stock_, init_Cash_):
    global n_per_year
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / n_per_year / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / n_per_year / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def VA(df_Stock_, init_Cash_):
    global n_per_year
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            diff = ((t + 1) * init_Cash_ / n_per_year) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            diff = ((t + 1) * init_Cash_ / n_per_year) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def simulation(method, forecast_year_, init_Cash_, i):
    ### Portfolio Simulation ###
    global n_per_year
    df_IRR = pd.DataFrame(columns=['Year', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'LS', 'DCA', 'VA'])
    df_IRR_Sum_ = pd.DataFrame(columns=['Iter', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'LS', 'DCA', 'VA'])
    df_LS = {}
    df_DCA = {}
    df_VA = {}
    if method == 1:
        df_Stock = monte_carlo(forecast_year_)
    elif method == 2:
        df_Stock = direct(forecast_year_)
    elif method == 3:
        df_Stock = bootstrap(forecast_year_)

    for year in range(forecast_year_):
        df_LS[year] = LS(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash_)
        df_DCA[year] = DCA(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash_)
        df_VA[year] = VA(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash_)

        df_IRR = df_IRR.append({}, ignore_index=True)
        df_IRR.loc[year]['Year'] = year + 1
        df_IRR.loc[year]['LS'] = df_LS[year].loc[n_per_year]['IRR']
        df_IRR.loc[year]['DCA'] = df_DCA[year].loc[n_per_year]['IRR']
        df_IRR.loc[year]['VA'] = df_VA[year].loc[n_per_year]['IRR']

    df_IRR = df_IRR.append({}, ignore_index=True)
    df_IRR.loc[forecast_year_]['Year'] = 'Avg'
    df_IRR.loc[forecast_year_]['SET_Final'] = df_Stock.iloc[-1]['S']
    df_IRR.loc[forecast_year_]['RR_Mean'] = '{:.2%}'.format(df_Stock.iloc[1:]['RR'].mean() * n_per_year)
    df_IRR.loc[forecast_year_]['RR_Std'] = '{:.2%}'.format(df_Stock.iloc[1:]['RR'].std() * np.sqrt(n_per_year))
    df_IRR.loc[forecast_year_]['RR_Skew'] = df_Stock.iloc[1:]['RR'].skew()
    df_IRR.loc[forecast_year_]['RR_Kurt'] = df_Stock.iloc[1:]['RR'].kurt()
    df_IRR.loc[forecast_year_]['LS'] = '{:.2%}'.format(gmean(1 + (df_IRR.iloc[:-2]['LS'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_IRR.loc[forecast_year_]['DCA'] = '{:.2%}'.format(gmean(1 + (df_IRR.iloc[:-2]['DCA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_IRR.loc[forecast_year_]['VA'] = '{:.2%}'.format(gmean(1 + (df_IRR.iloc[:-2]['VA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_IRR = df_IRR.fillna('')
    df_IRR = df_IRR.set_index('Year')

    ### Summary of IRR ###
    df_IRR_Sum_ = df_IRR_Sum_.append({}, ignore_index=True)
    df_IRR_Sum_['Iter'] = int(i + 1)
    df_IRR_Sum_['SET_Final'] = df_IRR.loc['Avg']['SET_Final']
    df_IRR_Sum_['RR_Mean'] = df_IRR.loc['Avg']['RR_Mean']
    df_IRR_Sum_['RR_Std'] = df_IRR.loc['Avg']['RR_Std']
    df_IRR_Sum_['RR_Skew'] = df_IRR.loc['Avg']['RR_Skew']
    df_IRR_Sum_['RR_Kurt'] = df_IRR.loc['Avg']['RR_Kurt']
    df_IRR_Sum_['LS'] = df_IRR.loc['Avg']['LS']
    df_IRR_Sum_['DCA'] = df_IRR.loc['Avg']['DCA']
    df_IRR_Sum_['VA'] = df_IRR.loc['Avg']['VA']

    # for col in df_IRR.columns.values:
    #     df_IRR[col] = df_IRR[col].str.rstrip('%').astype('float') / 100.0
    # df_IRR.to_excel('output{}.xlsx'.format(i))

    # if i == 0:
    # print()
    # print(df_Stock)
    # print(df_LS[0])
    # print(df_DCA[0])
    # for j in df_VA:
    #     print(df_VA[j])
    # print(df_IRR)

    return df_IRR_Sum_.values.tolist()


if __name__ == '__main__':
    ### Simulation Config ###
    method = 3  # 1: Monte Carlo, 2: Direct Test, 3: Bootstrap
    iter = 100
    forecast_year = 10
    np.random.seed(None)

    ### Initial value ###
    init_Cash = 120000.0

    results = []
    pool = Pool()
    iter = iter if method != 2 else 1
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, method, forecast_year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_IRR_Sum = pd.DataFrame(results, columns=['Iter', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'LS', 'DCA', 'VA'], dtype='object')
    df_IRR_Sum.sort_values(by='Iter', inplace=True)

    df_IRR_Sum = df_IRR_Sum.append({}, ignore_index=True)
    df_IRR_Sum.iloc[-1]['Iter'] = 'Avg'
    df_IRR_Sum.iloc[-1]['SET_Final'] = df_IRR_Sum.iloc[:-1]['SET_Final'].mean()
    df_IRR_Sum.iloc[-1]['RR_Mean'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['RR_Mean'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['RR_Std'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['RR_Std'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['RR_Skew'] = df_IRR_Sum.iloc[:-1]['RR_Skew'].mean()
    df_IRR_Sum.iloc[-1]['RR_Kurt'] = df_IRR_Sum.iloc[:-1]['RR_Kurt'].mean()
    df_IRR_Sum.iloc[-1]['LS'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['DCA'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['VA'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum = df_IRR_Sum.fillna('')
    df_IRR_Sum = df_IRR_Sum.set_index('Iter')
    df_IRR_Sum.rename(columns={'LS': 'Simulation_LS', 'DCA': 'Simulation_DCA', 'VA': 'Simulation_VA'}, inplace=True)
    df_IRR_Sum.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], col.split('_')[-1]) for col in df_IRR_Sum.columns])

    print(df_IRR_Sum)
    xlsx = pd.ExcelWriter('output.xlsx')
    df_IRR_Sum.to_excel(xlsx, "IRR_Sum")
    sheet = xlsx.book.worksheets[0]
    for cell in sheet['B'][3:] + sheet['E'][3:] + sheet['F'][3:]:
        cell.value = float(cell.value)
        cell.number_format = '#,##0.00'
    for cell in sheet['C'][3:] + sheet['D'][3:] + sheet['G'][3:] + sheet['H'][3:] + sheet['I'][3:]:
        cell.value = float(str(cell.value).rstrip('%')) / 100
        cell.number_format = '0.00%'
    xlsx.save()
