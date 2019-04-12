import time
import random
import pickle
import tqdm
from functools import partial
from multiprocessing import Pool
from sklearn.utils import resample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')


def LS(df_Forecast_, n_per_year_, init_Cash_):
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])
    for t in range(0, n_per_year_ + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / df_Forecast_.loc[t]['S']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year_):
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
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
        elif t == n_per_year_:
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
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
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year_) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def DCA(df_Forecast_, n_per_year_, init_Cash_):
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])
    for t in range(0, n_per_year_ + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / n_per_year_ / df_Forecast_.loc[t]['S']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year_):
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / n_per_year_ / df_Forecast_.loc[t]['S']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year_:
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
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
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year_) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def VA(df_Forecast_, n_per_year_, init_Cash_):
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])
    for t in range(0, n_per_year_ + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            diff = ((t + 1) * init_Cash_ / n_per_year_) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df_Forecast_.loc[t]['S']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year_):
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            diff = ((t + 1) * init_Cash_ / n_per_year_) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df_Forecast_.loc[t]['S']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year_:
            df.loc[t]['Inv.Asset Price'] = df_Forecast_.loc[t]['S']
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
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year_) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def simulation(forecast_year_, n_per_year_, init_S_, u_, sigma_, init_Cash_, i):
    # dt = 1 / n_per_year_
    #
    # ### df_Forecast Price ###
    # df_Forecast = pd.DataFrame(
    #     columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S'])
    #
    # for t in range(0, (forecast_year_ * n_per_year_) + 1):
    #     df_Forecast = df_Forecast.append({}, ignore_index=True)
    #     df_Forecast.loc[t]['Month'] = t
    #
    #     if t == 0:
    #         df_Forecast.loc[t]['S'] = init_S_
    #     elif t > 0 and (forecast_year_ * n_per_year_) + 1:
    #         df_Forecast.loc[t]['u.dt'] = u_ * dt
    #         df_Forecast.loc[t]['S(u.dt)'] = df_Forecast.loc[t - 1]['S'] * df_Forecast.loc[t]['u.dt']
    #         df_Forecast.loc[t]['N'] = np.random.normal()
    #         df_Forecast.loc[t]['N.sigma.sqrt(dt)'] = df_Forecast.loc[t]['N'] * sigma_ * np.sqrt(dt)
    #         df_Forecast.loc[t]['S(N.sigma.sqrt(dt))'] = df_Forecast.loc[t - 1]['S'] * df_Forecast.loc[t]['N.sigma.sqrt(dt)']
    #         df_Forecast.loc[t]['dS'] = df_Forecast.loc[t]['S(u.dt)'] + df_Forecast.loc[t]['S(N.sigma.sqrt(dt))']
    #         df_Forecast.loc[t]['S'] = df_Forecast.loc[t - 1]['S'] + df_Forecast.loc[t]['dS']
    #
    # df_Forecast = df_Forecast.fillna('')
    # df_Forecast['Month'] = df_Forecast['Month'].astype('int')
    # df_Forecast = df_Forecast.set_index('Month')
    #
    # ### Portfolio Simulation ###
    # df_IRR = pd.DataFrame(columns=['Year', 'LS', 'DCA', 'VA'])
    #
    # df_LS = {}
    # df_DCA = {}
    # df_VA = {}
    # for year in range(forecast_year_):
    #     df_LS[year] = LS(df_Forecast.iloc[(year * n_per_year_):((year + 1) * n_per_year_) + 1].reset_index(), n_per_year_, init_Cash_)
    #     df_DCA[year] = DCA(df_Forecast.iloc[(year * n_per_year_):((year + 1) * n_per_year_) + 1].reset_index(), n_per_year_, init_Cash_)
    #     df_VA[year] = VA(df_Forecast.iloc[(year * n_per_year_):((year + 1) * n_per_year_) + 1].reset_index(), n_per_year_, init_Cash_)
    #
    #     df_IRR = df_IRR.append({}, ignore_index=True)
    #     df_IRR.loc[year]['Year'] = year + 1
    #     df_IRR.loc[year]['LS'] = df_LS[year].loc[n_per_year_]['IRR']
    #     df_IRR.loc[year]['DCA'] = df_DCA[year].loc[n_per_year_]['IRR']
    #     df_IRR.loc[year]['VA'] = df_VA[year].loc[n_per_year_]['IRR']
    #
    # df_IRR = df_IRR.append({}, ignore_index=True)
    # df_IRR.loc[forecast_year_]['Year'] = 'Avg'
    # df_IRR.loc[forecast_year_]['LS'] = '{:.2%}'.format((df_IRR.iloc[:-1]['LS'].str.rstrip('%').astype('float') / 100.0).mean())
    # df_IRR.loc[forecast_year_]['DCA'] = '{:.2%}'.format((df_IRR.iloc[:-1]['DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    # df_IRR.loc[forecast_year_]['VA'] = '{:.2%}'.format((df_IRR.iloc[:-1]['VA'].str.rstrip('%').astype('float') / 100.0).mean())
    # df_IRR = df_IRR.fillna('')
    # df_IRR = df_IRR.set_index('Year')
    #
    # ### Summary of IRR ###
    df_IRR_Sum_ = pd.DataFrame(columns=['Iter', 'LS', 'DCA', 'VA'])
    # df_IRR_Sum_ = df_IRR_Sum_.append({}, ignore_index=True)
    # df_IRR_Sum_['Iter'] = int(i + 1)
    # df_IRR_Sum_['LS'] = df_IRR.loc['Avg']['LS']
    # df_IRR_Sum_['DCA'] = df_IRR.loc['Avg']['DCA']
    # df_IRR_Sum_['VA'] = df_IRR.loc['Avg']['VA']

    if i == 0:
        print()
        # print(df_Forecast)
        # print(df_LS[0])
        # print(df_DCA[0])
        # for j in df_VA:
        #     print(df_VA[j])
        # print(df_IRR)

    return df_IRR_Sum_.values.tolist()


if __name__ == '__main__':

    start = time.time()

    ### Simulation Config ###
    iter = 5
    forecast_year = 10
    n_per_year = 12
    np.random.seed(None)

    ### Initial value ###
    init_S = 100
    u = 0.1
    sigma = 0.15

    init_Cash = 120000.0

    results = []
    pool = Pool()
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, forecast_year, n_per_year, init_S, u, sigma, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_IRR_Sum = pd.DataFrame(results, columns=['Iter', 'LS', 'DCA', 'VA'], dtype='object')
    df_IRR_Sum.sort_values(by='Iter', inplace=True)

    df_IRR_Sum = df_IRR_Sum.append({}, ignore_index=True)
    df_IRR_Sum.iloc[-1]['Iter'] = 'Avg'
    df_IRR_Sum.iloc[-1]['LS'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['DCA'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['VA'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum = df_IRR_Sum.fillna('')
    df_IRR_Sum = df_IRR_Sum.set_index('Iter')

    print(df_IRR_Sum)
    # print(df_IRR_Sum.loc[iter])
