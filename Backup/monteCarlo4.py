import numpy as np
import pandas as pd
import pickle
import random
import math
import time
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.utils import resample

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# boot = resample(x, replace=True, n_samples=100000, random_state=None)
# print(boot.mean())


def LS(forecast):
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])
    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / forecast.loc[t]['S']
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Inv.Asset Price'] = forecast.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Beg. Cash'] = init_Cash
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Inv.Asset Price'] = forecast.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']

            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Inv.Asset Price'] = forecast.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def DCA(forecast):
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])
    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / n_per_year / forecast.loc[t]['S']
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Inv.Asset Price'] = forecast.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = 0.0
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Beg. Cash'] = init_Cash
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / n_per_year / forecast.loc[t]['S']
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Inv.Asset Price'] = forecast.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Inv.Asset Price'] = forecast.loc[t]['S']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain+Dividend'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


if __name__ == '__main__':

    starttime = time.time()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    iter = 3
    n_per_year = 12
    dt = 1 / n_per_year
    forecast_year = 10
    np.random.seed(None)

    init_S = 100
    u = 0.1
    sigma = 0.15

    init_Cash = 120000.0

    df_forecast = {}
    df_LS = {}
    df_DCA = {}
    df_IRR = {}
    df_IRR_Sum = pd.DataFrame(columns=['Iter', 'LS', 'DCA', 'VA'])

    for i in range(iter):
        ### Forecast Price ###
        df_forecast[i] = pd.DataFrame(
            columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S'])

        for t in range(0, (forecast_year * n_per_year) + 1):
            df_forecast[i] = df_forecast[i].append({}, ignore_index=True)
            df_forecast[i].loc[t]['Month'] = t

            if t == 0:
                df_forecast[i].loc[t]['S'] = init_S
            elif t > 0 and (forecast_year * n_per_year) + 1:
                df_forecast[i].loc[t]['u.dt'] = u * dt
                df_forecast[i].loc[t]['S(u.dt)'] = df_forecast[i].loc[t - 1]['S'] * df_forecast[i].loc[t]['u.dt']
                df_forecast[i].loc[t]['N'] = np.random.normal()
                df_forecast[i].loc[t]['N.sigma.sqrt(dt)'] = df_forecast[i].loc[t]['N'] * sigma * np.sqrt(dt)
                df_forecast[i].loc[t]['S(N.sigma.sqrt(dt))'] = df_forecast[i].loc[t - 1]['S'] * df_forecast[i].loc[t]['N.sigma.sqrt(dt)']
                df_forecast[i].loc[t]['dS'] = df_forecast[i].loc[t]['S(u.dt)'] + df_forecast[i].loc[t]['S(N.sigma.sqrt(dt))']
                df_forecast[i].loc[t]['S'] = df_forecast[i].loc[t - 1]['S'] + df_forecast[i].loc[t]['dS']

        df_forecast[i] = df_forecast[i].fillna('')
        df_forecast[i]['Month'] = df_forecast[i]['Month'].astype('int')
        df_forecast[i] = df_forecast[i].set_index('Month')
        if i in np.arange(0, iter, iter / 100):
            df_forecast[i].plot(y='S', kind='line', ax=ax[0])

        ### Portfolio Simulation ###
        df_IRR[i] = pd.DataFrame(columns=['Year', 'LS', 'DCA', 'VA'])

        for year in range(forecast_year):

            df_LS[i, year] = LS(df_forecast[i].iloc[(year * n_per_year):((year + 1) * n_per_year) + 1].reset_index())
            # print(df_forecast[i].iloc[(year * n_per_year):((year + 1) * n_per_year) + 1])
            # print(df_LS[i, year])

            df_DCA[i, year] = DCA(df_forecast[i].iloc[(year * n_per_year):((year + 1) * n_per_year) + 1].reset_index())
            # print(df_forecast[i].iloc[(year * n_per_year):((year + 1) * n_per_year) + 1])
            # print(df_DCA[i, year])

            df_IRR[i] = df_IRR[i].append({}, ignore_index=True)
            df_IRR[i].loc[year]['Year'] = year + 1
            df_IRR[i].loc[year]['LS'] = df_LS[i, year].loc[n_per_year]['IRR']
            df_IRR[i].loc[year]['DCA'] = df_DCA[i, year].loc[n_per_year]['IRR']

        df_IRR[i] = df_IRR[i].append({}, ignore_index=True)
        df_IRR[i].loc[forecast_year]['Year'] = 'Avg'
        df_IRR[i].loc[forecast_year]['LS'] = '{:.2%}'.format((df_IRR[i].iloc[:-1]['LS'].str.rstrip('%').astype('float') / 100.0).mean())
        df_IRR[i].loc[forecast_year]['DCA'] = '{:.2%}'.format((df_IRR[i].iloc[:-1]['DCA'].str.rstrip('%').astype('float') / 100.0).mean())
        df_IRR[i] = df_IRR[i].fillna('')
        df_IRR[i] = df_IRR[i].set_index('Year')
        # print(df_IRR[i])

        df_IRR_Sum = df_IRR_Sum.append({}, ignore_index=True)
        df_IRR_Sum.loc[i]['Iter'] = i + 1
        df_IRR_Sum.loc[i]['LS'] = df_IRR[i].loc['Avg']['LS']
        df_IRR_Sum.loc[i]['DCA'] = df_IRR[i].loc['Avg']['DCA']

        print('Iter: {}/{}'.format(i + 1, iter))


        # n_bin = 15
    # S_dist = []
    # for i in range(len(forecast)):
    #     S_dist.append(forecast[i].loc[forecast_year]['S'])
    # hmin, hmax = math.floor(min(S_dist)), math.ceil(max(S_dist))
    # gap = math.ceil((hmax - hmin) / n_bin)
    # bins = np.linspace(hmin, hmax, n_bin + 1)
    # ax[1].hist(S_dist, bins, ec='k', alpha=0.8)
    # ax[1].axvline(np.percentile(S_dist, 5), color='grey', linestyle='dashed', linewidth=1)
    # ax[1].axvline(np.percentile(S_dist, 25), color='grey', linestyle='dashed', linewidth=1)
    # ax[1].axvline(np.percentile(S_dist, 50), color='grey', linestyle='dashed', linewidth=1)
    # ax[1].axvline(np.percentile(S_dist, 75), color='grey', linestyle='dashed', linewidth=1)
    # ax[1].axvline(np.percentile(S_dist, 95), color='grey', linestyle='dashed', linewidth=1)
    # x0, x1 = ax[1].get_xlim()
    # y0, y1 = ax[1].get_ylim()
    # ax[1].text(np.percentile(S_dist, 5) + 0.01 * (x1 - x0), y0 + 0.90 * (y1 - y0), 'P5: {:.4f}'.format(np.percentile(S_dist, 5)), color='blue')
    # ax[1].text(np.percentile(S_dist, 25) + 0.01 * (x1 - x0), y0 + 0.85 * (y1 - y0), 'P25: {:.4f}'.format(np.percentile(S_dist, 25)), color='blue')
    # ax[1].text(np.percentile(S_dist, 50) + 0.01 * (x1 - x0), y0 + 0.90 * (y1 - y0), 'Med: {:.4f}'.format(np.percentile(S_dist, 50)), color='blue')
    # ax[1].text(np.percentile(S_dist, 75) + 0.01 * (x1 - x0), y0 + 0.85 * (y1 - y0), 'P75: {:.4f}'.format(np.percentile(S_dist, 75)), color='blue')
    # ax[1].text(np.percentile(S_dist, 95) + 0.01 * (x1 - x0), y0 + 0.90 * (y1 - y0), 'P95: {:.4f}'.format(np.percentile(S_dist, 95)), color='blue')
    #
    # fig.suptitle('Forecast')
    # ax[0].set_xlabel('Month', fontsize=10)
    # ax[0].set_ylabel('S', fontsize=10)
    # ax[0].legend().set_visible(False)
    # ax[1].set_xticks(bins)
    # ax[1].set_xlabel('S@Month={}'.format(forecast_year), fontsize=10)
    # ax[1].set_ylabel('Frequency', fontsize=10)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.92, left=0.08, right=0.92, hspace=0.25)
    # plt.show()

    df_IRR_Sum = df_IRR_Sum.append({}, ignore_index=True)
    df_IRR_Sum.loc[iter]['Iter'] = 'Avg'
    df_IRR_Sum.loc[iter]['LS'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.loc[iter]['DCA'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum = df_IRR_Sum.fillna('')
    df_IRR_Sum = df_IRR_Sum.set_index('Iter')
    print()
    print(df_IRR_Sum)
    # print(df_IRR_Sum.loc[iter])

    print('Elapsed time: {:.2f} seconds'.format(time.time() - starttime))