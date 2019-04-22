import numpy as np
import pandas as pd
import pickle
import random
import math
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.utils import resample

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')
np.random.seed(None)

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# boot = resample(x, replace=True, n_samples=100000, random_state=None)
# print(boot.mean())

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

iter = 1000
n_per_year = 12
dt = 1 / n_per_year
forecast_year = 10

init_S = 100
u = 0.1
sigma = 0.15

init_Cash = 120000.0

forecast = {}
LS = {}
DCA = {}
df_IRR = {}
df_IRR_Sum = pd.DataFrame(columns=['Iter', 'LS', 'DCA', 'VA'])
for i in range(iter):

    ### Forecast Price ###
    forecast[i] = pd.DataFrame(
        columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S'])

    for t in range(0, (forecast_year * n_per_year) + 1):
        forecast[i] = forecast[i].append({}, ignore_index=True)
        forecast[i].loc[t]['Month'] = t

        if t == 0:
            forecast[i].loc[t]['S'] = init_S
        elif t > 0 and (forecast_year * n_per_year) + 1:
            forecast[i].loc[t]['u.dt'] = u * dt
            forecast[i].loc[t]['S(u.dt)'] = forecast[i].loc[t - 1]['S'] * forecast[i].loc[t]['u.dt']
            forecast[i].loc[t]['N'] = np.random.normal()
            forecast[i].loc[t]['N.sigma.sqrt(dt)'] = forecast[i].loc[t]['N'] * sigma * np.sqrt(dt)
            forecast[i].loc[t]['S(N.sigma.sqrt(dt))'] = forecast[i].loc[t - 1]['S'] * forecast[i].loc[t]['N.sigma.sqrt(dt)']
            forecast[i].loc[t]['dS'] = forecast[i].loc[t]['S(u.dt)'] + forecast[i].loc[t]['S(N.sigma.sqrt(dt))']
            forecast[i].loc[t]['S'] = forecast[i].loc[t - 1]['S'] + forecast[i].loc[t]['dS']

    forecast[i] = forecast[i].fillna('')
    forecast[i]['Month'] = forecast[i]['Month'].astype('int')
    forecast[i] = forecast[i].set_index('Month')
    if i in np.arange(0, iter, iter / 100):
        forecast[i].plot(y='S', kind='line', ax=ax[0])

    ### Portfolio Simulation ###
    df_IRR[i] = pd.DataFrame(columns=['Year', 'LS', 'DCA', 'VA'])
    for year in range(1, forecast_year + 1):

        LS[i, year] = pd.DataFrame(
            columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                     'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                     'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])
        for t in range(0, n_per_year + 1):
            LS[i, year] = LS[i, year].append({}, ignore_index=True)
            LS[i, year].loc[t]['Month'] = t
            if t == 0:
                LS[i, year].loc[t]['Beg. Inv.Asset Volume'] = 0.0
                LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                LS[i, year].loc[t]['Net Inv.Asset Volume'] = LS[i, year].loc[t]['Beg. Inv.Asset Volume'] + LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume']
                LS[i, year].loc[t]['Inv.Asset Price'] = forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                LS[i, year].loc[t]['Beg. Inv.Asset Value'] = LS[i, year].loc[t]['Beg. Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Capital Gain+Dividend'] = 0.0
                LS[i, year].loc[t]['Change in Inv.Asset Value'] = LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Net Inv.Asset Value'] = LS[i, year].loc[t]['Net Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Beg. Cash'] = init_Cash
                LS[i, year].loc[t]['Change in Cash'] = -LS[i, year].loc[t]['Change in Inv.Asset Value'] if \
                    LS[i, year].loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
                LS[i, year].loc[t]['Net Cash'] = LS[i, year].loc[t]['Beg. Cash'] + LS[i, year].loc[t]['Change in Cash']
                LS[i, year].loc[t]['Total Wealth'] = LS[i, year].loc[t]['Net Inv.Asset Value'] + LS[i, year].loc[t]['Net Cash']
                LS[i, year].loc[t]['Profit/Loss'] = LS[i, year].loc[t]['Total Wealth'] - init_Cash
            elif t in range(1, n_per_year):
                LS[i, year].loc[t]['Beg. Inv.Asset Volume'] = LS[i, year].loc[t - 1]['Net Inv.Asset Volume']
                LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] = 0.0
                LS[i, year].loc[t]['Net Inv.Asset Volume'] = LS[i, year].loc[t]['Beg. Inv.Asset Volume'] + LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume']
                LS[i, year].loc[t]['Inv.Asset Price'] = forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                LS[i, year].loc[t]['Beg. Inv.Asset Value'] = LS[i, year].loc[t]['Beg. Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Capital Gain+Dividend'] = LS[i, year].loc[t]['Beg. Inv.Asset Value'] - LS[i, year].loc[t - 1]['Net Inv.Asset Value']
                LS[i, year].loc[t]['Change in Inv.Asset Value'] = LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Net Inv.Asset Value'] = LS[i, year].loc[t]['Net Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Beg. Cash'] = LS[i, year].loc[t - 1]['Net Cash']
                # todo interest not included yet
                LS[i, year].loc[t]['Change in Cash'] = -LS[i, year].loc[t]['Change in Inv.Asset Value'] if \
                    LS[i, year].loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
                LS[i, year].loc[t]['Net Cash'] = LS[i, year].loc[t]['Beg. Cash'] + LS[i, year].loc[t]['Change in Cash']
                LS[i, year].loc[t]['Total Wealth'] = LS[i, year].loc[t]['Net Inv.Asset Value'] + LS[i, year].loc[t]['Net Cash']
                LS[i, year].loc[t]['Profit/Loss'] = LS[i, year].loc[t]['Total Wealth'] - init_Cash
            elif t == n_per_year:
                LS[i, year].loc[t]['Beg. Inv.Asset Volume'] = LS[i, year].loc[t - 1]['Net Inv.Asset Volume']
                LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] = -LS[i, year].loc[t - 1]['Net Inv.Asset Volume']
                LS[i, year].loc[t]['Net Inv.Asset Volume'] = LS[i, year].loc[t]['Beg. Inv.Asset Volume'] + LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume']
                LS[i, year].loc[t]['Inv.Asset Price'] = forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                LS[i, year].loc[t]['Beg. Inv.Asset Value'] = LS[i, year].loc[t]['Beg. Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Capital Gain+Dividend'] = LS[i, year].loc[t]['Beg. Inv.Asset Value'] - LS[i, year].loc[t - 1]['Net Inv.Asset Value']
                LS[i, year].loc[t]['Change in Inv.Asset Value'] = LS[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Net Inv.Asset Value'] = LS[i, year].loc[t]['Net Inv.Asset Volume'] * LS[i, year].loc[t]['Inv.Asset Price']
                LS[i, year].loc[t]['Beg. Cash'] = LS[i, year].loc[t - 1]['Net Cash']
                LS[i, year].loc[t]['Change in Cash'] = -LS[i, year].loc[t]['Change in Inv.Asset Value'] if \
                    LS[i, year].loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
                LS[i, year].loc[t]['Net Cash'] = LS[i, year].loc[t]['Beg. Cash'] + LS[i, year].loc[t]['Change in Cash']
                LS[i, year].loc[t]['Total Wealth'] = LS[i, year].loc[t]['Net Inv.Asset Value'] + LS[i, year].loc[t]['Net Cash']
                LS[i, year].loc[t]['Profit/Loss'] = LS[i, year].loc[t]['Total Wealth'] - init_Cash
                LS[i, year].loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(LS[i, year]['Change in Cash'].tolist())) ** n_per_year) - 1)

        DCA[i, year] = pd.DataFrame(
            columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                     'Inv.Asset Price', 'Capital Gain+Dividend', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                     'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])
        for t in range(0, n_per_year + 1):
            DCA[i, year] = DCA[i, year].append({}, ignore_index=True)
            DCA[i, year].loc[t]['Month'] = t
            if t == 0:
                DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] = 0.0
                DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / n_per_year / forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                DCA[i, year].loc[t]['Net Inv.Asset Volume'] = DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] + DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume']
                DCA[i, year].loc[t]['Inv.Asset Price'] = forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                DCA[i, year].loc[t]['Beg. Inv.Asset Value'] = DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Capital Gain+Dividend'] = 0.0
                DCA[i, year].loc[t]['Change in Inv.Asset Value'] = DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Net Inv.Asset Value'] = DCA[i, year].loc[t]['Net Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Beg. Cash'] = init_Cash
                DCA[i, year].loc[t]['Change in Cash'] = -DCA[i, year].loc[t]['Change in Inv.Asset Value'] if \
                    DCA[i, year].loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
                DCA[i, year].loc[t]['Net Cash'] = DCA[i, year].loc[t]['Beg. Cash'] + DCA[i, year].loc[t]['Change in Cash']
                DCA[i, year].loc[t]['Total Wealth'] = DCA[i, year].loc[t]['Net Inv.Asset Value'] + DCA[i, year].loc[t]['Net Cash']
                DCA[i, year].loc[t]['Profit/Loss'] = DCA[i, year].loc[t]['Total Wealth'] - init_Cash
            elif t in range(1, n_per_year):
                DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] = DCA[i, year].loc[t - 1]['Net Inv.Asset Volume']
                DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / n_per_year / forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                DCA[i, year].loc[t]['Net Inv.Asset Volume'] = DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] + DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume']
                DCA[i, year].loc[t]['Inv.Asset Price'] = forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                DCA[i, year].loc[t]['Beg. Inv.Asset Value'] = DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Capital Gain+Dividend'] = DCA[i, year].loc[t]['Beg. Inv.Asset Value'] - DCA[i, year].loc[t - 1]['Net Inv.Asset Value']
                DCA[i, year].loc[t]['Change in Inv.Asset Value'] = DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Net Inv.Asset Value'] = DCA[i, year].loc[t]['Net Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Beg. Cash'] = DCA[i, year].loc[t - 1]['Net Cash']
                # todo interest not included yet
                DCA[i, year].loc[t]['Change in Cash'] = -DCA[i, year].loc[t]['Change in Inv.Asset Value'] if \
                    DCA[i, year].loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
                DCA[i, year].loc[t]['Net Cash'] = DCA[i, year].loc[t]['Beg. Cash'] + DCA[i, year].loc[t]['Change in Cash']
                DCA[i, year].loc[t]['Total Wealth'] = DCA[i, year].loc[t]['Net Inv.Asset Value'] + DCA[i, year].loc[t]['Net Cash']
                DCA[i, year].loc[t]['Profit/Loss'] = DCA[i, year].loc[t]['Total Wealth'] - init_Cash
            elif t == n_per_year:
                DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] = DCA[i, year].loc[t - 1]['Net Inv.Asset Volume']
                DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] = -DCA[i, year].loc[t - 1]['Net Inv.Asset Volume']
                DCA[i, year].loc[t]['Net Inv.Asset Volume'] = DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] + DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume']
                DCA[i, year].loc[t]['Inv.Asset Price'] = forecast[i].loc[t + ((year - 1) * n_per_year)]['S']
                DCA[i, year].loc[t]['Beg. Inv.Asset Value'] = DCA[i, year].loc[t]['Beg. Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Capital Gain+Dividend'] = DCA[i, year].loc[t]['Beg. Inv.Asset Value'] - DCA[i, year].loc[t - 1]['Net Inv.Asset Value']
                DCA[i, year].loc[t]['Change in Inv.Asset Value'] = DCA[i, year].loc[t]['Buy/Sell Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Net Inv.Asset Value'] = DCA[i, year].loc[t]['Net Inv.Asset Volume'] * DCA[i, year].loc[t]['Inv.Asset Price']
                DCA[i, year].loc[t]['Beg. Cash'] = DCA[i, year].loc[t - 1]['Net Cash']
                DCA[i, year].loc[t]['Change in Cash'] = -DCA[i, year].loc[t]['Change in Inv.Asset Value'] if \
                    DCA[i, year].loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
                DCA[i, year].loc[t]['Net Cash'] = DCA[i, year].loc[t]['Beg. Cash'] + DCA[i, year].loc[t]['Change in Cash']
                DCA[i, year].loc[t]['Total Wealth'] = DCA[i, year].loc[t]['Net Inv.Asset Value'] + DCA[i, year].loc[t]['Net Cash']
                DCA[i, year].loc[t]['Profit/Loss'] = DCA[i, year].loc[t]['Total Wealth'] - init_Cash
                DCA[i, year].loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(DCA[i, year]['Change in Cash'].tolist())) ** n_per_year) - 1)

        LS[i, year] = LS[i, year].fillna('')
        LS[i, year]['Month'] = LS[i, year]['Month'].astype('int')
        LS[i, year] = LS[i, year].set_index('Month')
        # print(LS[i, year])

        DCA[i, year] = DCA[i, year].fillna('')
        DCA[i, year]['Month'] = DCA[i, year]['Month'].astype('int')
        DCA[i, year] = DCA[i, year].set_index('Month')
        # print(DCA[i, year])

        df_IRR[i] = df_IRR[i].append({}, ignore_index=True)
        df_IRR[i].loc[year - 1]['Year'] = year
        df_IRR[i].loc[year - 1]['LS'] = LS[i, year].loc[n_per_year]['IRR']
        df_IRR[i].loc[year - 1]['DCA'] = DCA[i, year].loc[n_per_year]['IRR']

    df_IRR[i] = df_IRR[i].append({}, ignore_index=True)
    df_IRR[i].loc[year]['Year'] = 'Avg'
    df_IRR[i].loc[year]['LS'] = '{:.2%}'.format((df_IRR[i].iloc[:-1]['LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR[i].loc[year]['DCA'] = '{:.2%}'.format((df_IRR[i].iloc[:-1]['DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR[i] = df_IRR[i].fillna('')
    df_IRR[i] = df_IRR[i].set_index('Year')
    # print(df_IRR[i])

    df_IRR_Sum = df_IRR_Sum.append({}, ignore_index=True)
    df_IRR_Sum.loc[i]['Iter'] = i + 1
    df_IRR_Sum.loc[i]['LS'] = df_IRR[i].loc[year]['LS']
    df_IRR_Sum.loc[i]['DCA'] = df_IRR[i].loc[year]['DCA']

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
print(df_IRR_Sum.loc[iter])