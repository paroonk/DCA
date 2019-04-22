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

iter = 1
n_per_year = 12
dt = 1 / n_per_year
forecast_period = 10 * n_per_year

init_S = 100
u = 0.1
sigma = 0.15

init_Cash = 120000.0

forecast = {}
LS = {}
DCA = {}
for i in range(iter):
    forecast[i] = pd.DataFrame(
        columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S'])
    LS[i] = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Beg. Inv.Asset Value', 'Change in Value of Inv.Asset', 'Net Inv.Asset Value',
                 'Deposit', 'Beg. Cash', 'Change in Cash', 'Net Cash',
                 'Total Wealth', 'Total Deposit', 'Profit/Loss'])
    DCA[i] = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Beg. Inv.Asset Value', 'Change in Value of Inv.Asset', 'Net Inv.Asset Value',
                 'Deposit', 'Beg. Cash', 'Change in Cash', 'Net Cash',
                 'Total Wealth', 'Total Deposit', 'Profit/Loss'])

    for t in range(0, forecast_period + 1):
        forecast[i] = forecast[i].append({}, ignore_index=True)
        forecast[i].loc[t]['Month'] = t
        LS[i] = LS[i].append({}, ignore_index=True)
        LS[i].loc[t]['Month'] = t
        DCA[i] = DCA[i].append({}, ignore_index=True)
        DCA[i].loc[t]['Month'] = t

        if t == 0:
            forecast[i].loc[t]['S'] = init_S
        elif t > 0 and forecast_period + 1:
            forecast[i].loc[t]['u.dt'] = u * dt
            forecast[i].loc[t]['S(u.dt)'] = forecast[i].loc[t - 1]['S'] * forecast[i].loc[t]['u.dt']
            forecast[i].loc[t]['N'] = np.random.normal()
            forecast[i].loc[t]['N.sigma.sqrt(dt)'] = forecast[i].loc[t]['N'] * sigma * np.sqrt(dt)
            forecast[i].loc[t]['S(N.sigma.sqrt(dt))'] = forecast[i].loc[t - 1]['S'] * forecast[i].loc[t]['N.sigma.sqrt(dt)']
            forecast[i].loc[t]['dS'] = forecast[i].loc[t]['S(u.dt)'] + forecast[i].loc[t]['S(N.sigma.sqrt(dt))']
            forecast[i].loc[t]['S'] = forecast[i].loc[t - 1]['S'] + forecast[i].loc[t]['dS']

        if t == 0:
            LS[i].loc[t]['Beg. Inv.Asset Volume'] = 0.0
            LS[i].loc[t]['Buy/Sell Inv.Asset Volume'] = (init_Cash if divmod(t, 12)[1] == 0.0 else 0.0) / forecast[i].loc[t]['S']
            LS[i].loc[t]['Net Inv.Asset Volume'] = LS[i].loc[t]['Beg. Inv.Asset Volume'] + LS[i].loc[t]['Buy/Sell Inv.Asset Volume']
            LS[i].loc[t]['Inv.Asset Price'] = forecast[i].loc[t]['S']
            LS[i].loc[t]['Beg. Inv.Asset Value'] = LS[i].loc[t]['Beg. Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Change in Value of Inv.Asset'] = LS[i].loc[t]['Buy/Sell Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Net Inv.Asset Value'] = LS[i].loc[t]['Net Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Deposit'] = init_Cash if divmod(t, 12)[1] == 0.0 else 0.0
            LS[i].loc[t]['Beg. Cash'] = 0.0 + LS[i].loc[t]['Deposit']
            # todo interest not included yet
            LS[i].loc[t]['Change in Cash'] = -LS[i].loc[t]['Change in Value of Inv.Asset'] if LS[i].loc[t]['Change in Value of Inv.Asset'] != 0.0 else 0.0
            LS[i].loc[t]['Net Cash'] = LS[i].loc[t]['Beg. Cash'] + LS[i].loc[t]['Change in Cash']
            LS[i].loc[t]['Total Wealth'] = LS[i].loc[t]['Net Inv.Asset Value'] + LS[i].loc[t]['Net Cash']
            LS[i].loc[t]['Total Deposit'] = 0.0 + LS[i].loc[t]['Deposit']
            LS[i].loc[t]['Profit/Loss'] = LS[i].loc[t]['Total Wealth'] - LS[i].loc[t]['Total Deposit']
        elif t in range(1, forecast_period):
            LS[i].loc[t]['Beg. Inv.Asset Volume'] = LS[i].loc[t - 1]['Net Inv.Asset Volume']
            LS[i].loc[t]['Buy/Sell Inv.Asset Volume'] = (init_Cash if divmod(t, 12)[1] == 0.0 else 0.0) / forecast[i].loc[t]['S']
            LS[i].loc[t]['Net Inv.Asset Volume'] = LS[i].loc[t]['Beg. Inv.Asset Volume'] + LS[i].loc[t]['Buy/Sell Inv.Asset Volume']
            LS[i].loc[t]['Inv.Asset Price'] = forecast[i].loc[t]['S']
            LS[i].loc[t]['Beg. Inv.Asset Value'] = LS[i].loc[t]['Beg. Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Change in Value of Inv.Asset'] = LS[i].loc[t]['Buy/Sell Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Net Inv.Asset Value'] = LS[i].loc[t]['Net Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Deposit'] = init_Cash if divmod(t, 12)[1] == 0.0 else 0.0
            LS[i].loc[t]['Beg. Cash'] = LS[i].loc[t - 1]['Net Cash'] + LS[i].loc[t]['Deposit']
            LS[i].loc[t]['Change in Cash'] = -LS[i].loc[t]['Change in Value of Inv.Asset'] if LS[i].loc[t]['Change in Value of Inv.Asset'] != 0.0 else 0.0
            LS[i].loc[t]['Net Cash'] = LS[i].loc[t]['Beg. Cash'] + LS[i].loc[t]['Change in Cash']
            LS[i].loc[t]['Total Wealth'] = LS[i].loc[t]['Net Inv.Asset Value'] + LS[i].loc[t]['Net Cash']
            LS[i].loc[t]['Total Deposit'] = LS[i].loc[t - 1]['Total Deposit'] + LS[i].loc[t]['Deposit']
            LS[i].loc[t]['Profit/Loss'] = LS[i].loc[t]['Total Wealth'] - LS[i].loc[t]['Total Deposit']
        elif t == forecast_period:
            LS[i].loc[t]['Beg. Inv.Asset Volume'] = LS[i].loc[t - 1]['Net Inv.Asset Volume']
            LS[i].loc[t]['Buy/Sell Inv.Asset Volume'] = 0.0
            LS[i].loc[t]['Net Inv.Asset Volume'] = LS[i].loc[t]['Beg. Inv.Asset Volume'] + LS[i].loc[t]['Buy/Sell Inv.Asset Volume']
            LS[i].loc[t]['Inv.Asset Price'] = forecast[i].loc[t]['S']
            LS[i].loc[t]['Beg. Inv.Asset Value'] = LS[i].loc[t]['Beg. Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Change in Value of Inv.Asset'] = LS[i].loc[t]['Buy/Sell Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Net Inv.Asset Value'] = LS[i].loc[t]['Net Inv.Asset Volume'] * LS[i].loc[t]['Inv.Asset Price']
            LS[i].loc[t]['Deposit'] = 0.0
            LS[i].loc[t]['Beg. Cash'] = LS[i].loc[t - 1]['Net Cash'] + LS[i].loc[t]['Deposit']
            LS[i].loc[t]['Change in Cash'] = -LS[i].loc[t]['Change in Value of Inv.Asset'] if LS[i].loc[t]['Change in Value of Inv.Asset'] != 0.0 else 0.0
            LS[i].loc[t]['Net Cash'] = LS[i].loc[t]['Beg. Cash'] + LS[i].loc[t]['Change in Cash']
            LS[i].loc[t]['Total Wealth'] = LS[i].loc[t]['Net Inv.Asset Value'] + LS[i].loc[t]['Net Cash']
            LS[i].loc[t]['Total Deposit'] = LS[i].loc[t - 1]['Total Deposit'] + LS[i].loc[t]['Deposit']
            LS[i].loc[t]['Profit/Loss'] = LS[i].loc[t]['Total Wealth'] - LS[i].loc[t]['Total Deposit']

        if t == 0:
            DCA[i].loc[t]['Beg. Inv.Asset Volume'] = 0.0
            DCA[i].loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / 12 / forecast[i].loc[t]['S']
            DCA[i].loc[t]['Net Inv.Asset Volume'] = DCA[i].loc[t]['Beg. Inv.Asset Volume'] + DCA[i].loc[t]['Buy/Sell Inv.Asset Volume']
            DCA[i].loc[t]['Inv.Asset Price'] = forecast[i].loc[t]['S']
            DCA[i].loc[t]['Beg. Inv.Asset Value'] = DCA[i].loc[t]['Beg. Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Change in Value of Inv.Asset'] = DCA[i].loc[t]['Buy/Sell Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Net Inv.Asset Value'] = DCA[i].loc[t]['Net Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Deposit'] = init_Cash if divmod(t, 12)[1] == 0.0 else 0.0
            DCA[i].loc[t]['Beg. Cash'] = 0.0 + DCA[i].loc[t]['Deposit']
            # todo interest not included yet
            DCA[i].loc[t]['Change in Cash'] = -DCA[i].loc[t]['Change in Value of Inv.Asset'] if DCA[i].loc[t]['Change in Value of Inv.Asset'] != 0.0 else 0.0
            DCA[i].loc[t]['Net Cash'] = DCA[i].loc[t]['Beg. Cash'] + DCA[i].loc[t]['Change in Cash']
            DCA[i].loc[t]['Total Wealth'] = DCA[i].loc[t]['Net Inv.Asset Value'] + DCA[i].loc[t]['Net Cash']
            DCA[i].loc[t]['Total Deposit'] = 0.0 + DCA[i].loc[t]['Deposit']
            DCA[i].loc[t]['Profit/Loss'] = DCA[i].loc[t]['Total Wealth'] - DCA[i].loc[t]['Total Deposit']
        elif t in range(1, forecast_period):
            DCA[i].loc[t]['Beg. Inv.Asset Volume'] = DCA[i].loc[t - 1]['Net Inv.Asset Volume']
            DCA[i].loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / 12 / forecast[i].loc[t]['S']
            DCA[i].loc[t]['Net Inv.Asset Volume'] = DCA[i].loc[t]['Beg. Inv.Asset Volume'] + DCA[i].loc[t]['Buy/Sell Inv.Asset Volume']
            DCA[i].loc[t]['Inv.Asset Price'] = forecast[i].loc[t]['S']
            DCA[i].loc[t]['Beg. Inv.Asset Value'] = DCA[i].loc[t]['Beg. Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Change in Value of Inv.Asset'] = DCA[i].loc[t]['Buy/Sell Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Net Inv.Asset Value'] = DCA[i].loc[t]['Net Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Deposit'] = init_Cash if divmod(t, 12)[1] == 0.0 else 0.0
            DCA[i].loc[t]['Beg. Cash'] = DCA[i].loc[t - 1]['Net Cash'] + DCA[i].loc[t]['Deposit']
            DCA[i].loc[t]['Change in Cash'] = -DCA[i].loc[t]['Change in Value of Inv.Asset'] if DCA[i].loc[t]['Change in Value of Inv.Asset'] != 0.0 else 0.0
            DCA[i].loc[t]['Net Cash'] = DCA[i].loc[t]['Beg. Cash'] + DCA[i].loc[t]['Change in Cash']
            DCA[i].loc[t]['Total Wealth'] = DCA[i].loc[t]['Net Inv.Asset Value'] + DCA[i].loc[t]['Net Cash']
            DCA[i].loc[t]['Total Deposit'] = DCA[i].loc[t - 1]['Total Deposit'] + DCA[i].loc[t]['Deposit']
            DCA[i].loc[t]['Profit/Loss'] = DCA[i].loc[t]['Total Wealth'] - DCA[i].loc[t]['Total Deposit']
        elif t == forecast_period:
            DCA[i].loc[t]['Beg. Inv.Asset Volume'] = DCA[i].loc[t - 1]['Net Inv.Asset Volume']
            DCA[i].loc[t]['Buy/Sell Inv.Asset Volume'] = 0.0
            DCA[i].loc[t]['Net Inv.Asset Volume'] = DCA[i].loc[t]['Beg. Inv.Asset Volume'] + DCA[i].loc[t]['Buy/Sell Inv.Asset Volume']
            DCA[i].loc[t]['Inv.Asset Price'] = forecast[i].loc[t]['S']
            DCA[i].loc[t]['Beg. Inv.Asset Value'] = DCA[i].loc[t]['Beg. Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Change in Value of Inv.Asset'] = DCA[i].loc[t]['Buy/Sell Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Net Inv.Asset Value'] = DCA[i].loc[t]['Net Inv.Asset Volume'] * DCA[i].loc[t]['Inv.Asset Price']
            DCA[i].loc[t]['Deposit'] = 0.0
            DCA[i].loc[t]['Beg. Cash'] = DCA[i].loc[t - 1]['Net Cash'] + DCA[i].loc[t]['Deposit']
            DCA[i].loc[t]['Change in Cash'] = -DCA[i].loc[t]['Change in Value of Inv.Asset'] if DCA[i].loc[t]['Change in Value of Inv.Asset'] != 0.0 else 0.0
            DCA[i].loc[t]['Net Cash'] = DCA[i].loc[t]['Beg. Cash'] + DCA[i].loc[t]['Change in Cash']
            DCA[i].loc[t]['Total Wealth'] = DCA[i].loc[t]['Net Inv.Asset Value'] + DCA[i].loc[t]['Net Cash']
            DCA[i].loc[t]['Total Deposit'] = DCA[i].loc[t - 1]['Total Deposit'] + DCA[i].loc[t]['Deposit']
            DCA[i].loc[t]['Profit/Loss'] = DCA[i].loc[t]['Total Wealth'] - DCA[i].loc[t]['Total Deposit']

    forecast[i] = forecast[i].fillna('')
    forecast[i]['Month'] = forecast[i]['Month'].astype('int')
    forecast[i] = forecast[i].set_index('Month')
    if i in np.arange(0, iter, iter / 100):
        forecast[i].plot(y='S', kind='line', ax=ax[0])

    LS[i] = LS[i].fillna('')
    LS[i]['Month'] = LS[i]['Month'].astype('int')
    LS[i] = LS[i].set_index('Month')
    print(LS[i])

    DCA[i] = DCA[i].fillna('')
    DCA[i]['Month'] = DCA[i]['Month'].astype('int')
    DCA[i] = DCA[i].set_index('Month')
    print(DCA[i])

    # print('Iter: {}/{}'.format(i + 1, iter))

n_bin = 15
S_dist = []
for i in range(len(forecast)):
    S_dist.append(forecast[i].loc[forecast_period]['S'])
hmin, hmax = math.floor(min(S_dist)), math.ceil(max(S_dist))
gap = math.ceil((hmax - hmin) / n_bin)
bins = np.linspace(hmin, hmax, n_bin + 1)
ax[1].hist(S_dist, bins, ec='k', alpha=0.8)
ax[1].axvline(np.percentile(S_dist, 5), color='grey', linestyle='dashed', linewidth=1)
ax[1].axvline(np.percentile(S_dist, 25), color='grey', linestyle='dashed', linewidth=1)
ax[1].axvline(np.percentile(S_dist, 50), color='grey', linestyle='dashed', linewidth=1)
ax[1].axvline(np.percentile(S_dist, 75), color='grey', linestyle='dashed', linewidth=1)
ax[1].axvline(np.percentile(S_dist, 95), color='grey', linestyle='dashed', linewidth=1)
x0, x1 = ax[1].get_xlim()
y0, y1 = ax[1].get_ylim()
ax[1].text(np.percentile(S_dist, 5) + 0.01 * (x1 - x0), y0 + 0.90 * (y1 - y0), 'P5: {:.4f}'.format(np.percentile(S_dist, 5)), color='blue')
ax[1].text(np.percentile(S_dist, 25) + 0.01 * (x1 - x0), y0 + 0.85 * (y1 - y0), 'P25: {:.4f}'.format(np.percentile(S_dist, 25)), color='blue')
ax[1].text(np.percentile(S_dist, 50) + 0.01 * (x1 - x0), y0 + 0.90 * (y1 - y0), 'Med: {:.4f}'.format(np.percentile(S_dist, 50)), color='blue')
ax[1].text(np.percentile(S_dist, 75) + 0.01 * (x1 - x0), y0 + 0.85 * (y1 - y0), 'P75: {:.4f}'.format(np.percentile(S_dist, 75)), color='blue')
ax[1].text(np.percentile(S_dist, 95) + 0.01 * (x1 - x0), y0 + 0.90 * (y1 - y0), 'P95: {:.4f}'.format(np.percentile(S_dist, 95)), color='blue')

fig.suptitle('Forecast')
ax[0].set_xlabel('Month', fontsize=10)
ax[0].set_ylabel('S', fontsize=10)
ax[0].legend().set_visible(False)
ax[1].set_xticks(bins)
ax[1].set_xlabel('S@Month={}'.format(forecast_period), fontsize=10)
ax[1].set_ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.subplots_adjust(top=0.92, left=0.08, right=0.92, hspace=0.25)
plt.show()
