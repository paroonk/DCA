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
pd.options.display.float_format = '{:.4f}'.format
style.use('ggplot')
np.random.seed(None)

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# boot = resample(x, replace=True, n_samples=100000, random_state=None)
# print(boot.mean())

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

iter = 10

init_S = 449.96
u = 0.138
sigma = 0.158
n_per_year = 12
dt = 1 / n_per_year
forecast_period = 10 * n_per_year

forecast = {}
for i in range(iter):
    forecast[i] = pd.DataFrame(
        columns=['Period', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S'])
    forecast[i] = forecast[i].append({'Period': 0, 'S': init_S}, ignore_index=True)
    for t in range(1, forecast_period + 1):
        forecast[i] = forecast[i].append({}, ignore_index=True)
        forecast[i].loc[t]['Period'] = t
        forecast[i].loc[t]['u.dt'] = u * dt
        forecast[i].loc[t]['S(u.dt)'] = forecast[i].loc[t - 1]['S'] * forecast[i].loc[t]['u.dt']
        forecast[i].loc[t]['N'] = np.random.normal()
        forecast[i].loc[t]['N.sigma.sqrt(dt)'] = forecast[i].loc[t]['N'] * sigma * np.sqrt(dt)
        forecast[i].loc[t]['S(N.sigma.sqrt(dt))'] = forecast[i].loc[t - 1]['S'] * forecast[i].loc[t]['N.sigma.sqrt(dt)']
        forecast[i].loc[t]['dS'] = forecast[i].loc[t]['S(u.dt)'] + forecast[i].loc[t]['S(N.sigma.sqrt(dt))']
        forecast[i].loc[t]['S'] = forecast[i].loc[t - 1]['S'] + forecast[i].loc[t]['dS']
    forecast[i] = forecast[i].fillna('')
    forecast[i]['Period'] = forecast[i]['Period'].astype('int')
    forecast[i] = forecast[i].set_index('Period')
    if i in np.arange(0, iter, iter/100):
        forecast[i].plot(y='S', kind='line', ax=ax[0])
    print('Iter: {}/{}'.format(i + 1, iter))

n_bin = 15
S_dist = []
for i in range(len(forecast)):
    S_dist.append(forecast[i].loc[forecast_period]['S'])
hmin, hmax = math.floor(min(S_dist)), math.ceil(max(S_dist))
gap = math.ceil((hmax - hmin) / n_bin)
bins = np.linspace(hmin, hmax, n_bin + 1)
ax[1].hist(S_dist, bins, ec='k', alpha=0.8)
ax[1].axvline(np.percentile(S_dist, 25), color='k', linestyle='dashed', linewidth=1)
ax[1].axvline(np.percentile(S_dist, 50), color='k', linestyle='dashed', linewidth=1)
ax[1].axvline(np.percentile(S_dist, 75), color='k', linestyle='dashed', linewidth=1)
x0, x1 = ax[1].get_xlim()
y0, y1 = ax[1].get_ylim()
ax[1].text(np.percentile(S_dist, 25) + 0.01 * (x1 - x0), y0 + 0.90 * (y1 - y0), 'P25: {:.4f}'.format(np.percentile(S_dist, 25)), color='blue')
ax[1].text(np.percentile(S_dist, 50) + 0.01 * (x1 - x0), y0 + 0.85 * (y1 - y0), 'Med: {:.4f}'.format(np.percentile(S_dist, 50)), color='blue')
ax[1].text(np.percentile(S_dist, 75) + 0.01 * (x1 - x0), y0 + 0.80 * (y1 - y0), 'P75: {:.4f}'.format(np.percentile(S_dist, 75)), color='blue')

fig.suptitle('Forecast')
ax[0].set_xlabel('Period', fontsize=10)
ax[0].set_ylabel('S', fontsize=10)
ax[0].legend().set_visible(False)
ax[1].set_xticks(bins)
ax[1].set_xlabel('S@Period = {}'.format(forecast_period), fontsize=10)
ax[1].set_ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.subplots_adjust(top=0.92, left=0.08, right=0.92, hspace=0.25)
plt.show()
