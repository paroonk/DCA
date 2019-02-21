import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.ticker import FormatStrFormatter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
style.use('seaborn')

# file_name = 'Data'
# sheet_name = 'NAV'
# df_NAV = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
# df_NAV = df_NAV.set_index('Date').sort_index()
# with open('NAV.pickle', 'wb') as f:
#     pickle.dump(df_NAV, f)
# file_name = 'Data'
# sheet_name = 'Div'
# df_Div = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
# df_Div = df_Div.set_index('Date').sort_index()
# with open('Div.pickle', 'wb') as f:
#     pickle.dump(df_Div, f)

pickle_in = open('NAV.pickle', 'rb')
df_NAV = pickle.load(pickle_in)
pickle_in = open('Div.pickle', 'rb')
df_Div = pickle.load(pickle_in)

df_first = df_NAV.resample('MS').first()
df_last = df_NAV.resample('M').last()
df_return = df_first.head(1).append(df_last)
df_return = df_return.loc[:, df_return.isnull().sum()==0].pct_change().dropna()*100

df_Div = df_Div.fillna(0).cumsum()
df_NAV = df_NAV + df_Div
df_first = df_NAV.resample('MS').first()
df_last = df_NAV.resample('M').last()
df_return_div = df_first.head(1).append(df_last)
df_return_div = df_return_div.loc[:, df_return_div.isnull().sum()==0].pct_change().dropna()*100

data = [df_return.mean(), df_return_div.mean()]
fig, ax = plt.subplots(2, sharex=True)
title = ['Expected monthly return (%)', 'Expected monthly return include dividend (%)']
counts, bins, patches = {}, {}, {}
for i in range(len(data)):
    counts[i], bins[i], patches[i] = ax[i].hist(data[i], bins=20, facecolor='yellow', edgecolor='black')

    ax[i].set_xticks(bins[i])
    ax[i].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[i].xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    twentyfifth, seventyfifth = np.percentile(data[i], [25, 75])
    for patch, rightside, leftside in zip(patches[i], bins[i][1:], bins[i][:-1]):
        if rightside < twentyfifth:
            patch.set_facecolor('red')
        elif leftside > seventyfifth:
            patch.set_facecolor('green')

    bin_centers = 0.5 * np.diff(bins[i]) + bins[i][:-1]
    for count, x in zip(counts[i], bin_centers):
        ax[i].annotate('{:.0f}'.format(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        percent = '{:.1f}%'.format(100 * float(count) / counts[i].sum())
        ax[i].annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points', va='top', ha='center')

    ax[i].set_title('{} [Count = {:.0f}, Mean = {:.2f} SD = {:.2f}]'.format(title[i], data[i].count(), data[i].mean(), data[i].std()))

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, bottom=0.1)
plt.show()