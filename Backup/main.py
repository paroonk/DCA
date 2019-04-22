import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
style.use('ggplot')

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
df_NAV = df_NAV.fillna(method='ffill').fillna(method='bfill')

pickle_in = open('Div.pickle', 'rb')
df_Div = pickle.load(pickle_in)
df_Div = df_Div.fillna(0).cumsum()

col = 8

df_all = pd.concat([df_NAV.iloc[:, 0:col], (df_NAV + df_Div).iloc[:, 0:col]], axis='columns', keys=['1', '2'])
df_all = df_all.swaplevel(axis='columns')[df_NAV.columns[:col]]

file_name = 'DCA'
writer = pd.ExcelWriter('{}.xlsx'.format(file_name.upper()))
workbook = writer.book
df_all.to_excel(writer, sheet_name='all', header=True)
writer.save()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
df_all.plot(ax=ax, title='NAV Trend')
plt.xlim(pd.datetime(2009, 1, 1), pd.datetime(2019, 1, 1))
plt.ylim(0)
plt.tight_layout()
plt.show()

df_first = df_NAV.resample('MS').first()
df_last = df_NAV.resample('M').last()
df_return = df_first.head(1).append(df_last)
df_return = df_return.loc[:, df_return.isnull().sum() == 0].pct_change().dropna() * 100

df_NAV = df_NAV + df_Div
df_first = df_NAV.resample('MS').first()
df_last = df_NAV.resample('M').last()
df_return_div = df_first.head(1).append(df_last)
df_return_div = df_return_div.loc[:, df_return_div.isnull().sum() == 0].pct_change().dropna() * 100
