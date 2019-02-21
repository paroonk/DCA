import numpy as np
import pandas as pd
import pickle

# file_name = 'Data'
# sheet_name = 'NAV'
# df_NAV = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
# df_NAV = df_NAV.set_index('Date')
# with open('NAV.pickle', 'wb') as f:
#     pickle.dump(df_NAV, f)
# file_name = 'Data'
# sheet_name = 'Div'
# df_Div = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
# df_Div = df_Div.set_index('Date')
# with open('Div.pickle', 'wb') as f:
#     pickle.dump(df_Div, f)

pickle_in = open('NAV.pickle', 'rb')
df_NAV = pickle.load(pickle_in)
pickle_in = open('Div.pickle', 'rb')
df_Div = pickle.load(pickle_in)

# check = df_NAV.loc[:, df_NAV.loc['2009-01'].count()>0]
check = df_NAV.loc['2009-01'].count()>0
print(check.sum())