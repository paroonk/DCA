from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tqdm
import xlsxwriter.utility
from matplotlib import style
from scipy.stats.mstats import gmean

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')
col_Transaction = ['Month', 'NAV', 'Bid Price', 'Offer Price', 'Required Value', 'Shares Bought', 'Shares Owned', 'Portfolio Value',
                   'DPS', 'Div After Tax', 'Total Cost', 'Average Cost', 'CFF', 'CFI', 'Net Cash', 'Net Wealth', 'RoR']
col_Simulation = ['NAV', 'LS', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']
row_Simulation = ['Last', 'Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']
col_Iter = ['Iter', 'Fund_Code', 'Fund_Name', 'Category_Morningstar',
               'NAV_Last', 'NAV_Mean', 'NAV_Std', 'NAV_SR',
               'Avg. Cost_LS', 'Avg. Cost_DCA', 'Avg. Cost_VA', 'Avg. Cost_VA6', 'Avg. Cost_VA12', 'Avg. Cost_VA18',
               'Mean_LS', 'Mean_DCA', 'Mean_VA', 'Mean_VA6', 'Mean_VA12', 'Mean_VA18',
               'Std_LS', 'Std_DCA', 'Std_VA', 'Std_VA6', 'Std_VA12', 'Std_VA18',
               'SR_LS', 'SR_DCA', 'SR_VA', 'SR_VA6', 'SR_VA12', 'SR_VA18',
               'IRR_LS', 'IRR_DCA', 'IRR_VA', 'IRR_VA6', 'IRR_VA12', 'IRR_VA18',
            'Dividend_LS', 'Dividend_DCA', 'Dividend_VA', 'Dividend_VA6', 'Dividend_VA12', 'Dividend_VA18']

# Simulation Config #
forecast_Year = 10
n_per_year = 12
init_Cash = 120000.0
income_Tax = 10
Div_ReInvest = True


def get_col_widths(df, index=True):
    if index:
        idx_max = max([len(str(s)) for s in df.index.values] + [len(str(df.index.name))])
        col_widths = [idx_max] + [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    else:
        col_widths = [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    return col_widths


def LS(df_NAV, df_Div, df_Data, forecast_year, init_Cash):
    global n_per_year
    global col_Transaction
    global income_Tax
    global Div_ReInvest
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_NAV)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        df.loc[t]['NAV'] = df_NAV.loc[t]
        df.loc[t]['Bid Price'] = np.floor(df.loc[t]['NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['Offer Price'] = np.ceil(df.loc[t]['NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['DPS'] = df_Div.loc[t]
        if t == 0:
            df.loc[t]['Shares Bought'] = init_Cash / df.loc[t]['Offer Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = 0.0
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
        elif t in range(1, forecast_year * n_per_year):
            if Div_ReInvest:
                df.loc[t]['Shares Bought'] = 0.0 if divmod(t, n_per_year)[1] != 0 else (init_Cash + df.loc[t - 1]['Net Cash']) / df.loc[t]['Offer Price']
            else:
                df.loc[t]['Shares Bought'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash / df.loc[t]['Offer Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFF'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI'] + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])
        elif t == forecast_year * n_per_year:
            df.loc[t]['Shares Bought'] = -df.loc[t - 1]['Shares Owned']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFF'] = 0.0
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / -df.loc[t]['Shares Bought']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def DCA(df_NAV, df_Div, df_Data, forecast_year, init_Cash):
    global n_per_year
    global col_Transaction
    global income_Tax
    global Div_ReInvest
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_NAV)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        df.loc[t]['NAV'] = df_NAV.loc[t]
        df.loc[t]['Bid Price'] = np.floor(df.loc[t]['NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['Offer Price'] = np.ceil(df.loc[t]['NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['DPS'] = df_Div.loc[t]
        if t == 0:
            df.loc[t]['Shares Bought'] = init_Cash / n_per_year / df.loc[t]['Offer Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = 0.0
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
        elif t in range(1, forecast_year * n_per_year):
            if Div_ReInvest and (divmod(t, n_per_year)[1] != 0):
                df.loc[t]['Shares Bought'] = df.loc[t - 1]['Net Cash'] / (n_per_year - divmod(t, n_per_year)[1]) / df.loc[t]['Offer Price']
            else:
                df.loc[t]['Shares Bought'] = init_Cash / n_per_year / df.loc[t]['Offer Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFF'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI'] + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])
        elif t == forecast_year * n_per_year:
            df.loc[t]['Shares Bought'] = -df.loc[t - 1]['Shares Owned']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFF'] = 0.0
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / -df.loc[t]['Shares Bought']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def VA(df_NAV, df_Div, df_Data, VA_Growth, forecast_year, init_Cash):
    global n_per_year
    global col_Transaction
    global income_Tax
    global Div_ReInvest
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_NAV)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        df.loc[t]['NAV'] = df_NAV.loc[t]
        df.loc[t]['Bid Price'] = np.floor(df.loc[t]['NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['Offer Price'] = np.ceil(df.loc[t]['NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['DPS'] = df_Div.loc[t]
        if t == 0:
            df.loc[t]['Required Value'] = init_Cash / n_per_year
            diff = df.loc[t]['Required Value']
            df.loc[t]['Shares Bought'] = (init_Cash / df.loc[t]['Offer Price']) if diff > init_Cash else (diff / df.loc[t]['Bid Price'])
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = 0.0
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
        elif t in range(1, forecast_year * n_per_year):
            df.loc[t]['Required Value'] = init_Cash / n_per_year + (df.loc[t - 1]['Required Value'] * (1 + VA_Growth / n_per_year / 100))
            diff = df.loc[t]['Required Value'] - (df.loc[t]['Bid Price'] * df.loc[t - 1]['Shares Owned'])
            df.loc[t]['Shares Bought'] = (df.loc[t - 1]['Net Cash'] / df.loc[t]['Offer Price']) if diff > df.loc[t - 1]['Net Cash'] else (diff / df.loc[t]['Bid Price'])
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFF'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI'] + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])
        elif t == forecast_year * n_per_year:
            df.loc[t]['Required Value'] = 0.0
            diff = df.loc[t]['Required Value'] - (df.loc[t]['Bid Price'] * df.loc[t - 1]['Shares Owned'])
            df.loc[t]['Shares Bought'] = (df.loc[t - 1]['Net Cash'] / df.loc[t]['Offer Price']) if diff > df.loc[t - 1]['Net Cash'] else (diff / df.loc[t]['Bid Price'])
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFF'] = 0.0
            df.loc[t]['CFI'] = (-(df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] > 0.0 else df.loc[t]['Bid Price'] if df.loc[t]['Shares Bought'] < 0.0 else 0.0) * df.loc[t]['Shares Bought']) + \
                               df.loc[t]['Div After Tax']
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / -df.loc[t]['Shares Bought']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def simulation(df_FundNAV, df_FundDiv, df_FundData, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Simulation
    global row_Simulation
    global col_Iter

    df_Simulation = {}
    df_Simulation['Summary'] = pd.DataFrame(columns=col_Simulation, index=row_Simulation)
    df_Summary = pd.DataFrame(columns=col_Summary)

    df_NAV = pd.DataFrame(df_FundNAV.iloc[:, iter])
    df_NAV.columns = ['NAV']
    df_NAV['RoR'] = df_NAV.pct_change()
    df_NAV.reset_index(drop=True, inplace=True)
    df_NAV.index.name = 'Month'
    df_Div = pd.DataFrame(df_FundDiv.iloc[:, iter])
    df_Div.columns = ['Div']
    df_Data = pd.DataFrame(df_FundData.iloc[iter, :])

    selectFund = '1VAL-D'
    writer = pd.ExcelWriter('output/Fund_Sim_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    body_fmt = {
        'B': float_fmt,
        'C': float_fmt,
        'D': float_fmt,
    }

    if df_Data.loc['Fund Code'].iloc[0] == selectFund:
        sheet_name = selectFund
        df = df_NAV.copy()
        df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 2, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    df_Simulation['LS'] = LS(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, forecast_year, init_Cash)
    df_Simulation['DCA'] = DCA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, forecast_year, init_Cash)
    df_Simulation['VA'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 0, forecast_year, init_Cash)
    df_Simulation['VA6'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 6, forecast_year, init_Cash)
    df_Simulation['VA12'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 12, forecast_year, init_Cash)
    df_Simulation['VA18'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 18, forecast_year, init_Cash)

    # Risk Free Rate 10Y = 1.8416, Risk Free Rate 5Y = 1.4760
    RiskFree = 1.8416 if forecast_year == 10 else 1.4760
    for row in row_Simulation:
        df_Simulation['Summary'].loc['Last', 'NAV'] = df_NAV['NAV'].iloc[-1]
        df_Simulation['Summary'].loc['Mean', 'NAV'] = df_NAV['RoR'].iloc[1:].mean() * n_per_year
        df_Simulation['Summary'].loc['Std', 'NAV'] = df_NAV['RoR'].iloc[1:].std(ddof=0) * np.sqrt(n_per_year)
        df_Simulation['Summary'].loc['SR', 'NAV'] = (df_Simulation['Summary'].loc['Mean', 'NAV'] - RiskFree / 100) / df_Simulation['Summary'].loc['Std', 'NAV']
    # for column in col_Simulation:
    for column in ['LS', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']:
        df_Simulation['Summary'].loc['Avg. Cost', column] = df_Simulation[column]['Average Cost'].iloc[-1]
        df_Simulation['Summary'].loc['Mean', column] = df_Simulation[column]['RoR'].iloc[1:].mean() * n_per_year
        df_Simulation['Summary'].loc['Std', column] = df_Simulation[column]['RoR'].iloc[1:].std(ddof=0) * np.sqrt(n_per_year)
        df_Simulation['Summary'].loc['SR', column] = (df_Simulation['Summary'].loc['Mean', column] - RiskFree / 100) / df_Simulation['Summary'].loc['Std', column]
        df_Simulation['Summary'].loc['IRR', column] = ((1 + np.irr(df_Simulation[column]['CFI'].tolist())) ** n_per_year) - 1
        df_Simulation['Summary'].loc['Dividend', column] = df_Simulation[column]['Div After Tax'].sum()
    df_Simulation['Summary'] = df_Simulation['Summary'].fillna('')

    if df_Data.loc['Fund Code'].iloc[0] == selectFund:
        body_fmt = {
            'B': float_fmt,
            'C': float_fmt,
            'D': float_fmt,
            'E': float_fmt,
            'F': float_fmt,
            'G': float_fmt,
            'H': float_fmt,
            'I': float_fmt,
            'J': float_fmt,
            'K': float_fmt,
            'L': float_fmt,
            'M': float_fmt,
            'N': float_fmt,
            'O': float_fmt,
            'P': float_fmt,
            'Q': pct_fmt,
        }
        for Algo in ['LS', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']:
            sheet_name = '{}'.format(Algo)
            df = df_Simulation[Algo].copy()
            df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
            df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 2, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    body_fmt = {
        '2': float_fmt,
        '3': float_fmt,
        '4': float_fmt,
        '5': pct_fmt,
        '6': pct_fmt,
        '7': float2_fmt,
        '8': pct_fmt,
    }
    if df_Data.loc['Fund Code'].iloc[0] == selectFund:
        sheet_name = 'Summary'
        df = df_Simulation['Summary'].copy()
        df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for row in range(df.shape[0]):
            worksheet.set_row(row + 1, None, body_fmt[str(row + 2)])
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 4)
        writer.save()

    # Summary of IRR #
    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary['Iter'] = int(iter + 1)
    df_Summary['Fund_Code'] = df_FundData.loc[df_FundNAV.columns[iter], 'Fund Code']
    df_Summary['Fund_Name'] = df_FundData.loc[df_FundNAV.columns[iter], 'Local Name - Thai']
    df_Summary['Category_Morningstar'] = df_FundData.loc[df_FundNAV.columns[iter], 'Morningstar Category']
    for row in ['Last', 'Mean', 'Std', 'SR']:
        df_Summary['NAV_{}'.format(row)] = df_Simulation['Summary'].loc[row, 'NAV']
    for column in ['LS', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']:
        for row in ['Dividend', 'Avg. Cost', 'Mean', 'Std', 'SR', 'IRR']:
            df_Summary['{}_{}'.format(row, column)] = df_Simulation['Summary'].loc[row, column]

    return df_Summary.values.tolist()


if __name__ == '__main__':

    # Excel to Pickle #
    # df_FundNAV = pd.read_excel('data/Fund.xlsx', sheet_name='NAV').set_index('Date').replace(' ', np.nan)
    # df_FundNAV.to_pickle('data/FundNAV.pkl')
    # df_FundDiv = pd.read_excel('data/Fund.xlsx', sheet_name='Div').set_index('Date').replace(' ', np.nan)
    # df_FundDiv.to_pickle('data/FundDiv.pkl')
    # df_FundData = pd.read_excel('data/Fund.xlsx', sheet_name='Data').set_index('SecId')
    # df_FundData.to_pickle('data/FundData.pkl')

    # Import Pickle #
    df_FundNAV = pd.read_pickle('data/FundNAV.pkl')
    df_FundDiv = pd.read_pickle('data/FundDiv.pkl')
    df_FundData = pd.read_pickle('data/FundData.pkl')

    # Filtering Funds #
    FundType = ['Thailand Fund Equity Small/Mid-Cap', 'Thailand Fund Equity Large-Cap']
    df_FundNAV = df_FundNAV.loc[:, df_FundData['Morningstar Category'].isin(FundType).tolist()]
    df_FundNAV = df_FundNAV.loc[:, df_FundNAV.count() >= forecast_Year * n_per_year + 1]
    df_FundNAV = df_FundNAV.iloc[:forecast_Year * n_per_year + 1].sort_index()
    # todo Test only 10 funds
    # df_FundNAV = df_FundNAV.iloc[:, 0:10]

    df_FundDiv = df_FundDiv.loc[df_FundNAV.index, df_FundNAV.columns].fillna(0)
    df_FundData = df_FundData.loc[df_FundNAV.columns, :]

    results = []
    pool = Pool()
    iter = df_FundNAV.shape[1]
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, df_FundNAV, df_FundDiv, df_FundData, forecast_Year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_Summary = pd.DataFrame(results, columns=col_Iter, dtype='object')
    df_Summary.sort_values(by='Iter', inplace=True)

    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary.iloc[-1]['Iter'] = 'Avg'
    df_Summary.iloc[-1, 4:] = df_Summary.iloc[0:-1, 4:].mean()
    df_Summary = df_Summary.fillna('')
    df_Summary = df_Summary.set_index('Iter')
    df_Summary.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], col.split('_')[-1]) for col in df_Summary.columns])
    print(df_Summary.drop(columns=['Name'], level=1))

    writer = pd.ExcelWriter('output/Fund_Sum_{}Y_{}.xlsx'.format(forecast_Year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    text_fmt = workbook.add_format({'align': 'left'})

    sheet_name = 'Summary'
    df = df_Summary.copy()
    df = df.applymap(lambda x: round(x, 6) if isinstance(x, (int, float)) else x)
    df.to_excel(writer, sheet_name=sheet_name)
    worksheet = writer.sheets[sheet_name]
    body_fmt = {
        'B': text_fmt,
        'C': text_fmt,
        'D': text_fmt,
        'E': float_fmt,
        'F': pct_fmt,
        'G': pct_fmt,
        'H': float_fmt,
        'I': float_fmt,
        'J': float_fmt,
        'K': float_fmt,
        'L': float_fmt,
        'M': float_fmt,
        'N': float_fmt,
        'O': float_fmt,
        'P': float_fmt,
        'Q': float_fmt,
        'R': float_fmt,
        'S': float_fmt,
        'T': float_fmt,
        'U': pct_fmt,
        'V': pct_fmt,
        'W': pct_fmt,
        'X': pct_fmt,
        'Y': pct_fmt,
        'Z': pct_fmt,
        'AA': pct_fmt,
        'AB': pct_fmt,
        'AC': pct_fmt,
        'AD': pct_fmt,
        'AE': pct_fmt,
        'AF': pct_fmt,
        'AG': float2_fmt,
        'AH': float2_fmt,
        'AI': float2_fmt,
        'AJ': float2_fmt,
        'AK': float2_fmt,
        'AL': float2_fmt,
        'AM': pct_fmt,
        'AN': pct_fmt,
        'AO': pct_fmt,
        'AP': pct_fmt,
        'AQ': pct_fmt,
        'AR': pct_fmt,
    }
    for col, width in enumerate(get_col_widths(df, index=False), 1):
        worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
    writer.save()
