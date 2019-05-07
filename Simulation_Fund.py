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
n_per_year = 12
col_Transaction = ['Month', 'Beg. Fund Volume', 'Buy/Sell Fund Volume', 'Net Fund Volume',
                   'Fund NAV', 'Fund Bid Price', 'Fund Offer Price', 'Capital Gain', 'Beg. Fund Value', 'Change in Fund Value', 'Net Fund Value',
                   'Beg. Cash', 'Change in Cash', 'Dividend Gain', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR']
col_Simulation = ['Year', 'NAV_Last', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'IRR_LS', 'IRR_DCA', 'IRR_VA']
col_Summary = ['Iter', 'Fund_Code', 'Fund_Name', 'Category_GlobalBroad', 'Category_Morningstar', 'NAV_Last', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'IRR_LS', 'IRR_DCA', 'IRR_VA']

# Simulation Config #
forecast_year = 10
init_Cash = 120000.0


def get_col_widths(df, index=True):
    if index:
        idx_max = max([len(str(s)) for s in df.index.values] + [len(str(df.index.name))])
        col_widths = [idx_max] + [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    else:
        col_widths = [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    return col_widths


def LS(df_NAV_Y, df_Div_Y, df_Data, init_Cash):
    global n_per_year
    global col_Transaction
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = 0.0
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash
            df.loc[t]['Buy/Sell Fund Volume'] = init_Cash / df.loc[t]['Fund Offer Price']
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = 0.0
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = -df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.4%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def DCA(df_NAV_Y, df_Div_Y, df_Data, init_Cash):
    global n_per_year
    global col_Transaction
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = 0.0
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash
            df.loc[t]['Buy/Sell Fund Volume'] = init_Cash / n_per_year / df_NAV_Y.loc[t]
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = init_Cash / n_per_year / df_NAV_Y.loc[t]
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = -df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.4%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def VA(df_NAV_Y, df_Div_Y, df_Data, init_Cash):
    global n_per_year
    global col_Transaction
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = 0.0
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash
            diff = ((t + 1) * init_Cash / n_per_year) - df.loc[t]['Beg. Fund Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = diff / df_NAV_Y.loc[t]
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            diff = ((t + 1) * init_Cash / n_per_year) - df.loc[t]['Beg. Fund Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = diff / df_NAV_Y.loc[t]
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = -df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Change in Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Gain'] = df_Div_Y.loc[t] * df.loc[t]['Beg. Fund Volume'] * 0.9
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.4%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def simulation(df_FundNAV, df_FundDiv, df_FundData, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Simulation
    global col_Summary
    df_Simulation = pd.DataFrame(columns=col_Simulation)
    df_Summary = pd.DataFrame(columns=col_Summary)
    df_LS = {}
    df_DCA = {}
    df_VA = {}

    df_Price = pd.DataFrame(df_FundNAV.iloc[:, iter])
    df_Price.columns = ['S']
    df_Price['RR'] = df_Price.pct_change()
    df_Price.reset_index(drop=True, inplace=True)
    df_Price.index.name = 'Month'
    df_Div = pd.DataFrame(df_FundDiv.iloc[:, iter])
    df_Div.columns = ['Div']
    df_Data = pd.DataFrame(df_FundData.iloc[iter, :])

    writer = pd.ExcelWriter('output/Fund_Simulation_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    body_fmt = {
        'B': float_fmt,
        'C': float_fmt,
        'D': float_fmt,
    }

    for year in range(forecast_year):
        df_NAV_Y = df_Price.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True)
        df_Div_Y = df_Div.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['Div'].reset_index(drop=True)
        df_LS[year] = LS(df_NAV_Y, df_Div_Y, df_Data, init_Cash)
        df_DCA[year] = DCA(df_NAV_Y, df_Div_Y, df_Data, init_Cash)
        df_VA[year] = VA(df_NAV_Y, df_Div_Y, df_Data, init_Cash)
        df_Simulation = df_Simulation.append({}, ignore_index=True)
        df_Simulation.loc[year]['Year'] = year + 1
        df_Simulation.loc[year]['IRR_LS'] = df_LS[year].loc[n_per_year]['IRR']
        df_Simulation.loc[year]['IRR_DCA'] = df_DCA[year].loc[n_per_year]['IRR']
        df_Simulation.loc[year]['IRR_VA'] = df_VA[year].loc[n_per_year]['IRR']

        if df_Data.loc['Fund Code'].iloc[0] == '1S-LTF' and year == 0:
            sheet_name = 'Stock'
            df = df_Price.copy()
            df.loc[0, 'S'] = df.loc[0, 'S'].astype(float).round(4)
            df.loc[1:] = df.loc[1:].astype(float).round(4)
            df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

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
                'Q': float_fmt,
                'R': pct_fmt,
            }
            sheet_name = 'LS'
            df = df_LS[year].copy()
            df.loc[n_per_year, 'IRR'] = round(float(df.loc[n_per_year, 'IRR'].rstrip('%')) / 100.0, 4)
            df = df.round(4)
            df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

            sheet_name = 'DCA'
            df = df_DCA[year].copy()
            df.loc[n_per_year, 'IRR'] = round(float(df.loc[n_per_year, 'IRR'].rstrip('%')) / 100.0, 4)
            df = df.round(4)
            df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

            sheet_name = 'VA'
            df = df_VA[year].copy()
            df.loc[n_per_year, 'IRR'] = round(float(df.loc[n_per_year, 'IRR'].rstrip('%')) / 100.0, 4)
            df = df.round(4)
            df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    df_Simulation = df_Simulation.append({}, ignore_index=True)
    df_Simulation.loc[forecast_year]['Year'] = 'Avg'
    df_Simulation.loc[forecast_year]['NAV_Last'] = df_Price.iloc[-1]['S']
    df_Simulation.loc[forecast_year]['RR_Mean'] = '{:.4%}'.format(df_Price.iloc[1:]['RR'].mean() * n_per_year)
    df_Simulation.loc[forecast_year]['RR_Std'] = '{:.4%}'.format(df_Price.iloc[1:]['RR'].std() * np.sqrt(n_per_year))
    df_Simulation.loc[forecast_year]['RR_Skew'] = df_Price.iloc[1:]['RR'].skew()
    df_Simulation.loc[forecast_year]['RR_Kurt'] = df_Price.iloc[1:]['RR'].kurt()
    df_Simulation.loc[forecast_year]['IRR_LS'] = '{:.4%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_LS'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation.loc[forecast_year]['IRR_DCA'] = '{:.4%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_DCA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation.loc[forecast_year]['IRR_VA'] = '{:.4%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_VA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation = df_Simulation.fillna('')
    df_Simulation = df_Simulation.set_index('Year')

    if df_Data.loc['Fund Code'].iloc[0] == '1S-LTF':
        body_fmt = {
            'B': float_fmt,
            'C': pct_fmt,
            'D': pct_fmt,
            'E': float_fmt,
            'F': float_fmt,
            'G': pct_fmt,
            'H': pct_fmt,
            'I': pct_fmt,
        }
        sheet_name = 'Summary'
        df = df_Simulation.copy()
        df.loc['Avg', 'NAV_Last'] = df.loc['Avg', 'NAV_Last'].astype(float).round(4)
        df.loc['Avg', 'RR_Mean'] = round(float(df.loc['Avg', 'RR_Mean'].rstrip('%')) / 100.0, 4)
        df.loc['Avg', 'RR_Std'] = round(float(df.loc['Avg', 'RR_Std'].rstrip('%')) / 100.0, 4)
        df.loc['Avg', 'RR_Skew'] = df.loc['Avg', 'RR_Skew'].astype(float).round(4)
        df.loc['Avg', 'RR_Kurt'] = df.loc['Avg', 'RR_Kurt'].astype(float).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_LS'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_LS'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_DCA'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_DCA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_VA'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_VA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
        writer.save()

    # Summary of IRR #
    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary['Iter'] = int(iter + 1)
    df_Summary['Fund_Code'] = df_FundData.loc[df_FundNAV.columns[iter], 'Fund Code']
    df_Summary['Fund_Name'] = df_FundData.loc[df_FundNAV.columns[iter], 'Local Name - Thai']
    df_Summary['Category_GlobalBroad'] = df_FundData.loc[df_FundNAV.columns[iter], 'Global Broad Category Group']
    df_Summary['Category_Morningstar'] = df_FundData.loc[df_FundNAV.columns[iter], 'Morningstar Category']
    df_Summary['NAV_Last'] = df_Simulation.loc['Avg']['NAV_Last']
    df_Summary['RR_Mean'] = df_Simulation.loc['Avg']['RR_Mean']
    df_Summary['RR_Std'] = df_Simulation.loc['Avg']['RR_Std']
    df_Summary['RR_Skew'] = df_Simulation.loc['Avg']['RR_Skew']
    df_Summary['RR_Kurt'] = df_Simulation.loc['Avg']['RR_Kurt']
    df_Summary['IRR_LS'] = df_Simulation.loc['Avg']['IRR_LS']
    df_Summary['IRR_DCA'] = df_Simulation.loc['Avg']['IRR_DCA']
    df_Summary['IRR_VA'] = df_Simulation.loc['Avg']['IRR_VA']

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

    # Filter Only 10Y Fund #
    df_FundNAV = df_FundNAV.loc[:, df_FundNAV.count() >= forecast_year * n_per_year + 1]
    # df_FundNAV = df_FundNAV.iloc[:, 0:10]
    df_FundDiv = df_FundDiv.loc[:, df_FundNAV.columns].fillna(0)
    df_FundNAV = df_FundNAV.iloc[:forecast_year * n_per_year + 1].sort_index()
    df_FundDiv = df_FundDiv.iloc[:forecast_year * n_per_year + 1].sort_index()
    df_FundData = df_FundData.loc[df_FundNAV.columns, :]

    results = []
    pool = Pool()
    iter = df_FundNAV.shape[1]
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, df_FundNAV, df_FundDiv, df_FundData, forecast_year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_Summary = pd.DataFrame(results, columns=col_Summary, dtype='object')
    df_Summary.sort_values(by='Iter', inplace=True)
    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary.iloc[-1]['Iter'] = 'Avg'
    df_Summary.iloc[-1]['NAV_Last'] = df_Summary.iloc[:-1]['NAV_Last'].mean()
    df_Summary.iloc[-1]['RR_Mean'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR_Mean'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR_Std'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR_Std'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR_Skew'] = df_Summary.iloc[:-1]['RR_Skew'].mean()
    df_Summary.iloc[-1]['RR_Kurt'] = df_Summary.iloc[:-1]['RR_Kurt'].mean()
    df_Summary.iloc[-1]['IRR_LS'] = '{:.4%}'.format((df_Summary.iloc[:-1]['IRR_LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['IRR_DCA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['IRR_DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['IRR_VA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['IRR_VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary = df_Summary.fillna('')
    df_Summary = df_Summary.set_index('Iter')
    df_Summary.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], col.split('_')[-1]) for col in df_Summary.columns])
    print(df_Summary.drop(columns=['Name'], level=1))

    writer = pd.ExcelWriter('output/Fund_Summary_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    text_fmt = workbook.add_format({'align': 'left'})

    sheet_name = 'Summary'
    df = df_Summary.copy()
    df['RR', 'Mean'] = df['RR', 'Mean'].str.rstrip('%').astype('float') / 100.0
    df['RR', 'Std'] = df['RR', 'Std'].str.rstrip('%').astype('float') / 100.0
    df['IRR', 'LS'] = df['IRR', 'LS'].str.rstrip('%').astype('float') / 100.0
    df['IRR', 'DCA'] = df['IRR', 'DCA'].str.rstrip('%').astype('float') / 100.0
    df['IRR', 'VA'] = df['IRR', 'VA'].str.rstrip('%').astype('float') / 100.0
    df = df.round(4)
    df.to_excel(writer, sheet_name=sheet_name)
    worksheet = writer.sheets[sheet_name]
    body_fmt = {
        'B': text_fmt,
        'C': text_fmt,
        'D': text_fmt,
        'E': text_fmt,
        'F': float_fmt,
        'G': pct_fmt,
        'H': pct_fmt,
        'I': float_fmt,
        'J': float_fmt,
        'K': pct_fmt,
        'L': pct_fmt,
        'M': pct_fmt,
    }
    for col, width in enumerate(get_col_widths(df, index=False), 1):
        worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
    writer.save()
