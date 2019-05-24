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
                   'Fund NAV', 'Fund Bid Price', 'Fund Offer Price', 'Beg. Fund Value', 'Capital Gain', 'Buy/Sell Fund Value', 'Net Fund Value',
                   'Beg. Cash', 'Change in Cash', 'Dividend Per Unit', 'Dividend Gain', 'Income Tax', 'Net Cash', 'Total Wealth',
                   'Acc. Capital Gain', 'Acc. Dividend Gain', 'Acc. Fee & Tax', 'Net Profit', 'RR', 'ROI', 'IRR']
col_Simulation = ['Year', 'NAV_Last', 'NAV_Mean', 'NAV_Std', 'NAV_Skew', 'NAV_Kurt', 'RR_LS', 'RR_DCA', 'RR_VA', 'ROI_LS', 'ROI_DCA', 'ROI_VA', 'IRR_LS', 'IRR_DCA', 'IRR_VA']
col_Iter = ['Iter', 'Fund_Code', 'Fund_Name', 'Category_Morningstar', 'NAV_Last', 'NAV_Mean', 'NAV_Std', 'NAV_Skew', 'NAV_Kurt',
               'RR.Mean_LS', 'RR.Mean_DCA', 'RR.Mean_VA', 'RR.Std_LS', 'RR.Std_DCA', 'RR.Std_VA', 'RR.SR_LS', 'RR.SR_DCA', 'RR.SR_VA',
               'ROI.Mean_LS', 'ROI.Mean_DCA', 'ROI.Mean_VA', 'ROI.Std_LS', 'ROI.Std_DCA', 'ROI.Std_VA', 'ROI.SR_LS', 'ROI.SR_DCA', 'ROI.SR_VA',
               'IRR_LS', 'IRR_DCA', 'IRR_VA']

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
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = 0.0
            df.loc[t]['Acc. Dividend Gain'] = 0.0
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = 0.0
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = df.loc[t]['Capital Gain'] + df.loc[t - 1]['Acc. Capital Gain']
            df.loc[t]['Acc. Dividend Gain'] = df.loc[t]['Dividend Gain'] + df.loc[t - 1]['Acc. Dividend Gain']
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax'] \
                                          + df.loc[t - 1]['Acc. Fee & Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = -df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = df.loc[t]['Capital Gain'] + df.loc[t - 1]['Acc. Capital Gain']
            df.loc[t]['Acc. Dividend Gain'] = df.loc[t]['Dividend Gain'] + df.loc[t - 1]['Acc. Dividend Gain']
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax'] \
                                          + df.loc[t - 1]['Acc. Fee & Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['RR'] = '{:.4%}'.format(df.loc[t]['Net Profit'] / init_Cash)
            df.loc[t]['ROI'] = '{:.4%}'.format((df.loc[df['Change in Cash'] >= 0]['Change in Cash'].sum() / -df.loc[df['Change in Cash'] < 0]['Change in Cash'].sum()) - 1)
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
            df.loc[t]['Buy/Sell Fund Volume'] = init_Cash / n_per_year / df.loc[t]['Fund Offer Price']
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = 0.0
            df.loc[t]['Acc. Dividend Gain'] = 0.0
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = init_Cash / n_per_year / df.loc[t]['Fund Offer Price']
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = df.loc[t]['Capital Gain'] + df.loc[t - 1]['Acc. Capital Gain']
            df.loc[t]['Acc. Dividend Gain'] = df.loc[t]['Dividend Gain'] + df.loc[t - 1]['Acc. Dividend Gain']
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax'] \
                                          + df.loc[t - 1]['Acc. Fee & Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = -df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = df.loc[t]['Capital Gain'] + df.loc[t - 1]['Acc. Capital Gain']
            df.loc[t]['Acc. Dividend Gain'] = df.loc[t]['Dividend Gain'] + df.loc[t - 1]['Acc. Dividend Gain']
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax'] \
                                          + df.loc[t - 1]['Acc. Fee & Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['RR'] = '{:.4%}'.format(df.loc[t]['Net Profit'] / init_Cash)
            df.loc[t]['ROI'] = '{:.4%}'.format((df.loc[df['Change in Cash'] >= 0]['Change in Cash'].sum() / -df.loc[df['Change in Cash'] < 0]['Change in Cash'].sum()) - 1)
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
            df.loc[t]['Buy/Sell Fund Volume'] = diff / df.loc[t]['Fund NAV']
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = 0.0
            df.loc[t]['Acc. Dividend Gain'] = 0.0
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
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
            df.loc[t]['Buy/Sell Fund Volume'] = diff / df.loc[t]['Fund NAV']
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = df.loc[t]['Capital Gain'] + df.loc[t - 1]['Acc. Capital Gain']
            df.loc[t]['Acc. Dividend Gain'] = df.loc[t]['Dividend Gain'] + df.loc[t - 1]['Acc. Dividend Gain']
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax'] \
                                          + df.loc[t - 1]['Acc. Fee & Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Fund NAV'] = df_NAV_Y.loc[t]
            df.loc[t]['Fund Bid Price'] = np.floor(df.loc[t]['Fund NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Fund Offer Price'] = np.ceil(df.loc[t]['Fund NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000
            df.loc[t]['Beg. Fund Volume'] = df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Beg. Fund Value'] = df.loc[t]['Beg. Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Fund Value'] - df.loc[t - 1]['Net Fund Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Fund Volume'] = -df.loc[t - 1]['Net Fund Volume']
            df.loc[t]['Buy/Sell Fund Value'] = df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund NAV']
            if df.loc[t]['Buy/Sell Fund Volume'] > 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Offer Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] < 0.0:
                df.loc[t]['Change in Cash'] = -df.loc[t]['Buy/Sell Fund Volume'] * df.loc[t]['Fund Bid Price']
            elif df.loc[t]['Buy/Sell Fund Volume'] == 0.0:
                df.loc[t]['Change in Cash'] = 0.0
            df.loc[t]['Dividend Per Unit'] = df_Div_Y.loc[t]
            df.loc[t]['Dividend Gain'] = df.loc[t]['Dividend Per Unit'] * df.loc[t]['Beg. Fund Volume']
            df.loc[t]['Income Tax'] = df.loc[t]['Dividend Gain'] * -0.1
            df.loc[t]['Net Fund Volume'] = df.loc[t]['Beg. Fund Volume'] + df.loc[t]['Buy/Sell Fund Volume']
            df.loc[t]['Net Fund Value'] = df.loc[t]['Net Fund Volume'] * df.loc[t]['Fund NAV']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash'] + df.loc[t]['Dividend Gain'] + df.loc[t]['Income Tax']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Fund Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Acc. Capital Gain'] = df.loc[t]['Capital Gain'] + df.loc[t - 1]['Acc. Capital Gain']
            df.loc[t]['Acc. Dividend Gain'] = df.loc[t]['Dividend Gain'] + df.loc[t - 1]['Acc. Dividend Gain']
            df.loc[t]['Acc. Fee & Tax'] = ((df.loc[t]['Fund NAV'] - df.loc[t]['Fund Offer Price']) if (df.loc[t]['Buy/Sell Fund Volume'] > 0) else
                                           (df.loc[t]['Fund NAV'] - df.loc[t]['Fund Bid Price'])) * df.loc[t]['Buy/Sell Fund Volume'] + df.loc[t]['Income Tax'] \
                                          + df.loc[t - 1]['Acc. Fee & Tax']
            df.loc[t]['Net Profit'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['RR'] = '{:.4%}'.format(df.loc[t]['Net Profit'] / init_Cash)
            df.loc[t]['ROI'] = '{:.4%}'.format((df.loc[df['Change in Cash'] >= 0]['Change in Cash'].sum() / -df.loc[df['Change in Cash'] < 0]['Change in Cash'].sum()) - 1)
            df.loc[t]['IRR'] = '{:.4%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def simulation(df_FundNAV, df_FundDiv, df_FundData, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Simulation
    global col_Iter

    algo = ['LS', 'DCA', 'VA']
    df_Simulation = {}
    for i in range(len(algo)):
        df_Simulation[algo[i]] = {}
    df_Simulation['Summary'] = pd.DataFrame(columns=col_Simulation)
    df_Summary = pd.DataFrame(columns=col_Summary)

    df_Price = pd.DataFrame(df_FundNAV.iloc[:, iter])
    df_Price.columns = ['S']
    df_Price['RR'] = df_Price.pct_change()
    df_Price.reset_index(drop=True, inplace=True)
    df_Price.index.name = 'Month'
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
        sheet_name = 'Fund'
        df = df_Price.copy()
        df.loc[0, 'S'] = df.loc[0, 'S'].astype(float).round(4)
        df.loc[1:] = df.loc[1:].astype(float).round(4)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 2, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    for year in range(forecast_year):
        df_NAV_Y = df_Price.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True)
        df_Div_Y = df_Div.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['Div'].reset_index(drop=True)
        df_Simulation['LS'][year] = LS(df_NAV_Y, df_Div_Y, df_Data, init_Cash)
        df_Simulation['DCA'][year] = DCA(df_NAV_Y, df_Div_Y, df_Data, init_Cash)
        df_Simulation['VA'][year] = VA(df_NAV_Y, df_Div_Y, df_Data, init_Cash)
        df_Simulation['Summary'] = df_Simulation['Summary'].append({}, ignore_index=True)
        df_Simulation['Summary'].loc[year]['Year'] = year + 1
        df_Simulation['Summary'].loc[year]['NAV_Last'] = df_Price.iloc[(year + 1) * n_per_year]['S'].round(4)
        df_Simulation['Summary'].loc[year]['NAV_Mean'] = '{:.4%}'.format(df_Price.iloc[year * n_per_year + 1:(year + 1) * n_per_year + 1]['RR'].mean() * n_per_year)
        df_Simulation['Summary'].loc[year]['NAV_Std'] = '{:.4%}'.format(df_Price.iloc[year * n_per_year + 1:(year + 1) * n_per_year + 1]['RR'].std() * np.sqrt(n_per_year))
        df_Simulation['Summary'].loc[year]['NAV_Skew'] = round(df_Price.iloc[year * n_per_year + 1:(year + 1) * n_per_year + 1]['RR'].skew(), 4)
        df_Simulation['Summary'].loc[year]['NAV_Kurt'] = round(df_Price.iloc[year * n_per_year + 1:(year + 1) * n_per_year + 1]['RR'].kurt(), 4)
        df_Simulation['Summary'].loc[year]['RR_LS'] = df_Simulation['LS'][year].loc[n_per_year]['RR']
        df_Simulation['Summary'].loc[year]['RR_DCA'] = df_Simulation['DCA'][year].loc[n_per_year]['RR']
        df_Simulation['Summary'].loc[year]['RR_VA'] = df_Simulation['VA'][year].loc[n_per_year]['RR']
        df_Simulation['Summary'].loc[year]['ROI_LS'] = df_Simulation['LS'][year].loc[n_per_year]['ROI']
        df_Simulation['Summary'].loc[year]['ROI_DCA'] = df_Simulation['DCA'][year].loc[n_per_year]['ROI']
        df_Simulation['Summary'].loc[year]['ROI_VA'] = df_Simulation['VA'][year].loc[n_per_year]['ROI']
        df_Simulation['Summary'].loc[year]['IRR_LS'] = df_Simulation['LS'][year].loc[n_per_year]['IRR']
        df_Simulation['Summary'].loc[year]['IRR_DCA'] = df_Simulation['DCA'][year].loc[n_per_year]['IRR']
        df_Simulation['Summary'].loc[year]['IRR_VA'] = df_Simulation['VA'][year].loc[n_per_year]['IRR']

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
                'Q': float_fmt,
                'R': float_fmt,
                'S': float_fmt,
                'T': float_fmt,
                'U': float_fmt,
                'V': float_fmt,
                'W': pct_fmt,
                'X': pct_fmt,
                'Y': pct_fmt,
            }

            for i in range(len(algo)):
                sheet_name = '{}'.format(algo[i])
                df = df_Simulation[algo[i]][year].copy()
                df.index.names = ['Year{} / Month'.format(year + 1)]
                df.loc[n_per_year, 'RR'] = float(df.loc[n_per_year, 'RR'].rstrip('%')) / 100.0
                df.loc[n_per_year, 'ROI'] = float(df.loc[n_per_year, 'ROI'].rstrip('%')) / 100.0
                df.loc[n_per_year, 'IRR'] = float(df.loc[n_per_year, 'IRR'].rstrip('%')) / 100.0
                df = df.round(4)
                df.to_excel(writer, sheet_name=sheet_name, startrow=year * 15)
                worksheet = writer.sheets[sheet_name]
                for col, width in enumerate(get_col_widths(df, index=False), 1):
                    worksheet.set_column(col, col, width + 2, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    df_Simulation['Summary'] = df_Simulation['Summary'].append({}, ignore_index=True)
    df_Simulation['Summary'].iloc[-1]['Year'] = 'Avg'
    df_Simulation['Summary'].iloc[-1]['NAV_Last'] = df_Price.iloc[-1]['S'].astype(float).round(4)
    df_Simulation['Summary'].iloc[-1]['NAV_Mean'] = '{:.4%}'.format(df_Price.iloc[1:]['RR'].mean() * n_per_year)
    df_Simulation['Summary'].iloc[-1]['NAV_Std'] = '{:.4%}'.format(df_Price.iloc[1:]['RR'].std() * np.sqrt(n_per_year))
    df_Simulation['Summary'].iloc[-1]['NAV_Skew'] = df_Price.iloc[1:]['RR'].skew().astype(float).round(4)
    df_Simulation['Summary'].iloc[-1]['NAV_Kurt'] = df_Price.iloc[1:]['RR'].kurt().astype(float).round(4)
    df_Simulation['Summary'].iloc[-1]['RR_LS'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['RR_LS'].str.rstrip('%').astype('float').mean() / 100.0)
    df_Simulation['Summary'].iloc[-1]['RR_DCA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['RR_DCA'].str.rstrip('%').astype('float').mean() / 100.0)
    df_Simulation['Summary'].iloc[-1]['RR_VA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['RR_VA'].str.rstrip('%').astype('float').mean() / 100.0)
    df_Simulation['Summary'].iloc[-1]['ROI_LS'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['ROI_LS'].str.rstrip('%').astype('float').mean() / 100.0)
    df_Simulation['Summary'].iloc[-1]['ROI_DCA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['ROI_DCA'].str.rstrip('%').astype('float').mean() / 100.0)
    df_Simulation['Summary'].iloc[-1]['ROI_VA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['ROI_VA'].str.rstrip('%').astype('float').mean() / 100.0)
    df_Simulation['Summary'].iloc[-1]['IRR_LS'] = '{:.4%}'.format(gmean(1 + (df_Simulation['Summary'].iloc[:-1]['IRR_LS'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation['Summary'].iloc[-1]['IRR_DCA'] = '{:.4%}'.format(gmean(1 + (df_Simulation['Summary'].iloc[:-1]['IRR_DCA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation['Summary'].iloc[-1]['IRR_VA'] = '{:.4%}'.format(gmean(1 + (df_Simulation['Summary'].iloc[:-1]['IRR_VA'].str.rstrip('%').astype('float') / 100.0)) - 1)

    df_Simulation['Summary'] = df_Simulation['Summary'].append({}, ignore_index=True)
    df_Simulation['Summary'].iloc[-1]['Year'] = 'Std'
    df_Simulation['Summary'].iloc[-1]['RR_LS'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['RR_LS'].str.rstrip('%').astype('float').std() / 100.0)
    df_Simulation['Summary'].iloc[-1]['RR_DCA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['RR_DCA'].str.rstrip('%').astype('float').std() / 100.0)
    df_Simulation['Summary'].iloc[-1]['RR_VA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['RR_VA'].str.rstrip('%').astype('float').std() / 100.0)
    df_Simulation['Summary'].iloc[-1]['ROI_LS'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['ROI_LS'].str.rstrip('%').astype('float').std() / 100.0)
    df_Simulation['Summary'].iloc[-1]['ROI_DCA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['ROI_DCA'].str.rstrip('%').astype('float').std() / 100.0)
    df_Simulation['Summary'].iloc[-1]['ROI_VA'] = '{:.4%}'.format(df_Simulation['Summary'].iloc[:forecast_year]['ROI_VA'].str.rstrip('%').astype('float').std() / 100.0)

    df_Simulation['Summary'] = df_Simulation['Summary'].append({}, ignore_index=True)
    df_Simulation['Summary'].iloc[-1]['Year'] = 'SR'
    # Risk Free Rate 10Y = 1.8416, Risk Free Rate 5Y = 1.4760
    RiskFree = 1.8416 if forecast_year == 10 else 1.4760
    df_Simulation['Summary'].iloc[-1]['RR_LS'] = (
            (df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Avg', 'RR_LS'].str.rstrip('%').astype('float').iloc[0] - RiskFree) /
            df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Std', 'RR_LS'].str.rstrip('%').astype('float').iloc[0]).round(4)
    df_Simulation['Summary'].iloc[-1]['RR_DCA'] = (
            (df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Avg', 'RR_DCA'].str.rstrip('%').astype('float').iloc[0] - RiskFree) /
            df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Std', 'RR_DCA'].str.rstrip('%').astype('float').iloc[0]).round(4)
    df_Simulation['Summary'].iloc[-1]['RR_VA'] = (
            (df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Avg', 'RR_VA'].str.rstrip('%').astype('float').iloc[0] - RiskFree) /
            df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Std', 'RR_VA'].str.rstrip('%').astype('float').iloc[0]).round(4)
    df_Simulation['Summary'].iloc[-1]['ROI_LS'] = (
            (df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Avg', 'ROI_LS'].str.rstrip('%').astype('float').iloc[0] - RiskFree) /
            df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Std', 'ROI_LS'].str.rstrip('%').astype('float').iloc[0]).round(4)
    df_Simulation['Summary'].iloc[-1]['ROI_DCA'] = (
            (df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Avg', 'ROI_DCA'].str.rstrip('%').astype('float').iloc[0] - RiskFree) /
            df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Std', 'ROI_DCA'].str.rstrip('%').astype('float').iloc[0]).round(4)
    df_Simulation['Summary'].iloc[-1]['ROI_VA'] = (
            (df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Avg', 'ROI_VA'].str.rstrip('%').astype('float').iloc[0] - RiskFree) /
            df_Simulation['Summary'].loc[df_Simulation['Summary']['Year'] == 'Std', 'ROI_VA'].str.rstrip('%').astype('float').iloc[0]).round(4)

    df_Simulation['Summary'] = df_Simulation['Summary'].fillna('')
    df_Simulation['Summary'] = df_Simulation['Summary'].set_index('Year')

    if df_Data.loc['Fund Code'].iloc[0] == selectFund:
        body_fmt = {
            'B': float_fmt,
            'C': pct_fmt,
            'D': pct_fmt,
            'E': float2_fmt,
            'F': float2_fmt,
            'G': pct_fmt,
            'H': pct_fmt,
            'I': pct_fmt,
            'J': pct_fmt,
            'K': pct_fmt,
            'L': pct_fmt,
            'M': pct_fmt,
            'N': pct_fmt,
            'O': pct_fmt,
        }
        sheet_name = 'Summary'
        df = df_Simulation['Summary'].copy()
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'NAV_Mean'] = (df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'NAV_Mean'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'NAV_Std'] = (df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'NAV_Std'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'RR_LS'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'RR_LS'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'RR_DCA'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'RR_DCA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'RR_VA'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'RR_VA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'ROI_LS'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'ROI_LS'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'ROI_DCA'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'ROI_DCA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'ROI_VA'] = (
                df.loc[list(range(1, forecast_year + 1)) + ['Avg', 'Std'], 'ROI_VA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_LS'] = (df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_LS'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_DCA'] = (df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_DCA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_VA'] = (df.loc[list(range(1, forecast_year + 1)) + ['Avg'], 'IRR_VA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df = df.round(4)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 2, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
        worksheet.set_row(df.shape[0], None, float2_fmt)
        writer.save()

    # Summary of IRR #
    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary['Iter'] = int(iter + 1)
    df_Summary['Fund_Code'] = df_FundData.loc[df_FundNAV.columns[iter], 'Fund Code']
    df_Summary['Fund_Name'] = df_FundData.loc[df_FundNAV.columns[iter], 'Local Name - Thai']
    df_Summary['Category_Morningstar'] = df_FundData.loc[df_FundNAV.columns[iter], 'Morningstar Category']
    df_Summary['NAV_Last'] = df_Simulation['Summary'].loc['Avg']['NAV_Last']
    df_Summary['NAV_Mean'] = df_Simulation['Summary'].loc['Avg']['NAV_Mean']
    df_Summary['NAV_Std'] = df_Simulation['Summary'].loc['Avg']['NAV_Std']
    df_Summary['NAV_Skew'] = df_Simulation['Summary'].loc['Avg']['NAV_Skew']
    df_Summary['NAV_Kurt'] = df_Simulation['Summary'].loc['Avg']['NAV_Kurt']
    df_Summary['RR.Mean_LS'] = df_Simulation['Summary'].loc['Avg']['RR_LS']
    df_Summary['RR.Mean_DCA'] = df_Simulation['Summary'].loc['Avg']['RR_DCA']
    df_Summary['RR.Mean_VA'] = df_Simulation['Summary'].loc['Avg']['RR_VA']
    df_Summary['RR.Std_LS'] = df_Simulation['Summary'].loc['Std']['RR_LS']
    df_Summary['RR.Std_DCA'] = df_Simulation['Summary'].loc['Std']['RR_DCA']
    df_Summary['RR.Std_VA'] = df_Simulation['Summary'].loc['Std']['RR_VA']
    df_Summary['RR.SR_LS'] = df_Simulation['Summary'].loc['SR']['RR_LS']
    df_Summary['RR.SR_DCA'] = df_Simulation['Summary'].loc['SR']['RR_DCA']
    df_Summary['RR.SR_VA'] = df_Simulation['Summary'].loc['SR']['RR_VA']
    df_Summary['ROI.Mean_LS'] = df_Simulation['Summary'].loc['Avg']['ROI_LS']
    df_Summary['ROI.Mean_DCA'] = df_Simulation['Summary'].loc['Avg']['ROI_DCA']
    df_Summary['ROI.Mean_VA'] = df_Simulation['Summary'].loc['Avg']['ROI_VA']
    df_Summary['ROI.Std_LS'] = df_Simulation['Summary'].loc['Std']['ROI_LS']
    df_Summary['ROI.Std_DCA'] = df_Simulation['Summary'].loc['Std']['ROI_DCA']
    df_Summary['ROI.Std_VA'] = df_Simulation['Summary'].loc['Std']['ROI_VA']
    df_Summary['ROI.SR_LS'] = df_Simulation['Summary'].loc['SR']['ROI_LS']
    df_Summary['ROI.SR_DCA'] = df_Simulation['Summary'].loc['SR']['ROI_DCA']
    df_Summary['ROI.SR_VA'] = df_Simulation['Summary'].loc['SR']['ROI_VA']
    df_Summary['IRR_LS'] = df_Simulation['Summary'].loc['Avg']['IRR_LS']
    df_Summary['IRR_DCA'] = df_Simulation['Summary'].loc['Avg']['IRR_DCA']
    df_Summary['IRR_VA'] = df_Simulation['Summary'].loc['Avg']['IRR_VA']

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
    df_FundNAV = df_FundNAV.loc[:, df_FundNAV.count() >= forecast_year * n_per_year + 1]
    df_FundNAV = df_FundNAV.iloc[:forecast_year * n_per_year + 1].sort_index()
    # todo Test only 10 funds
    df_FundNAV = df_FundNAV.iloc[:, 0:10]

    df_FundDiv = df_FundDiv.loc[df_FundNAV.index, df_FundNAV.columns].fillna(0)
    df_FundData = df_FundData.loc[df_FundNAV.columns, :]

    results = []
    pool = Pool()
    iter = df_FundNAV.shape[1]
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, df_FundNAV, df_FundDiv, df_FundData, forecast_year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_Summary = pd.DataFrame(results, columns=col_Iter, dtype='object')
    df_Summary.sort_values(by='Iter', inplace=True)

    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary.iloc[-1]['Iter'] = 'Avg'
    df_Summary.iloc[-1]['NAV_Last'] = df_Summary.iloc[:-1]['NAV_Last'].mean()
    df_Summary.iloc[-1]['NAV_Mean'] = '{:.4%}'.format((df_Summary.iloc[:-1]['NAV_Mean'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['NAV_Std'] = '{:.4%}'.format((df_Summary.iloc[:-1]['NAV_Std'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['NAV_Skew'] = df_Summary.iloc[:-1]['NAV_Skew'].mean()
    df_Summary.iloc[-1]['NAV_Kurt'] = df_Summary.iloc[:-1]['NAV_Kurt'].mean()
    df_Summary.iloc[-1]['RR.Mean_LS'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR.Mean_LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR.Mean_DCA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR.Mean_DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR.Mean_VA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR.Mean_VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR.Std_LS'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR.Std_LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR.Std_DCA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR.Std_DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR.Std_VA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['RR.Std_VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['RR.SR_LS'] = df_Summary.iloc[:-1]['RR.SR_LS'].mean()
    df_Summary.iloc[-1]['RR.SR_DCA'] = df_Summary.iloc[:-1]['RR.SR_DCA'].mean()
    df_Summary.iloc[-1]['RR.SR_VA'] = df_Summary.iloc[:-1]['RR.SR_VA'].mean()
    df_Summary.iloc[-1]['ROI.Mean_LS'] = '{:.4%}'.format((df_Summary.iloc[:-1]['ROI.Mean_LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['ROI.Mean_DCA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['ROI.Mean_DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['ROI.Mean_VA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['ROI.Mean_VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['ROI.Std_LS'] = '{:.4%}'.format((df_Summary.iloc[:-1]['ROI.Std_LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['ROI.Std_DCA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['ROI.Std_DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['ROI.Std_VA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['ROI.Std_VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['ROI.SR_LS'] = df_Summary.iloc[:-1]['ROI.SR_LS'].mean()
    df_Summary.iloc[-1]['ROI.SR_DCA'] = df_Summary.iloc[:-1]['ROI.SR_DCA'].mean()
    df_Summary.iloc[-1]['ROI.SR_VA'] = df_Summary.iloc[:-1]['ROI.SR_VA'].mean()
    df_Summary.iloc[-1]['IRR_LS'] = '{:.4%}'.format((df_Summary.iloc[:-1]['IRR_LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['IRR_DCA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['IRR_DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary.iloc[-1]['IRR_VA'] = '{:.4%}'.format((df_Summary.iloc[:-1]['IRR_VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_Summary = df_Summary.fillna('')
    df_Summary = df_Summary.set_index('Iter')
    df_Summary.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], col.split('_')[-1]) for col in df_Summary.columns])
    print(df_Summary.drop(columns=['Name'], level=1))

    writer = pd.ExcelWriter('output/Fund_Sum_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    text_fmt = workbook.add_format({'align': 'left'})

    sheet_name = 'Summary'
    df = df_Summary.copy()
    df['NAV', 'Mean'] = df['NAV', 'Mean'].str.rstrip('%').astype('float') / 100.0
    df['NAV', 'Std'] = df['NAV', 'Std'].str.rstrip('%').astype('float') / 100.0
    df['RR.Mean', 'LS'] = df['RR.Mean', 'LS'].str.rstrip('%').astype('float') / 100.0
    df['RR.Mean', 'DCA'] = df['RR.Mean', 'DCA'].str.rstrip('%').astype('float') / 100.0
    df['RR.Mean', 'VA'] = df['RR.Mean', 'VA'].str.rstrip('%').astype('float') / 100.0
    df['RR.Std', 'LS'] = df['RR.Std', 'LS'].str.rstrip('%').astype('float') / 100.0
    df['RR.Std', 'DCA'] = df['RR.Std', 'DCA'].str.rstrip('%').astype('float') / 100.0
    df['RR.Std', 'VA'] = df['RR.Std', 'VA'].str.rstrip('%').astype('float') / 100.0
    df['ROI.Mean', 'LS'] = df['ROI.Mean', 'LS'].str.rstrip('%').astype('float') / 100.0
    df['ROI.Mean', 'DCA'] = df['ROI.Mean', 'DCA'].str.rstrip('%').astype('float') / 100.0
    df['ROI.Mean', 'VA'] = df['ROI.Mean', 'VA'].str.rstrip('%').astype('float') / 100.0
    df['ROI.Std', 'LS'] = df['ROI.Std', 'LS'].str.rstrip('%').astype('float') / 100.0
    df['ROI.Std', 'DCA'] = df['ROI.Std', 'DCA'].str.rstrip('%').astype('float') / 100.0
    df['ROI.Std', 'VA'] = df['ROI.Std', 'VA'].str.rstrip('%').astype('float') / 100.0
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
        'E': float_fmt,
        'F': pct_fmt,
        'G': pct_fmt,
        'H': float_fmt,
        'I': float_fmt,
        'J': pct_fmt,
        'K': pct_fmt,
        'L': pct_fmt,
        'M': pct_fmt,
        'N': pct_fmt,
        'O': pct_fmt,
        'P': float2_fmt,
        'Q': float2_fmt,
        'R': float2_fmt,
        'S': pct_fmt,
        'T': pct_fmt,
        'U': pct_fmt,
        'V': pct_fmt,
        'W': pct_fmt,
        'X': pct_fmt,
        'Y': float2_fmt,
        'Z': float2_fmt,
        'AA': float2_fmt,
        'AB': pct_fmt,
        'AC': pct_fmt,
        'AD': pct_fmt,
    }
    for col, width in enumerate(get_col_widths(df, index=False), 1):
        worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
    writer.save()
