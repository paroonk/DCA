from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tqdm
import xlsxwriter.utility
from matplotlib import style
from scipy.stats.mstats import gmean
from sklearn.utils import resample


def get_col_widths(df, index=True):
    if index:
        idx_max = max([len(str(s)) for s in df.index.values] + [len(str(df.index.name))])
        col_widths = [idx_max] + [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    else:
        col_widths = [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    return col_widths


pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')
n_per_year = 12
col_Transaction = ['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                   'Inv.Asset Price', 'Capital Gain', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                   'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR']
col_Simulation = ['Year', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'IRR_LS', 'IRR_DCA', 'IRR_VA']
col_Summary = ['Iter', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'IRR_LS', 'IRR_DCA', 'IRR_VA']

# Simulation Config #
method = 3  # 1: Direct Test, 2: Monte Carlo, 3: Bootstrap
iter = 2
forecast_year = 10
init_Cash = 120000.0


def direct(df_SET, forecast_year):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RR', 'S'])

    RR = df_SET.iloc[1:]['RR'].values
    SETi = df_SET.iloc[1:]['SETi'].values
    init_S = df_SET.iloc[0]['SETi']

    for t in range(0, (forecast_year * n_per_year) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year * n_per_year) + 1:
            df.loc[t]['RR'] = RR[t - 1]
            df.loc[t]['S'] = SETi[t - 1]

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def monte_carlo(df_SET, forecast_year):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S', 'RR'])

    init_S = df_SET.iloc[0]['SETi']
    u = df_SET.iloc[1:]['RR'].mean() * n_per_year
    sigma = df_SET.iloc[1:]['RR'].std() * np.sqrt(n_per_year)
    dt = 1 / n_per_year

    for t in range(0, (forecast_year * n_per_year) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year * n_per_year) + 1:
            df.loc[t]['u.dt'] = u * dt
            df.loc[t]['S(u.dt)'] = df.loc[t - 1]['S'] * df.loc[t]['u.dt']
            df.loc[t]['N'] = np.random.normal()
            df.loc[t]['N.sigma.sqrt(dt)'] = df.loc[t]['N'] * sigma * np.sqrt(dt)
            df.loc[t]['S(N.sigma.sqrt(dt))'] = df.loc[t - 1]['S'] * df.loc[t]['N.sigma.sqrt(dt)']
            df.loc[t]['dS'] = df.loc[t]['S(u.dt)'] + df.loc[t]['S(N.sigma.sqrt(dt))']
            df.loc[t]['S'] = df.loc[t - 1]['S'] + df.loc[t]['dS']
            df.loc[t]['RR'] = (df.loc[t]['S'] - df.loc[t - 1]['S']) / df.loc[t - 1]['S']

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def bootstrap(df_SET, forecast_year):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RR', 'dS', 'S'])

    RR = df_SET.iloc[1:]['RR'].values
    RR = resample(RR, replace=True, n_samples=forecast_year * n_per_year, random_state=None)
    init_S = df_SET.iloc[0]['SETi']

    for t in range(0, (forecast_year * 12) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year * n_per_year) + 1:
            df.loc[t]['RR'] = RR[t - 1]
            df.loc[t]['dS'] = df.loc[t]['RR'] * df.loc[t - 1]['S']
            df.loc[t]['S'] = df.loc[t - 1]['S'] + df.loc[t]['dS']

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def LS(df_Price_Y, init_Cash):
    global n_per_year
    global col_Transaction
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = 0.0
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.4%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def DCA(df_Price_Y, init_Cash):
    global n_per_year
    global col_Transaction
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / n_per_year / df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash / n_per_year / df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.4%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def VA(df_Price_Y, init_Cash):
    global n_per_year
    global col_Transaction
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash
            diff = ((t + 1) * init_Cash / n_per_year) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            diff = ((t + 1) * init_Cash / n_per_year) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Price_Y.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = -df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash
            df.loc[t]['IRR'] = '{:.4%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def simulation(method, df_SET, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Simulation
    global col_Summary
    df_Simulation = pd.DataFrame(columns=col_Simulation)
    df_Summary = pd.DataFrame(columns=col_Summary)
    df_LS = {}
    df_DCA = {}
    df_VA = {}

    if method == 1:
        df_Price = direct(df_SET, forecast_year)
        writer = pd.ExcelWriter('output/DirectTest_Simulation_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': float_fmt,
            'C': float_fmt,
        }
    elif method == 2:
        df_Price = monte_carlo(df_SET, forecast_year)
        writer = pd.ExcelWriter('output/MonteCarlo_Simulation_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': float_fmt,
            'C': float_fmt,
            'D': float_fmt,
            'E': float_fmt,
            'F': float_fmt,
            'G': float_fmt,
            'H': float_fmt,
            'I': float_fmt,
        }
    elif method == 3:
        df_Price = bootstrap(df_SET, forecast_year)
        writer = pd.ExcelWriter('output/Bootstrap_Simulation_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': float_fmt,
            'C': float_fmt,
            'D': float_fmt,
        }

    for year in range(forecast_year):
        df_Price_Y = df_Price.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True)
        df_LS[year] = LS(df_Price_Y, init_Cash)
        df_DCA[year] = DCA(df_Price_Y, init_Cash)
        df_VA[year] = VA(df_Price_Y, init_Cash)
        df_Simulation = df_Simulation.append({}, ignore_index=True)
        df_Simulation.loc[year]['Year'] = year + 1
        df_Simulation.loc[year]['IRR_LS'] = df_LS[year].loc[n_per_year]['IRR']
        df_Simulation.loc[year]['IRR_DCA'] = df_DCA[year].loc[n_per_year]['IRR']
        df_Simulation.loc[year]['IRR_VA'] = df_VA[year].loc[n_per_year]['IRR']

        if iter == 0 and year == 0:
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
                'O': pct_fmt,
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
    df_Simulation.loc[forecast_year]['SET_Final'] = df_Price.iloc[-1]['S']
    df_Simulation.loc[forecast_year]['RR_Mean'] = '{:.4%}'.format(df_Price.iloc[1:]['RR'].mean() * n_per_year)
    df_Simulation.loc[forecast_year]['RR_Std'] = '{:.4%}'.format(df_Price.iloc[1:]['RR'].std() * np.sqrt(n_per_year))
    df_Simulation.loc[forecast_year]['RR_Skew'] = df_Price.iloc[1:]['RR'].skew()
    df_Simulation.loc[forecast_year]['RR_Kurt'] = df_Price.iloc[1:]['RR'].kurt()
    df_Simulation.loc[forecast_year]['IRR_LS'] = '{:.4%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_LS'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation.loc[forecast_year]['IRR_DCA'] = '{:.4%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_DCA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation.loc[forecast_year]['IRR_VA'] = '{:.4%}'.format(gmean(1 + (df_Simulation.iloc[:-1]['IRR_VA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_Simulation = df_Simulation.fillna('')
    df_Simulation = df_Simulation.set_index('Year')

    if iter == 0:
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
        df.loc['Avg', 'SET_Final'] = df.loc['Avg', 'SET_Final'].astype(float).round(4)
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
    df_Summary['SET_Final'] = df_Simulation.loc['Avg']['SET_Final']
    df_Summary['RR_Mean'] = df_Simulation.loc['Avg']['RR_Mean']
    df_Summary['RR_Std'] = df_Simulation.loc['Avg']['RR_Std']
    df_Summary['RR_Skew'] = df_Simulation.loc['Avg']['RR_Skew']
    df_Summary['RR_Kurt'] = df_Simulation.loc['Avg']['RR_Kurt']
    df_Summary['IRR_LS'] = df_Simulation.loc['Avg']['IRR_LS']
    df_Summary['IRR_DCA'] = df_Simulation.loc['Avg']['IRR_DCA']
    df_Summary['IRR_VA'] = df_Simulation.loc['Avg']['IRR_VA']

    return df_Summary.values.tolist()


if __name__ == '__main__':

    # Price Dataframe #
    df_SET = pd.read_excel('data/SET_TR.xlsx', sheet_name='Sheet1')
    df_SET = df_SET.iloc[(len(df_SET.index) - (forecast_year * n_per_year) - 1):]

    results = []
    pool = Pool()
    iter = iter if method != 1 else 1
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, method, df_SET, forecast_year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_Summary = pd.DataFrame(results, columns=col_Summary, dtype='object')
    df_Summary.sort_values(by='Iter', inplace=True)

    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary.iloc[-1]['Iter'] = 'Avg'
    df_Summary.iloc[-1]['SET_Final'] = df_Summary.iloc[:-1]['SET_Final'].mean()
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
    print(df_Summary)

    if method == 1:
        writer = pd.ExcelWriter('output/DirectTest_Summary_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    elif method == 2:
        writer = pd.ExcelWriter('output/MonteCarlo_Summary_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    elif method == 3:
        writer = pd.ExcelWriter('output/Bootstrap_Summary_{}.xlsx'.format(pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})

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
        'B': float_fmt,
        'C': pct_fmt,
        'D': pct_fmt,
        'E': float_fmt,
        'F': float_fmt,
        'G': pct_fmt,
        'H': pct_fmt,
        'I': pct_fmt,
    }
    for col, width in enumerate(get_col_widths(df, index=False), 1):
        worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
    writer.save()
