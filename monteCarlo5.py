from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tqdm
import xlsxwriter.utility
from matplotlib import style
from scipy.stats.mstats import gmean
from sklearn.utils import resample

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
style.use('ggplot')
n_per_year = 12


def monte_carlo(forecast_year_):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S', 'RR'])

    ### Monte Carlo Config ###
    df_SET = pd.read_excel('SET.xlsx', sheet_name='Sheet1')
    init_S = df_SET.iloc[0]['SETi']  # 449.96
    u = df_SET.iloc[1:]['RR'].mean() * n_per_year  # 0.1376
    sigma = df_SET.iloc[1:]['RR'].std() * np.sqrt(n_per_year)  # 0.1580
    dt = 1 / n_per_year

    for t in range(0, (forecast_year_ * n_per_year) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year_ * n_per_year) + 1:
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


def direct(forecast_year_):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RR', 'S'])

    df_SET = pd.read_excel('SET.xlsx', sheet_name='Sheet1')
    RR = df_SET.iloc[1:]['RR'].values
    SETi = df_SET.iloc[1:]['SETi'].values
    init_S = df_SET.iloc[0]['SETi']

    for t in range(0, (forecast_year_ * n_per_year) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year_ * n_per_year) + 1:
            df.loc[t]['RR'] = RR[t - 1]
            df.loc[t]['S'] = SETi[t - 1]

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def bootstrap(forecast_year_):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RR', 'dS', 'S'])

    df_SET = pd.read_excel('SET.xlsx', sheet_name='Sheet1')
    RR = df_SET.iloc[1:]['RR'].values
    RR = resample(RR, replace=True, n_samples=forecast_year_ * n_per_year, random_state=None)
    init_S = df_SET.iloc[0]['SETi']

    for t in range(0, (forecast_year_ * 12) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year_ * n_per_year) + 1:
            df.loc[t]['RR'] = RR[t - 1]
            df.loc[t]['dS'] = df.loc[t]['RR'] * df.loc[t - 1]['S']
            df.loc[t]['S'] = df.loc[t - 1]['S'] + df.loc[t]['dS']

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def LS(df_Stock_, init_Cash_):
    global n_per_year
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = 0.0
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
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
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def DCA(df_Stock_, init_Cash_):
    global n_per_year
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / n_per_year / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = init_Cash_ / n_per_year / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
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
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def VA(df_Stock_, init_Cash_):
    global n_per_year
    df = pd.DataFrame(
        columns=['Month', 'Beg. Inv.Asset Volume', 'Buy/Sell Inv.Asset Volume', 'Net Inv.Asset Volume',
                 'Inv.Asset Price', 'Capital Gain', 'Beg. Inv.Asset Value', 'Change in Inv.Asset Value', 'Net Inv.Asset Value',
                 'Beg. Cash', 'Change in Cash', 'Net Cash', 'Total Wealth', 'Profit/Loss', 'IRR'])

    for t in range(0, n_per_year + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        if t == 0:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = 0.0
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = 0.0
            df.loc[t]['Beg. Cash'] = init_Cash_
            diff = ((t + 1) * init_Cash_ / n_per_year) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t in range(1, n_per_year):
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
            df.loc[t]['Beg. Inv.Asset Volume'] = df.loc[t - 1]['Net Inv.Asset Volume']
            df.loc[t]['Beg. Inv.Asset Value'] = df.loc[t]['Beg. Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Capital Gain'] = df.loc[t]['Beg. Inv.Asset Value'] - df.loc[t - 1]['Net Inv.Asset Value']
            df.loc[t]['Beg. Cash'] = df.loc[t - 1]['Net Cash']
            # todo interest not included yet
            diff = ((t + 1) * init_Cash_ / n_per_year) - df.loc[t]['Beg. Inv.Asset Value']
            diff = diff if diff <= df.loc[t]['Beg. Cash'] else df.loc[t]['Beg. Cash']
            df.loc[t]['Buy/Sell Inv.Asset Volume'] = diff / df_Stock_.loc[t]
            df.loc[t]['Change in Inv.Asset Value'] = df.loc[t]['Buy/Sell Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Change in Cash'] = -df.loc[t]['Change in Inv.Asset Value'] if df.loc[t]['Change in Inv.Asset Value'] != 0.0 else 0.0
            df.loc[t]['Net Inv.Asset Volume'] = df.loc[t]['Beg. Inv.Asset Volume'] + df.loc[t]['Buy/Sell Inv.Asset Volume']
            df.loc[t]['Net Inv.Asset Value'] = df.loc[t]['Net Inv.Asset Volume'] * df.loc[t]['Inv.Asset Price']
            df.loc[t]['Net Cash'] = df.loc[t]['Beg. Cash'] + df.loc[t]['Change in Cash']
            df.loc[t]['Total Wealth'] = df.loc[t]['Net Inv.Asset Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
        elif t == n_per_year:
            df.loc[t]['Inv.Asset Price'] = df_Stock_.loc[t]
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
            df.loc[t]['Profit/Loss'] = df.loc[t]['Total Wealth'] - init_Cash_
            df.loc[t]['IRR'] = '{:.2%}'.format(((1 + np.irr(df['Change in Cash'].tolist())) ** n_per_year) - 1)

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def simulation(method, forecast_year_, init_Cash_, i):
    ### Portfolio Simulation ###
    global n_per_year
    df_IRR = pd.DataFrame(columns=['Year', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'LS', 'DCA', 'VA'])
    df_IRR_Sum_ = pd.DataFrame(columns=['Iter', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'LS', 'DCA', 'VA'])
    df_LS = {}
    df_DCA = {}
    df_VA = {}

    if method == 1:
        df_Stock = monte_carlo(forecast_year_)
        writer = pd.ExcelWriter('Out_Sim_MonteCarlo.xlsx')
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
    elif method == 2:
        df_Stock = direct(forecast_year_)
        writer = pd.ExcelWriter('Out_Sim_DirectTest.xlsx')
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': float_fmt,
            'C': float_fmt,
        }
    elif method == 3:
        df_Stock = bootstrap(forecast_year_)
        writer = pd.ExcelWriter('Out_Sim_Bootstrap.xlsx')
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': float_fmt,
            'C': float_fmt,
            'D': float_fmt,
        }

    for year in range(forecast_year_):
        df_LS[year] = LS(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash_)
        df_DCA[year] = DCA(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash_)
        df_VA[year] = VA(df_Stock.iloc[(year * n_per_year):((year + 1) * n_per_year) + 1]['S'].reset_index(drop=True), init_Cash_)

        df_IRR = df_IRR.append({}, ignore_index=True)
        df_IRR.loc[year]['Year'] = year + 1
        df_IRR.loc[year]['LS'] = df_LS[year].loc[n_per_year]['IRR']
        df_IRR.loc[year]['DCA'] = df_DCA[year].loc[n_per_year]['IRR']
        df_IRR.loc[year]['VA'] = df_VA[year].loc[n_per_year]['IRR']

        if i == 0:

            if year == 0:
                sheet_name = 'Stock'
                df = df_Stock
                if method == 1:
                    df.loc[0, 'S'] = df.loc[0, 'S'].astype(float).round(4)
                    df.loc[1:] = df.loc[1:].astype(float).round(4)
                elif method == 2:
                    df.loc[0, 'S'] = df.loc[0, 'S'].astype(float).round(4)
                    df.loc[1:] = df.loc[1:].astype(float).round(4)
                elif method == 3:
                    df.loc[0, 'S'] = df.loc[0, 'S'].astype(float).round(4)
                    df.loc[1:] = df.loc[1:].astype(float).round(4)
                df.to_excel(writer, sheet_name=sheet_name)
                worksheet = writer.sheets[sheet_name]
                for col, width in enumerate(get_col_widths(df, index=False), 1):
                    worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

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
            df = df_LS[year]
            df.loc[n_per_year, 'IRR'] = round(float(df.loc[n_per_year, 'IRR'].rstrip('%')) / 100.0, 4)
            df = df.round(4)
            df.to_excel(writer, sheet_name=sheet_name, startrow=year * 15)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

            sheet_name = 'DCA'
            df = df_DCA[year]
            df.loc[n_per_year, 'IRR'] = round(float(df.loc[n_per_year, 'IRR'].rstrip('%')) / 100.0, 4)
            df = df.round(4)
            df.to_excel(writer, sheet_name=sheet_name, startrow=year * 15)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

            sheet_name = 'VA'
            df = df_VA[year]
            df.loc[n_per_year, 'IRR'] = round(float(df.loc[n_per_year, 'IRR'].rstrip('%')) / 100.0, 4)
            df = df.round(4)
            df.to_excel(writer, sheet_name=sheet_name, startrow=year * 15)
            worksheet = writer.sheets[sheet_name]
            for col, width in enumerate(get_col_widths(df, index=False), 1):
                worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    df_IRR = df_IRR.append({}, ignore_index=True)
    df_IRR.loc[forecast_year_]['Year'] = 'Avg'
    df_IRR.loc[forecast_year_]['SET_Final'] = df_Stock.iloc[-1]['S']
    df_IRR.loc[forecast_year_]['RR_Mean'] = '{:.2%}'.format(df_Stock.iloc[1:]['RR'].mean() * n_per_year)
    df_IRR.loc[forecast_year_]['RR_Std'] = '{:.2%}'.format(df_Stock.iloc[1:]['RR'].std() * np.sqrt(n_per_year))
    df_IRR.loc[forecast_year_]['RR_Skew'] = df_Stock.iloc[1:]['RR'].skew()
    df_IRR.loc[forecast_year_]['RR_Kurt'] = df_Stock.iloc[1:]['RR'].kurt()
    df_IRR.loc[forecast_year_]['LS'] = '{:.2%}'.format(gmean(1 + (df_IRR.iloc[:-2]['LS'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_IRR.loc[forecast_year_]['DCA'] = '{:.2%}'.format(gmean(1 + (df_IRR.iloc[:-2]['DCA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_IRR.loc[forecast_year_]['VA'] = '{:.2%}'.format(gmean(1 + (df_IRR.iloc[:-2]['VA'].str.rstrip('%').astype('float') / 100.0)) - 1)
    df_IRR = df_IRR.fillna('')
    df_IRR = df_IRR.set_index('Year')

    if i == 0:
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
        sheet_name = 'IRR'
        df = df_IRR
        df.loc['Avg', 'SET_Final'] = df.loc['Avg', 'SET_Final'].astype(float).round(4)
        df.loc['Avg', 'RR_Mean'] = round(float(df.loc['Avg', 'RR_Mean'].rstrip('%')) / 100.0, 4)
        df.loc['Avg', 'RR_Std'] = round(float(df.loc['Avg', 'RR_Std'].rstrip('%')) / 100.0, 4)
        df.loc['Avg', 'RR_Skew'] = df.loc['Avg', 'RR_Skew'].astype(float).round(4)
        df.loc['Avg', 'RR_Kurt'] = df.loc['Avg', 'RR_Kurt'].astype(float).round(4)
        df.loc[list(range(1, forecast_year_ + 1)) + ['Avg'], 'LS'] = (
                    df.loc[list(range(1, forecast_year_ + 1)) + ['Avg'], 'LS'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year_ + 1)) + ['Avg'], 'DCA'] = (
                    df.loc[list(range(1, forecast_year_ + 1)) + ['Avg'], 'DCA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.loc[list(range(1, forecast_year_ + 1)) + ['Avg'], 'VA'] = (
                    df.loc[list(range(1, forecast_year_ + 1)) + ['Avg'], 'VA'].str.rstrip('%').astype('float') / 100.0).round(4)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
        writer.save()

    ### Summary of IRR ###
    df_IRR_Sum_ = df_IRR_Sum_.append({}, ignore_index=True)
    df_IRR_Sum_['Iter'] = int(i + 1)
    df_IRR_Sum_['SET_Final'] = df_IRR.loc['Avg']['SET_Final']
    df_IRR_Sum_['RR_Mean'] = df_IRR.loc['Avg']['RR_Mean']
    df_IRR_Sum_['RR_Std'] = df_IRR.loc['Avg']['RR_Std']
    df_IRR_Sum_['RR_Skew'] = df_IRR.loc['Avg']['RR_Skew']
    df_IRR_Sum_['RR_Kurt'] = df_IRR.loc['Avg']['RR_Kurt']
    df_IRR_Sum_['LS'] = df_IRR.loc['Avg']['LS']
    df_IRR_Sum_['DCA'] = df_IRR.loc['Avg']['DCA']
    df_IRR_Sum_['VA'] = df_IRR.loc['Avg']['VA']

    return df_IRR_Sum_.values.tolist()


def get_col_widths(df, index=True):
    if index:
        idx_max = max([len(str(s)) for s in df.index.values] + [len(str(df.index.name))])
        col_widths = [idx_max] + [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    else:
        col_widths = [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    return col_widths


if __name__ == '__main__':
    ### Simulation Config ###
    method = 1  # 1: Monte Carlo, 2: Direct Test, 3: Bootstrap
    iter = 5
    forecast_year = 10
    np.random.seed(None)

    ### Initial value ###
    init_Cash = 120000.0

    results = []
    pool = Pool()
    iter = iter if method != 2 else 1
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, method, forecast_year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_IRR_Sum = pd.DataFrame(results, columns=['Iter', 'SET_Final', 'RR_Mean', 'RR_Std', 'RR_Skew', 'RR_Kurt', 'LS', 'DCA', 'VA'], dtype='object')
    df_IRR_Sum.sort_values(by='Iter', inplace=True)

    df_IRR_Sum = df_IRR_Sum.append({}, ignore_index=True)
    df_IRR_Sum.iloc[-1]['Iter'] = 'Avg'
    df_IRR_Sum.iloc[-1]['SET_Final'] = df_IRR_Sum.iloc[:-1]['SET_Final'].mean()
    df_IRR_Sum.iloc[-1]['RR_Mean'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['RR_Mean'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['RR_Std'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['RR_Std'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['RR_Skew'] = df_IRR_Sum.iloc[:-1]['RR_Skew'].mean()
    df_IRR_Sum.iloc[-1]['RR_Kurt'] = df_IRR_Sum.iloc[:-1]['RR_Kurt'].mean()
    df_IRR_Sum.iloc[-1]['LS'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['LS'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['DCA'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['DCA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum.iloc[-1]['VA'] = '{:.2%}'.format((df_IRR_Sum.iloc[:-1]['VA'].str.rstrip('%').astype('float') / 100.0).mean())
    df_IRR_Sum = df_IRR_Sum.fillna('')
    df_IRR_Sum = df_IRR_Sum.set_index('Iter')
    df_IRR_Sum.rename(columns={'LS': 'Simulation_LS', 'DCA': 'Simulation_DCA', 'VA': 'Simulation_VA'}, inplace=True)
    df_IRR_Sum.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], col.split('_')[-1]) for col in df_IRR_Sum.columns])

    # print(df_IRR_Sum)

    if method == 1:
        writer = pd.ExcelWriter('Out_IRR_MonteCarlo.xlsx')
    elif method == 2:
        writer = pd.ExcelWriter('Out_IRR_DirectTest.xlsx')
    elif method == 3:
        writer = pd.ExcelWriter('Out_IRR_Bootstrap.xlsx')
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})

    sheet_name = 'IRR_Sum'
    df = df_IRR_Sum
    df['RR', 'Mean'] = df['RR', 'Mean'].str.rstrip('%').astype('float') / 100.0
    df['RR', 'Std'] = df['RR', 'Std'].str.rstrip('%').astype('float') / 100.0
    df['Simulation', 'LS'] = df['Simulation', 'LS'].str.rstrip('%').astype('float') / 100.0
    df['Simulation', 'DCA'] = df['Simulation', 'DCA'].str.rstrip('%').astype('float') / 100.0
    df['Simulation', 'VA'] = df['Simulation', 'VA'].str.rstrip('%').astype('float') / 100.0
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
        worksheet.set_column(col, col, width + 3, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
    writer.save()
