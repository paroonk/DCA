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
col_Transaction = ['Month', 'Price', 'Required Value', 'Shares Bought', 'Shares Owned', 'Portfolio Value',
                   'Total Cost', 'Average Cost', 'CFF', 'CFI', 'Net Cash', 'Net Wealth', 'RoR']
col_Simulation = ['SET', 'LS', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']
row_Simulation = ['Last', 'Avg. Cost', 'Mean', 'Std', 'SR', 'IRR']
col_Summary = ['Iter',
               'SET_Last', 'SET_Mean', 'SET_Std', 'SET_SR',
               'Avg. Cost_LS', 'Avg. Cost_DCA', 'Avg. Cost_VA', 'Avg. Cost_VA6', 'Avg. Cost_VA12', 'Avg. Cost_VA18',
               'Mean_LS', 'Mean_DCA', 'Mean_VA', 'Mean_VA6', 'Mean_VA12', 'Mean_VA18',
               'Std_LS', 'Std_DCA', 'Std_VA', 'Std_VA6', 'Std_VA12', 'Std_VA18',
               'SR_LS', 'SR_DCA', 'SR_VA', 'SR_VA6', 'SR_VA12', 'SR_VA18',
               'IRR_LS', 'IRR_DCA', 'IRR_VA', 'IRR_VA6', 'IRR_VA12', 'IRR_VA18']

# Simulation Config #
method = 3  # 1: Direct Test, 2: Monte Carlo, 3: Bootstrap
iter = 1000
forecast_Year = 10
n_per_year = 12
init_Cash = 120000.0
Div_ReInvest = True


def get_col_widths(df, index=True):
    if index:
        idx_max = max([len(str(s)) for s in df.index.values] + [len(str(df.index.name))])
        col_widths = [idx_max] + [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    else:
        col_widths = [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    return col_widths


def direct(df_SET, forecast_year):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RoR', 'S'])

    RoR = df_SET.iloc[1:]['RoR'].values
    SETi = df_SET.iloc[1:]['SET'].values
    init_S = df_SET.iloc[0]['SET']

    for t in range(0, (forecast_year * n_per_year) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year * n_per_year) + 1:
            df.loc[t]['RoR'] = RoR[t - 1]
            df.loc[t]['S'] = SETi[t - 1]

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')
    return df


def monte_carlo(df_SET, forecast_year):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'u.dt', 'S(u.dt)', 'N', 'N.sigma.sqrt(dt)', 'S(N.sigma.sqrt(dt))', 'dS', 'S', 'RoR'])

    init_S = df_SET.iloc[0]['SET']
    u = df_SET.iloc[1:]['RoR'].mean() * n_per_year
    sigma = df_SET.iloc[1:]['RoR'].std() * np.sqrt(n_per_year)
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
            df.loc[t]['RoR'] = (df.loc[t]['S'] - df.loc[t - 1]['S']) / df.loc[t - 1]['S']

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def bootstrap(df_SET, forecast_year):
    global n_per_year
    df = pd.DataFrame(columns=['Month', 'RoR', 'dS', 'S'])

    RoR = df_SET.iloc[1:]['RoR'].values
    RoR = resample(RoR, replace=True, n_samples=forecast_year * n_per_year, random_state=None)
    init_S = df_SET.iloc[0]['SET']

    for t in range(0, (forecast_year * 12) + 1):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t

        if t == 0:
            df.loc[t]['S'] = init_S
        elif t > 0 and (forecast_year * n_per_year) + 1:
            df.loc[t]['RoR'] = RoR[t - 1]
            df.loc[t]['dS'] = df.loc[t]['RoR'] * df.loc[t - 1]['S']
            df.loc[t]['S'] = df.loc[t - 1]['S'] + df.loc[t]['dS']

    df = df.fillna('')
    df['Month'] = df['Month'].astype('int')
    df = df.set_index('Month')

    return df


def LS(df_Price, forecast_year, init_Cash):
    global n_per_year
    global col_Transaction
    global Div_ReInvest
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_Price)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        df.loc[t]['Price'] = df_Price.loc[t]
        if t == 0:
            df.loc[t]['Shares Bought'] = init_Cash / df.loc[t]['Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
        elif t in range(1, forecast_year * n_per_year):
            if Div_ReInvest:
                df.loc[t]['Shares Bought'] = 0.0 if divmod(t, n_per_year)[1] != 0 else (init_Cash + df.loc[t - 1]['Net Cash']) / df.loc[t]['Price']
            else:
                df.loc[t]['Shares Bought'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash / df.loc[t]['Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI'] + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])
        elif t == forecast_year * n_per_year:
            df.loc[t]['Shares Bought'] = -df.loc[t - 1]['Shares Owned']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = 0.0
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
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


def DCA(df_Price, forecast_year, init_Cash):
    global n_per_year
    global col_Transaction
    global Div_ReInvest
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_Price)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        df.loc[t]['Price'] = df_Price.loc[t]
        if t == 0:
            df.loc[t]['Shares Bought'] = init_Cash / n_per_year / df.loc[t]['Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
        elif t in range(1, forecast_year * n_per_year):
            if Div_ReInvest and (divmod(t, n_per_year)[1] != 0):
                df.loc[t]['Shares Bought'] = df.loc[t - 1]['Net Cash'] / (n_per_year - divmod(t, n_per_year)[1]) / df.loc[t]['Price']
            else:
                df.loc[t]['Shares Bought'] = init_Cash / n_per_year / df.loc[t]['Price']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI'] + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])
        elif t == forecast_year * n_per_year:
            df.loc[t]['Shares Bought'] = -df.loc[t - 1]['Shares Owned']
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = 0.0
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
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


def VA(df_Price, VA_Growth, VA_LimitBuy, forecast_year, init_Cash):
    global n_per_year
    global col_Transaction
    global Div_ReInvest
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_Price)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Month'] = t
        df.loc[t]['Price'] = df_Price.loc[t]
        if t == 0:
            df.loc[t]['Required Value'] = init_Cash / n_per_year
            diff = df.loc[t]['Required Value']
            df.loc[t]['Shares Bought'] = (init_Cash / df.loc[t]['Price']) if diff > init_Cash else (diff / df.loc[t]['Price'])
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
        elif t in range(1, forecast_year * n_per_year):
            df.loc[t]['Required Value'] = init_Cash / n_per_year + (df.loc[t - 1]['Required Value'] * (1 + VA_Growth / n_per_year / 100))
            diff = df.loc[t]['Required Value'] - (df.loc[t]['Price'] * df.loc[t - 1]['Shares Owned'])
            df.loc[t]['Shares Bought'] = (df.loc[t - 1]['Net Cash'] / df.loc[t]['Price']) if diff > df.loc[t - 1]['Net Cash'] else (diff / df.loc[t]['Price'])
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = 0.0 if divmod(t, n_per_year)[1] != 0 else init_Cash
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
            df.loc[t]['Net Cash'] = df.loc[t]['CFF'] + df.loc[t]['CFI'] + df.loc[t - 1]['Net Cash']
            df.loc[t]['Net Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Net Cash']
            df.loc[t]['Total Cost'] = -df.loc[t]['CFI'] + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['RoR'] = (df.loc[t]['Net Wealth'] - (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) - df.loc[t]['CFF']) / (
                    (df.loc[t - 1]['Net Wealth'] if t != 1 else init_Cash) + df.loc[t]['CFF'])
        elif t == forecast_year * n_per_year:
            df.loc[t]['Required Value'] = 0.0
            diff = df.loc[t]['Required Value'] - (df.loc[t]['Price'] * df.loc[t - 1]['Shares Owned'])
            df.loc[t]['Shares Bought'] = (df.loc[t - 1]['Net Cash'] / df.loc[t]['Price']) if diff > df.loc[t - 1]['Net Cash'] else (diff / df.loc[t]['Price'])
            df.loc[t]['Shares Owned'] = df.loc[t]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['CFF'] = 0.0
            df.loc[t]['CFI'] = -(df.loc[t]['Price'] * df.loc[t]['Shares Bought'])
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


def simulation(method, df_SET, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Simulation
    global row_Simulation
    global col_Summary

    df_Simulation = {}
    df_Simulation['Summary'] = pd.DataFrame(columns=col_Simulation, index=row_Simulation)
    df_Summary = pd.DataFrame(columns=col_Summary)

    if method == 1:
        df_Price = direct(df_SET, forecast_year)
        writer = pd.ExcelWriter('output/DT_Sim_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': pct_fmt,
            'C': float_fmt,
        }
    elif method == 2:
        df_Price = monte_carlo(df_SET, forecast_year)
        writer = pd.ExcelWriter('output/MC_Sim_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': float_fmt,
            'C': float_fmt,
            'D': float_fmt,
            'E': float_fmt,
            'F': float_fmt,
            'G': float_fmt,
            'H': float_fmt,
            'I': pct_fmt,
        }
    elif method == 3:
        df_Price = bootstrap(df_SET, forecast_year)
        writer = pd.ExcelWriter('output/BS_Sim_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
        workbook = writer.book
        float_fmt = workbook.add_format({'num_format': '#,##0.00'})
        float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        body_fmt = {
            'B': pct_fmt,
            'C': float_fmt,
            'D': float_fmt,
        }

    if iter == 0:
        sheet_name = 'SET'
        df = df_Price.copy()
        df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 2, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    df_Simulation['LS'] = LS(df_Price['S'].reset_index(drop=True), forecast_year, init_Cash)
    df_Simulation['DCA'] = DCA(df_Price['S'].reset_index(drop=True), forecast_year, init_Cash)
    VA_LimitBuy = np.inf
    df_Simulation['VA'] = VA(df_Price['S'].reset_index(drop=True), 0, VA_LimitBuy, forecast_year, init_Cash)
    df_Simulation['VA6'] = VA(df_Price['S'].reset_index(drop=True), 6, VA_LimitBuy, forecast_year, init_Cash)
    df_Simulation['VA12'] = VA(df_Price['S'].reset_index(drop=True), 12, VA_LimitBuy, forecast_year, init_Cash)
    df_Simulation['VA18'] = VA(df_Price['S'].reset_index(drop=True), 18, VA_LimitBuy, forecast_year, init_Cash)

    # Risk Free Rate 10Y = 1.8416, Risk Free Rate 5Y = 1.4760
    RiskFree = 1.8416 if forecast_year == 10 else 1.4760
    for row in row_Simulation:
        df_Simulation['Summary'].loc['Last', 'SET'] = df_Price['S'].iloc[-1]
        df_Simulation['Summary'].loc['Mean', 'SET'] = df_Price['RoR'].iloc[1:].mean() * n_per_year
        df_Simulation['Summary'].loc['Std', 'SET'] = df_Price['RoR'].iloc[1:].std() * np.sqrt(n_per_year)
        df_Simulation['Summary'].loc['SR', 'SET'] = (df_Simulation['Summary'].loc['Mean', 'SET'] - RiskFree / 100) / df_Simulation['Summary'].loc['Std', 'SET']
    # for column in col_Simulation:
    for column in ['LS', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']:
        df_Simulation['Summary'].loc['Avg. Cost', column] = df_Simulation[column]['Average Cost'].iloc[-1]
        df_Simulation['Summary'].loc['Mean', column] = df_Simulation[column]['RoR'].iloc[1:].mean() * n_per_year
        df_Simulation['Summary'].loc['Std', column] = df_Simulation[column]['RoR'].iloc[1:].std() * np.sqrt(n_per_year)
        df_Simulation['Summary'].loc['SR', column] = (df_Simulation['Summary'].loc['Mean', column] - RiskFree / 100) / df_Simulation['Summary'].loc['Std', column]
        df_Simulation['Summary'].loc['IRR', column] = ((1 + np.irr(df_Simulation[column]['CFI'].tolist())) ** n_per_year) - 1
    df_Simulation['Summary'] = df_Simulation['Summary'].fillna('')

    if iter == 0:
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

    body_fmt = [float_fmt, float_fmt, pct_fmt, pct_fmt, float2_fmt, pct_fmt]
    if iter == 0:
        sheet_name = 'Summary'
        df = df_Simulation['Summary'].copy()
        df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for row in range(df.shape[0]):
            worksheet.set_row(row + 1, None, body_fmt[row])
        for col, width in enumerate(get_col_widths(df, index=False), 1):
            worksheet.set_column(col, col, width + 4)
        writer.save()

    # Summary of IRR #
    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary['Iter'] = int(iter + 1)
    for row in ['Last', 'Mean', 'Std', 'SR']:
        df_Summary['SET_{}'.format(row)] = df_Simulation['Summary'].loc[row, 'SET']
    for column in ['LS', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']:
        for row in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR']:
            df_Summary['{}_{}'.format(row, column)] = df_Simulation['Summary'].loc[row, column]

    return df_Summary.values.tolist()


if __name__ == '__main__':

    # Price Dataframe #
    df_SET = pd.read_excel('data/SET_TR.xlsx', sheet_name='Sheet1')
    df_SET = df_SET.iloc[(len(df_SET.index) - (forecast_Year * n_per_year) - 1):]

    results = []
    pool = Pool()
    iter = iter if method != 1 else 1
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, method, df_SET, forecast_Year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_Summary = pd.DataFrame(results, columns=col_Summary, dtype='object')
    df_Summary.sort_values(by='Iter', inplace=True)

    df_Summary = df_Summary.append({}, ignore_index=True)
    df_Summary.iloc[-1]['Iter'] = 'Avg'
    df_Summary.iloc[-1, 4:] = df_Summary.iloc[0:-1, 4:].mean()
    df_Summary = df_Summary.fillna('')
    df_Summary = df_Summary.set_index('Iter')
    df_Summary.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], col.split('_')[-1]) for col in df_Summary.columns])
    print(df_Summary.drop(columns=['Name'], level=1))

    if method == 1:
        writer = pd.ExcelWriter('output/DT_Sum_{}Y_{}.xlsx'.format(forecast_Year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    elif method == 2:
        writer = pd.ExcelWriter('output/MC_Sum_{}Y_{}.xlsx'.format(forecast_Year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    elif method == 3:
        writer = pd.ExcelWriter('output/BS_Sum_{}Y_{}.xlsx'.format(forecast_Year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})

    if method == 1:
        writer = pd.ExcelWriter('output/DT_Sum_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    elif method == 2:
        writer = pd.ExcelWriter('output/MC_Sum_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
    elif method == 3:
        writer = pd.ExcelWriter('output/BS_Sum_{}Y_{}.xlsx'.format(forecast_year, pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')))
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
        'B': float_fmt,
        'C': pct_fmt,
        'D': pct_fmt,
        'E': float_fmt,
        'F': float_fmt,
        'G': float_fmt,
        'H': float_fmt,
        'I': float_fmt,
        'J': float_fmt,
        'K': float_fmt,
        'L': pct_fmt,
        'M': pct_fmt,
        'N': pct_fmt,
        'O': pct_fmt,
        'P': pct_fmt,
        'Q': pct_fmt,
        'R': pct_fmt,
        'S': pct_fmt,
        'T': pct_fmt,
        'U': pct_fmt,
        'V': pct_fmt,
        'W': pct_fmt,
        'X': float2_fmt,
        'Y': float2_fmt,
        'Z': float2_fmt,
        'AA': float2_fmt,
        'AB': float2_fmt,
        'AC': float2_fmt,
        'AD': pct_fmt,
        'AE': pct_fmt,
        'AF': pct_fmt,
        'AG': pct_fmt,
        'AH': pct_fmt,
        'AI': pct_fmt,
    }
    for col, width in enumerate(get_col_widths(df, index=False), 1):
        worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
    writer.save()
