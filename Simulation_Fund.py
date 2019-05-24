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
col_Transaction = ['Period', 'NAV', 'Bid Price', 'Offer Price', 'Shares Owned', 'Shares Bought', 'Required Value', 'Portfolio Value',
                   'Cash', 'Wealth', 'Total Cost', 'Average Cost', 'DPS', 'Div After Tax', 'CFF', 'CFI', 'TWR']
col_Simulation = ['NAV', 'DCA', 'VA', 'VA6', 'VA12', 'VA18']
row_Simulation = ['Last', 'Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']
col_Iter = ['Iter', 'Fund_Code', 'Fund_Period',
            'NAV_Last', 'NAV_Mean', 'NAV_Std', 'NAV_SR',
            'Avg. Cost_DCA', 'Avg. Cost_VA', 'Avg. Cost_VA6', 'Avg. Cost_VA12', 'Avg. Cost_VA18',
            'Mean_DCA', 'Mean_VA', 'Mean_VA6', 'Mean_VA12', 'Mean_VA18',
            'Std_DCA', 'Std_VA', 'Std_VA6', 'Std_VA12', 'Std_VA18',
            'SR_DCA', 'SR_VA', 'SR_VA6', 'SR_VA12', 'SR_VA18',
            'IRR_DCA', 'IRR_VA', 'IRR_VA6', 'IRR_VA12', 'IRR_VA18',
            'Dividend_DCA', 'Dividend_VA', 'Dividend_VA6', 'Dividend_VA12', 'Dividend_VA18']

# Simulation Config #
forecast_year = 5
n_per_year = 12
init_Cash = 10000
income_Tax = 10
riskFree = 2.0324


def get_col_widths(df, index=True):
    if index:
        idx_max = [max([len(str(s)) for s in df.index.get_level_values(idx)] + [len(str(idx))]) for idx in df.index.names]
        col_widths = idx_max + [max([len(str(s)) for s in df[col].values] + ([len(str(x)) for x in col] if df.columns.nlevels > 1 else [len(str(col))])) for col in df.columns]
    else:
        col_widths = [max([len(str(s)) for s in df[col].values] + ([len(str(x)) for x in col] if df.columns.nlevels > 1 else [len(str(col))])) for col in df.columns]
    return col_widths


def DCA(df_NAV, df_Div, df_Data, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Transaction
    global income_Tax
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_NAV)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Period'] = divmod(iter, (10 - forecast_year * n_per_year))[1] + 1 + t
        df.loc[t]['NAV'] = df_NAV.loc[t]
        df.loc[t]['Bid Price'] = np.floor(df.loc[t]['NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['Offer Price'] = np.ceil(df.loc[t]['NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000 if df_Data.loc['Actual Front Load (%)'].iloc[0] > 0.0 else df.loc[t]['NAV'] + 0.0001
        df.loc[t]['DPS'] = df_Div.loc[t]
        if t == 0:
            df.loc[t]['Shares Owned'] = 0.0
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Cash'] = 0.0
            df.loc[t]['Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Cash']
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['Shares Bought'] = df.loc[t]['CFF'] / df.loc[t]['Offer Price']
            df.loc[t]['Div After Tax'] = 0.0
            df.loc[t]['CFI'] = -((df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] >= 0.0 else df.loc[t]['Bid Price']) * df.loc[t]['Shares Bought']) + df.loc[t]['Div After Tax']
            df.loc[t]['Total Cost'] = 0.0
        elif t in range(1, forecast_year * n_per_year):
            df.loc[t]['Shares Owned'] = df.loc[t - 1]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Cash'] = df.loc[t - 1]['CFF'] + df.loc[t - 1]['CFI'] + df.loc[t - 1]['Cash']
            df.loc[t]['Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Cash']
            df.loc[t]['CFF'] = init_Cash
            df.loc[t]['Shares Bought'] = (df.loc[t]['CFF'] + df.loc[t]['Cash'] / (forecast_year * n_per_year - t)) / df.loc[t]['Offer Price']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFI'] = -((df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] >= 0.0 else df.loc[t]['Bid Price']) * df.loc[t]['Shares Bought']) + df.loc[t]['Div After Tax']
            df.loc[t]['Total Cost'] = (df.loc[t - 1]['Offer Price'] * df.loc[t - 1]['Shares Bought']) + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['TWR'] = (df.loc[t]['Wealth'] / (df.loc[t - 1]['Wealth'] + df.loc[t - 1]['CFF'])) - 1
        elif t == forecast_year * n_per_year:
            df.loc[t]['Shares Owned'] = df.loc[t - 1]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Cash'] = df.loc[t - 1]['CFF'] + df.loc[t - 1]['CFI'] + df.loc[t - 1]['Cash']
            df.loc[t]['Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Cash']
            df.loc[t]['CFF'] = 0.0
            df.loc[t]['Shares Bought'] = -df.loc[t]['Shares Owned']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFI'] = -((df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] >= 0.0 else df.loc[t]['Bid Price']) * df.loc[t]['Shares Bought']) + df.loc[t]['Div After Tax']
            df.loc[t]['Total Cost'] = (df.loc[t - 1]['Offer Price'] * df.loc[t - 1]['Shares Bought']) + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['TWR'] = (df.loc[t]['Wealth'] / (df.loc[t - 1]['Wealth'] + df.loc[t - 1]['CFF'])) - 1

    df = df.fillna('')
    df['Period'] = df['Period'].astype('int')
    df = df.set_index('Period')
    return df


def VA(df_NAV, df_Div, df_Data, VA_Growth, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Transaction
    global income_Tax
    df = pd.DataFrame(columns=col_Transaction)

    for t in range(0, len(df_NAV)):
        df = df.append({}, ignore_index=True)
        df.loc[t]['Period'] = divmod(iter, (10 - forecast_year * n_per_year))[1] + 1 + t
        df.loc[t]['NAV'] = df_NAV.loc[t]
        df.loc[t]['Bid Price'] = np.floor(df.loc[t]['NAV'] / (1 + df_Data.loc['Actual Deferred Load (%)'].iloc[0] / 100) * 10000) / 10000
        df.loc[t]['Offer Price'] = np.ceil(df.loc[t]['NAV'] * (1 + df_Data.loc['Actual Front Load (%)'].iloc[0] / 100) * 10000) / 10000 if df_Data.loc['Actual Front Load (%)'].iloc[0] > 0.0 else df.loc[t]['NAV'] + 0.0001
        df.loc[t]['DPS'] = df_Div.loc[t]
        if t == 0:
            df.loc[t]['Shares Owned'] = 0.0
            df.loc[t]['Required Value'] = init_Cash
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Cash'] = 0.0
            df.loc[t]['Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Cash']
            df.loc[t]['CFF'] = init_Cash
            diff = df.loc[t]['Required Value'] - (df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned'])
            df.loc[t]['Shares Bought'] = diff / df.loc[t]['Bid Price'] if diff < (df.loc[t]['CFF'] + df.loc[t]['Cash']) else (df.loc[t]['CFF'] + df.loc[t]['Cash']) / df.loc[t]['Offer Price']
            df.loc[t]['Div After Tax'] = 0.0
            df.loc[t]['CFI'] = -((df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] >= 0.0 else df.loc[t]['Bid Price']) * df.loc[t]['Shares Bought']) + df.loc[t]['Div After Tax']
            df.loc[t]['Total Cost'] = 0.0
        elif t in range(1, forecast_year * n_per_year):
            df.loc[t]['Shares Owned'] = df.loc[t - 1]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Required Value'] = init_Cash + (df.loc[t - 1]['Required Value'] * (1 + VA_Growth / n_per_year / 100))
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Cash'] = df.loc[t - 1]['CFF'] + df.loc[t - 1]['CFI'] + df.loc[t - 1]['Cash']
            df.loc[t]['Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Cash']
            df.loc[t]['CFF'] = init_Cash
            diff = df.loc[t]['Required Value'] - (df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned'])
            df.loc[t]['Shares Bought'] = diff / df.loc[t]['Bid Price'] if diff < (df.loc[t]['CFF'] + df.loc[t]['Cash']) else (df.loc[t]['CFF'] + df.loc[t]['Cash']) / df.loc[t]['Offer Price']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFI'] = -((df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] >= 0.0 else df.loc[t]['Bid Price']) * df.loc[t]['Shares Bought']) + df.loc[t]['Div After Tax']
            df.loc[t]['Total Cost'] = (df.loc[t - 1]['Offer Price'] * df.loc[t - 1]['Shares Bought']) + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['TWR'] = (df.loc[t]['Wealth'] / (df.loc[t - 1]['Wealth'] + df.loc[t - 1]['CFF'])) - 1
        elif t == forecast_year * n_per_year:
            df.loc[t]['Shares Owned'] = df.loc[t - 1]['Shares Bought'] + df.loc[t - 1]['Shares Owned']
            df.loc[t]['Required Value'] = 0.0
            df.loc[t]['Portfolio Value'] = df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned']
            df.loc[t]['Cash'] = df.loc[t - 1]['CFF'] + df.loc[t - 1]['CFI'] + df.loc[t - 1]['Cash']
            df.loc[t]['Wealth'] = df.loc[t]['Portfolio Value'] + df.loc[t]['Cash']
            df.loc[t]['CFF'] = 0.0
            diff = df.loc[t]['Required Value'] - (df.loc[t]['Bid Price'] * df.loc[t]['Shares Owned'])
            df.loc[t]['Shares Bought'] = diff / df.loc[t]['Bid Price'] if diff < (df.loc[t]['CFF'] + df.loc[t]['Cash']) else (df.loc[t]['CFF'] + df.loc[t]['Cash']) / df.loc[t]['Offer Price']
            df.loc[t]['Div After Tax'] = df.loc[t]['DPS'] * df.loc[t - 1]['Shares Owned'] * (1 - income_Tax / 100)
            df.loc[t]['CFI'] = -((df.loc[t]['Offer Price'] if df.loc[t]['Shares Bought'] >= 0.0 else df.loc[t]['Bid Price']) * df.loc[t]['Shares Bought']) + df.loc[t]['Div After Tax']
            df.loc[t]['Total Cost'] = (df.loc[t - 1]['Offer Price'] * df.loc[t - 1]['Shares Bought']) + df.loc[t - 1]['Total Cost']
            df.loc[t]['Average Cost'] = df.loc[t]['Total Cost'] / df.loc[t]['Shares Owned']
            df.loc[t]['TWR'] = (df.loc[t]['Wealth'] / (df.loc[t - 1]['Wealth'] + df.loc[t - 1]['CFF'])) - 1

    df = df.fillna('')
    df['Period'] = df['Period'].astype('int')
    df = df.set_index('Period')
    return df


def simulation(df_FundNAV, df_FundDiv, df_FundData, forecast_year, init_Cash, iter):
    global n_per_year
    global col_Simulation
    global row_Simulation
    global col_Iter
    global riskFree

    df_Simulation = {}
    df_Simulation['Summary'] = pd.DataFrame(columns=col_Simulation, index=row_Simulation)
    df_Result = pd.DataFrame(columns=col_Iter)

    fund = divmod(iter, (10 - forecast_year) * n_per_year)[0]
    period = divmod(iter, (10 - forecast_year) * n_per_year)[1]

    df_NAV = pd.DataFrame(df_FundNAV.iloc[period:period + forecast_year * n_per_year + 1, fund])
    df_NAV.columns = ['NAV']
    df_NAV['RoR'] = df_NAV.pct_change()
    df_NAV.reset_index(drop=True, inplace=True)
    df_NAV.index.name = 'Period'
    df_Div = pd.DataFrame(df_FundDiv.iloc[period:period + forecast_year * n_per_year + 1, fund])
    df_Div.columns = ['Div']
    df_Data = pd.DataFrame(df_FundData.iloc[fund, :])

    # selectFund = 'KFSDIV'
    # writer = pd.ExcelWriter('output/{}Y_P{}_{}.xlsx'.format(forecast_year, period + 1, selectFund))
    # workbook = writer.book
    # float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    # float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
    # pct_fmt = workbook.add_format({'num_format': '0.00%'})
    # body_fmt = {
    #     'A': None,
    #     'B': float_fmt,
    #     'C': float_fmt,
    #     'D': float_fmt,
    # }
    #
    # if df_Data.loc['Fund Code'].iloc[0] == selectFund:
    #     sheet_name = selectFund
    #     df = df_NAV.copy()
    #     df.index = range(period + 1, period + forecast_year * n_per_year + 2)
    #     df.iloc[:, :-1] = df.iloc[:, :-1].applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    #     df.iloc[:, -1:] = df.iloc[:, -1:].applymap(lambda x: round(x, 6) if isinstance(x, (int, float)) else x)
    #     df.to_excel(writer, sheet_name=sheet_name)
    #     worksheet = writer.sheets[sheet_name]
    #     for col, width in enumerate(get_col_widths(df)):
    #         worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    df_Simulation['DCA'] = DCA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, forecast_year, init_Cash, iter)
    df_Simulation['VA'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 0, forecast_year, init_Cash, iter)
    df_Simulation['VA6'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 6, forecast_year, init_Cash, iter)
    df_Simulation['VA12'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 12, forecast_year, init_Cash, iter)
    df_Simulation['VA18'] = VA(df_NAV['NAV'].reset_index(drop=True), df_Div['Div'].reset_index(drop=True), df_Data, 18, forecast_year, init_Cash, iter)

    for row in row_Simulation:
        df_Simulation['Summary'].loc['Last', 'NAV'] = df_NAV['NAV'].iloc[-1]
        df_Simulation['Summary'].loc['Mean', 'NAV'] = df_NAV['RoR'].iloc[1:].mean() * n_per_year
        df_Simulation['Summary'].loc['Std', 'NAV'] = df_NAV['RoR'].iloc[1:].std(ddof=0) * np.sqrt(n_per_year)
        df_Simulation['Summary'].loc['SR', 'NAV'] = (df_Simulation['Summary'].loc['Mean', 'NAV'] - riskFree / 100) / df_Simulation['Summary'].loc['Std', 'NAV']
    for column in ['DCA', 'VA', 'VA6', 'VA12', 'VA18']:
        df_Simulation['Summary'].loc['Avg. Cost', column] = df_Simulation[column]['Average Cost'].iloc[-1]
        df_Simulation['Summary'].loc['Mean', column] = df_Simulation[column]['TWR'].iloc[1:].mean() * n_per_year
        df_Simulation['Summary'].loc['Std', column] = df_Simulation[column]['TWR'].iloc[1:].std(ddof=0) * np.sqrt(n_per_year)
        df_Simulation['Summary'].loc['SR', column] = (df_Simulation['Summary'].loc['Mean', column] - riskFree / 100) / df_Simulation['Summary'].loc['Std', column]
        df_Simulation['Summary'].loc['IRR', column] = ((1 + np.irr(df_Simulation[column]['CFI'].tolist())) ** n_per_year) - 1
        df_Simulation['Summary'].loc['Dividend', column] = df_Simulation[column]['Div After Tax'].sum()
    df_Simulation['Summary'] = df_Simulation['Summary'].fillna('')

    # if df_Data.loc['Fund Code'].iloc[0] == selectFund:
    #     body_fmt = {
    #         'A': None,
    #         'B': float_fmt,
    #         'C': float_fmt,
    #         'D': float_fmt,
    #         'E': float_fmt,
    #         'F': float_fmt,
    #         'G': float_fmt,
    #         'H': float_fmt,
    #         'I': float_fmt,
    #         'J': float_fmt,
    #         'K': float_fmt,
    #         'L': float_fmt,
    #         'M': float_fmt,
    #         'N': float_fmt,
    #         'O': float_fmt,
    #         'P': float_fmt,
    #         'Q': pct_fmt,
    #     }
    #     for Algo in col_Simulation[1:]:
    #         sheet_name = '{}'.format(Algo)
    #         df = df_Simulation[Algo].copy()
    #         df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    #         df.to_excel(writer, sheet_name=sheet_name)
    #         worksheet = writer.sheets[sheet_name]
    #         for col, width in enumerate(get_col_widths(df)):
    #             worksheet.set_column(col, col, width + 2, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])
    #
    # body_fmt = {
    #     '2': float_fmt,
    #     '3': float_fmt,
    #     '4': pct_fmt,
    #     '5': pct_fmt,
    #     '6': float2_fmt,
    #     '7': pct_fmt,
    #     '8': float_fmt,
    # }
    # if df_Data.loc['Fund Code'].iloc[0] == selectFund:
    #     sheet_name = 'Summary'
    #     df = df_Simulation['Summary'].copy()
    #     df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    #     df.to_excel(writer, sheet_name=sheet_name)
    #     worksheet = writer.sheets[sheet_name]
    #     for row in range(df.shape[0]):
    #         worksheet.set_row(row + 1, None, body_fmt[str(row + 2)])
    #     for col, width in enumerate(get_col_widths(df)):
    #         worksheet.set_column(col, col, width + 1)
    #     writer.save()

    # Summary #
    df_Result = df_Result.append({}, ignore_index=True)
    df_Result['Iter'] = int(iter + 1)
    df_Result['Fund_Code'] = df_FundData.loc[df_FundNAV.columns[fund], 'Fund Code']
    df_Result['Fund_Period'] = int(period + 1)
    for row in ['Last', 'Mean', 'Std', 'SR']:
        df_Result['NAV_{}'.format(row)] = df_Simulation['Summary'].loc[row, 'NAV']
    for column in col_Simulation[1:]:
        for row in row_Simulation[1:]:
            df_Result['{}_{}'.format(row, column)] = df_Simulation['Summary'].loc[row, column]

    return df_Result.values.tolist()


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
    # FundSelect = ['KFSDIV']
    # df_FundNAV = df_FundNAV.loc[:, df_FundData['Fund Code'].isin(FundSelect).tolist()]
    total_year = 10
    df_FundNAV = df_FundNAV.loc[:, df_FundNAV.count() >= total_year * n_per_year + 1]
    df_FundNAV = df_FundNAV.iloc[:total_year * n_per_year + 1].sort_index()
    # todo Test
    # df_FundNAV = df_FundNAV.iloc[:, 0:5]

    df_FundDiv = df_FundDiv.loc[df_FundNAV.index, df_FundNAV.columns].fillna(0)
    df_FundData = df_FundData.loc[df_FundNAV.columns, :]

    results = []
    pool = Pool()
    fund_number = df_FundNAV.shape[1]
    period_number = (total_year - forecast_year) * n_per_year
    iter = fund_number * period_number
    for result in tqdm.tqdm(pool.imap_unordered(partial(simulation, df_FundNAV, df_FundDiv, df_FundData, forecast_year, init_Cash), range(iter)), total=iter):
        results.extend(result)

    df_Iter = {}
    df_Iter['Result'] = pd.DataFrame(results, columns=col_Iter, dtype='object')
    df_Iter['Result'].sort_values(by='Iter', inplace=True)

    df_Iter['Result'] = df_Iter['Result'].fillna('')
    df_Iter['Result'] = df_Iter['Result'].set_index('Iter')
    df_Iter['Result'].columns = pd.MultiIndex.from_tuples([(col.split('_')[0], col.split('_')[-1]) for col in df_Iter['Result'].columns])

    writer = pd.ExcelWriter('output/#Summary_{}Y.xlsx'.format(forecast_year))
    workbook = writer.book
    float_fmt = workbook.add_format({'num_format': '#,##0.00'})
    float2_fmt = workbook.add_format({'num_format': '#,##0.0000'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    text_fmt = workbook.add_format({'align': 'left'})

    sheet_name = 'Result'
    df = df_Iter['Result'].copy()
    df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    df.to_excel(writer, sheet_name=sheet_name)
    worksheet = writer.sheets[sheet_name]
    body_fmt = {
        'A': None,
        'B': text_fmt,
        'C': text_fmt,
        'D': float_fmt,
        'E': pct_fmt,
        'F': pct_fmt,
        'G': float_fmt,
        'H': float_fmt,
        'I': float_fmt,
        'J': float_fmt,
        'K': float_fmt,
        'L': float_fmt,
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
        'W': float2_fmt,
        'X': float2_fmt,
        'Y': float2_fmt,
        'Z': float2_fmt,
        'AA': float2_fmt,
        'AB': pct_fmt,
        'AC': pct_fmt,
        'AD': pct_fmt,
        'AE': pct_fmt,
        'AF': pct_fmt,
        'AG': float_fmt,
        'AH': float_fmt,
        'AI': float_fmt,
        'AJ': float_fmt,
        'AK': float_fmt,
    }
    for col, width in enumerate(get_col_widths(df)):
        worksheet.set_column(col, col, width + 1, body_fmt[xlsxwriter.utility.xl_col_to_name(col)])

    df_Iter['Summary'] = pd.DataFrame(columns=col_Simulation, index=row_Simulation)
    for row in ['Last', 'Mean', 'Std', 'SR']:
        df_Iter['Summary'].loc[row, 'NAV'] = df_Iter['Result']['NAV'][row].mean()
    for column in ['DCA', 'VA', 'VA6', 'VA12', 'VA18']:
        for row in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']:
            df_Iter['Summary'].loc[row, column] = df_Iter['Result'][row][column].mean()

    body_fmt = {
        '2': float_fmt,
        '3': float_fmt,
        '4': pct_fmt,
        '5': pct_fmt,
        '6': float2_fmt,
        '7': pct_fmt,
        '8': float_fmt,
    }
    sheet_name = 'Summary'
    df = df_Iter['Summary'].copy()
    df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    df.to_excel(writer, sheet_name=sheet_name)
    worksheet = writer.sheets[sheet_name]
    for row in range(df.shape[0]):
        worksheet.set_row(row + 1, None, body_fmt[str(row + 2)])
    for col, width in enumerate(get_col_widths(df)):
        worksheet.set_column(col, col, width + 1)
    writer.save()
