import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter.utility
from matplotlib import style
from scipy import optimize

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
# style.use('ggplot')


if __name__ == '__main__':
    # Import Pickle #
    df_FundNAV = pd.read_pickle('data/FundNAV.pkl')
    df_FundDiv = pd.read_pickle('data/FundDiv.pkl')
    df_FundData = pd.read_pickle('data/FundData.pkl')

    FundType = ['Thailand Fund Equity Small/Mid-Cap', 'Thailand Fund Equity Large-Cap']
    df_FundNAV = df_FundNAV.loc[:, df_FundData['Morningstar Category'].isin(FundType).tolist()]
    total_year = 10
    n_per_year = 12
    df_FundNAV = df_FundNAV.loc[:, df_FundNAV.count() >= total_year * n_per_year + 1]
    df_FundNAV = df_FundNAV.iloc[:total_year * n_per_year + 1].sort_index()
    df_FundDiv = df_FundDiv.loc[df_FundNAV.index, df_FundNAV.columns].fillna(0)
    df_FundData = df_FundData.loc[df_FundNAV.columns, :]

    df_Div = df_FundDiv.loc[:, df_FundDiv.sum(axis=0) > 0]
    df_NonDiv = df_FundDiv.loc[:, df_FundDiv.sum(axis=0) <= 0]
    writer = pd.ExcelWriter('output/#FundType.xlsx')
    workbook = writer.book
    sheet_name = 'Div'
    df_Div.to_excel(writer, sheet_name=sheet_name)
    sheet_name = 'NonDiv'
    df_NonDiv.to_excel(writer, sheet_name=sheet_name)
    writer.save()
