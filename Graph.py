import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter.utility
from matplotlib import style
from scipy import stats

pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', 7)
pd.options.display.float_format = '{:.2f}'.format
# style.use('ggplot')
sns.set(font_scale=1.1)

if __name__ == '__main__':
    # Excel to Pickle #
    # df_1Y = pd.read_excel('data/#Summary_1Y.xlsx', sheet_name='Result', header=[0,1], index_col=[0])
    # df_1Y_Ind = {}
    # for ind in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']:
    #     df_1Y_Ind[ind] = df_1Y[ind].melt(var_name='Algorithm', value_vars=df_1Y[ind].columns, value_name=ind)
    # df_1Y = pd.concat([df_1Y_Ind[ind] for ind in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']], axis=1).T.drop_duplicates().T
    # df_1Y.to_pickle('data/Summary_1Y.pkl')
    #
    # df_3Y = pd.read_excel('data/#Summary_3Y.xlsx', sheet_name='Result', header=[0,1], index_col=[0])
    # df_3Y_Ind = {}
    # for ind in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']:
    #     df_3Y_Ind[ind] = df_3Y[ind].melt(var_name='Algorithm', value_vars=df_3Y[ind].columns, value_name=ind)
    # df_3Y = pd.concat([df_3Y_Ind[ind] for ind in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']], axis=1).T.drop_duplicates().T
    # df_3Y.to_pickle('data/Summary_3Y.pkl')
    #
    # df_5Y = pd.read_excel('data/#Summary_5Y.xlsx', sheet_name='Result', header=[0,1], index_col=[0])
    # df_5Y_Ind = {}
    # for ind in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']:
    #     df_5Y_Ind[ind] = df_5Y[ind].melt(var_name='Algorithm', value_vars=df_5Y[ind].columns, value_name=ind)
    # df_5Y = pd.concat([df_5Y_Ind[ind] for ind in ['Avg. Cost', 'Mean', 'Std', 'SR', 'IRR', 'Dividend']], axis=1).T.drop_duplicates().T
    # df_5Y.to_pickle('data/Summary_5Y.pkl')

    # Import Pickle #
    df = {}
    df['1Y'] = pd.read_pickle('data/Summary_1Y.pkl')
    df['3Y'] = pd.read_pickle('data/Summary_3Y.pkl')
    df['5Y'] = pd.read_pickle('data/Summary_5Y.pkl')

    fig = plt.figure(figsize=(10, 7), dpi=80)
    # ax = sns.distplot(df_1Y['Avg. Cost'], bins=20, hist_kws=dict(edgecolor='k', linewidth=1), kde_kws={'linestyle':'--'})

    for year in ['1Y', '3Y', '5Y']:
        ax = sns.kdeplot(df[year]['Avg. Cost'], linestyle='--', label=year)

    # bplot = sns.boxplot(y='Avg. Cost', x='Algorithm',
    #                     data=df_1Y,
    #                     width=0.5,
    #                     palette="colorblind")
    # bplot = sns.stripplot(y='Avg. Cost', x='Algorithm',
    #                       data=df_1Y,
    #                       jitter=True,
    #                       marker='o',
    #                       alpha=0.5,
    #                       color='black')
    plt.title = 'Average Cost'
    plt.show()