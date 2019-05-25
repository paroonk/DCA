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
# sns.set(font_scale=1.1)


def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


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

    graph = 'Avg. Cost'

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey='row', figsize=(10, 10), dpi=80)
    fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True, sharey='row', figsize=(10, 10), dpi=80)
    palette = plt.get_cmap('tab10')
    p_index = 0
    for row, year in enumerate(['1Y', '3Y', '5Y']):
        for col, algorithm in enumerate(['DCA', 'VA']):
            sns.distplot(df[year].loc[df[year]['Algorithm'] == algorithm][graph], bins=30, kde=False, label='{} {}'.format(algorithm, year), color=palette(p_index), hist_kws=dict(edgecolor='k', linewidth=1), ax=ax[row, col])
            sns.kdeplot(df[year].loc[df[year]['Algorithm'] == algorithm][graph], label='{} {}'.format(algorithm, year), linestyle='--', ax=ax2[row])
            p_index = p_index + 1
            ax[row, col].set_title('{} {}'.format(algorithm, year))
            ax[row, col].tick_params(labelbottom=True)
            ax[row, col].set(ylabel='Frequency')
        ax2[row].set_title('{}'.format(year))
        ax2[row].tick_params(labelbottom=True)

    fig.suptitle('Distributions of {}'.format(graph), y=1.0, fontsize=17)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig2.suptitle('Normal Distribution Curves of {}'.format(graph), y=1.0, fontsize=17)
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92)
    plt.show()

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