import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup, NavigableString, Comment
import numpy as np
import pandas as pd
import pickle
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)


def get_col_widths(dataframe):
    # First we find the maximum length of the index column
    idx_max = max([len(str(s)) for s in dataframe.index.values] + [len(str(dataframe.index.name))])
    # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
    return [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [len(col)]) for col in dataframe.columns]


### Stock Select ###
stock_code = 'hmpro'
driver = webdriver.Chrome()
driver.get('https://accounts.jitta.com/login?applicationName=jittadotcoms&redirectUrl=/stock/bkk:{}/financial'.format(
    stock_code))
username = driver.find_element_by_xpath('//*[@id="__next"]/div/div[3]/div/div/div[5]/div/input')
username.send_keys('paroonk@hotmail.com')
password = driver.find_element_by_xpath('//*[@id="__next"]/div/div[3]/div/div/div[6]/div/input')
password.send_keys('bjm816438')
driver.find_element_by_xpath('//*[@id="__next"]/div/div[3]/div/div/div[7]/button[1]').click()
driver.implicitly_wait(15)

file_name = stock_code
writer = pd.ExcelWriter('{}.xlsx'.format(file_name.upper()))
workbook = writer.book

### Balance Statement ###
sheet_name = 'BS'
driver.find_element_by_xpath(
    '//*[@id="app"]/div/div[4]/div/div/div/div[3]/div[1]/div/div/div/div[1]/div/div/div[1]/div[2]/button[2]').click()
soup = BeautifulSoup(driver.page_source, 'lxml')

data = soup.find('div', {'class': 'Table__TableWrapper-s15k9hcr-0 ebQcFv'})
df_list = []
for div in data.find_all('div'):
    for child in div.children:
        if isinstance(child, NavigableString) and not isinstance(child, Comment) and str(child).strip() != "":
            df_list.append('{}'.format(str(child).strip()))
df_list.remove('Per Share Items')
df_list.remove('Supplemental Items')
df_list = np.array(df_list).reshape(6, int(len(df_list) / 6))
df = pd.DataFrame(df_list).T
df.replace('-', 0, inplace=True)
col_list = df.columns.tolist()
col_list = col_list[0:1] + col_list[1:][::-1]
df = df[col_list]
df.to_csv('{}.csv'.format(sheet_name), sep='\t', index=False)
df = pd.read_csv('{}.csv'.format(sheet_name), sep='\t', thousands=',', header=1, index_col=0)
os.remove('{}.csv'.format(sheet_name))

df.to_excel(writer, sheet_name=sheet_name, header=True)
worksheet = writer.sheets[sheet_name]
for i, width in enumerate(get_col_widths(df)):
    worksheet.set_column(i, i, width)

### Income Statement ###
sheet_name = 'IS'
driver.find_element_by_xpath(
    '//*[@id="app"]/div/div[4]/div/div/div/div[3]/div[1]/div/div/div/div[1]/div/div/div[1]/div[1]/button[2]').click()
soup = BeautifulSoup(driver.page_source, 'lxml')

data = soup.find('div', {'class': 'Table__TableWrapper-s15k9hcr-0 ebQcFv'})
df_list = []
for div in data.find_all('div'):
    for child in div.children:
        if isinstance(child, NavigableString) and not isinstance(child, Comment) and str(child).strip() != "":
            df_list.append('{}'.format(str(child).strip()))
df_list.remove('ASSETS')
df_list.remove('LIABILITIES')
df_list.remove('EQUITIES')
df_list.remove('Supplemental Items')
df_list = np.array(df_list).reshape(6, int(len(df_list) / 6))
df = pd.DataFrame(df_list).T
df.replace('-', 0, inplace=True)
col_list = df.columns.tolist()
col_list = col_list[0:1] + col_list[1:][::-1]
df = df[col_list]
df.to_csv('{}.csv'.format(sheet_name), sep='\t', index=False)
df = pd.read_csv('{}.csv'.format(sheet_name), sep='\t', thousands=',', header=1, index_col=0)
os.remove('{}.csv'.format(sheet_name))
# print(df)

df.to_excel(writer, sheet_name=sheet_name, header=True)
worksheet = writer.sheets[sheet_name]
for i, width in enumerate(get_col_widths(df)):
    worksheet.set_column(i, i, width)


### Cashflow Statement ###
sheet_name = 'CF'
driver.find_element_by_xpath(
    '//*[@id="app"]/div/div[4]/div/div/div/div[3]/div[1]/div/div/div/div[1]/div/div/div[1]/div[1]/button[3]').click()
soup = BeautifulSoup(driver.page_source, 'lxml')

data = soup.find('div', {'class': 'Table__TableWrapper-s15k9hcr-0 ebQcFv'})
df_list = []
for div in data.find_all('div'):
    for child in div.children:
        if isinstance(child, NavigableString) and not isinstance(child, Comment) and str(child).strip() != "":
            df_list.append('{}'.format(str(child).strip()))
df_list.remove('CASH FROM OPERATING ACTIVITIES')
df_list.remove('CASH FROM INVESTING ACTIVITIES')
df_list.remove('CASH FROM FINANCING ACTIVITIES')
df_list.remove('Supplemental Items')
df_list = np.array(df_list).reshape(6, int(len(df_list) / 6))
df = pd.DataFrame(df_list).T
df.replace('-', 0, inplace=True)
col_list = df.columns.tolist()
col_list = col_list[0:1] + col_list[1:][::-1]
df = df[col_list]
df.to_csv('{}.csv'.format(sheet_name), sep='\t', index=False)
df = pd.read_csv('{}.csv'.format(sheet_name), sep='\t', thousands=',', header=1, index_col=0)
os.remove('{}.csv'.format(sheet_name))
# print(df)

df.to_excel(writer, sheet_name=sheet_name, header=True)
worksheet = writer.sheets[sheet_name]
for i, width in enumerate(get_col_widths(df)):
    worksheet.set_column(i, i, width)


### Save ###
writer.save()
driver.close()

