import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup, NavigableString, Comment
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)

driver = webdriver.Chrome()
driver.get('https://accounts.jitta.com/login?applicationName=jittadotcoms&redirectUrl=/stock/bkk:vih/financial')

username = driver.find_element_by_xpath('//*[@id="__next"]/div/div[3]/div/div/div[5]/div/input')
username.send_keys('paroonk@hotmail.com')
password = driver.find_element_by_xpath('//*[@id="__next"]/div/div[3]/div/div/div[6]/div/input')
password.send_keys('bjm816438')

driver.find_element_by_xpath('//*[@id="__next"]/div/div[3]/div/div/div[7]/button[1]').click()

driver.implicitly_wait(15)
driver.find_element_by_xpath('//*[@id="app"]/div/div[4]/div/div/div/div[3]/div[1]/div/div/div/div[1]/div/div/div[1]/div[2]/button[2]').click()

soup = BeautifulSoup(driver.page_source, 'lxml')
data = soup.find('div', {'class': 'Table__TableWrapper-s15k9hcr-0 ebQcFv'})
df_list = []
for div in data.find_all('div'):
    for child in div.children:
        if isinstance(child, NavigableString) and not isinstance(child, Comment) and str(child).strip() != "":
            #print('"{}"'.format(str(child).strip()))
            df_list.append('{}'.format(str(child).strip()))
df_list.remove('Per Share Items')
df_list.remove('Supplemental Items')
df_list = np.array(df_list).reshape(6, int(len(df_list)/6))
df = pd.DataFrame(df_list).T
print(df)
df.to_csv('Test', sep='\t', encoding='utf-8')