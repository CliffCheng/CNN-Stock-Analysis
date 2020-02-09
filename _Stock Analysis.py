#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style 
import pandas_datareader.data as web
import time
from tqdm import tqdm_notebook as tqdm


import schedule


import requests
from collections import OrderedDict
import bs4 as bs
import pickle


from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

from fastai.vision import *
from fastai.metrics import error_rate

import pathlib


# In[3]:


def save_sp500_tickers():
    ''' This function goes to the S&P500 wikipedia page and web scrapes the tickers from the list.'''
    # Get S&P500 list from wikipedia table using beautiful soup. Save in pickle file.
    
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class','wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        if '.' in ticker:
            ticker = ticker.replace('.','-')
        tickers.append(ticker)
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)
    

#save_sp500_tickers()


# In[4]:


def get_data_from_yahoo(reload_sp500 = False):
    '''This function is used incase Alpha Vantage has issues. Yahoo only provides ADJ Closes.'''
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()
    
    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


# In[10]:


def API_Data_Alpha(tickers):
    '''
    Opens ticker list data and retrieves API data from Alpha Vantage. 
    Alpha Vantage only allows 5 calls per minute or 500 calls per day.
    Imbedded timer runs a call, waits 60 seconds, and then runs the next call in the list.
    
    Input:
    tickers = Name of ticker (i.e. 'TSLA' for Tesla stock).
    '''
    n = 0
    with tqdm(total = len(tickers)) as pbar:
        while n < len(tickers):
            rsi_dataframe(tickers[n])
            time.sleep(60)
            n += 1
            pbar.update(1)
        return (f'{tickers[n]} Complete')



with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
directory = os.listdir()

'''
Make sure this you are in the directory that has a list of all the stock names.
'''

def get_missing_stocks(tickers, directory):
    '''
    Input:
    tickers = Name of ticker (i.e. 'TSLA' for Tesla stock).
    directory: Current directory (os.listdir()).
    '''
    missing_stocks = list(set(tickers)-set(directory))
    n = 0
    with tqdm(total=len(missing_stocks)) as pbar:
        while n < len(missing_stocks):
            rsi_dataframe(missing_stocks[n])
            time.sleep(60)
            n +=1
            pbar.update(1)
        return (f'{missing_stocks[n]} Complete')
#get_missing_stocks(tickers, directory)

def rsi_dataframe(stock):
    '''
    Use this function in the API calling, takes the data we want from Alpha Vantage and saves it into a CSV.
    Input:
    
    stock = Name of ticker (i.e. 'TSLA' for Tesla stock).
    
    Output:
    
    csv dataframe.
    
    '''
    stock = stock
    #api_key = '2KKOCMZP4P0O865A'
    api_key = 'YVGW80PFI6I0ZH6Z'
    period = 60
    
    ts = TimeSeries(key = api_key, output_format='pandas')
    data_ts = ts.get_daily_adjusted(stock.upper(), outputsize = 'full')
    
    #indicator
    ti = TechIndicators(key = api_key, output_format='pandas')
    data_rsi = ti.get_rsi(symbol=stock.upper(), interval = 'daily', time_period = period, series_type = 'close')
    data_aroon = ti.get_aroon(symbol=stock.upper(), interval = 'daily', time_period = period)
    data_mfi = ti.get_mfi(symbol=stock.upper(), interval = 'daily', time_period = period)
    data_dx = ti.get_dx(symbol=stock.upper(), interval = 'daily', time_period = period)

    #Get dfs removing unwanted lines. 
    df_adj_price = data_ts[0][period::]
    df_rsi = data_rsi[0][period::]
    df_aroon = data_aroon[0][period::]
    df_mfi = data_mfi[0][period::]
    df_dx = data_dx[0][period::]
    
    #Merge all dfs into one. 
    df_adj_rsi = df_adj_price.merge(df_rsi, on='date')
    df_aroon_mfi = df_aroon.merge(df_mfi, on='date')
    

    df_1_2 = df_adj_rsi.merge(df_aroon_mfi, on='date')
    df = df_1_2.merge(df_dx, on='date')

    df.to_csv(str(stock))
    #return print(df_adj_price, df_rsi, df_adx, df_ema)


# In[6]:


def csv_clean(stock):
    '''
    Input:
    
    stock = Name of ticker (i.e. 'TSLA' for Tesla stock). This will search the current directory for the csv file.
    
    Output:
    A dataframe that is a dictionary within a dictionary. Each month of every year is the key, within that key
    are the days consisting of that month with the corresponding data as values.
    
    '''
    df = pd.read_csv(stock)
    df = df[['date', '5. adjusted close' ,'RSI', 'Aroon Up', 'Aroon Down', 'MFI', 'DX']]
    df = df.rename({'5. adjusted close':'Adjusted_Close', 'date': 'Date'}, axis = 'columns')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Aroon'] = df['Aroon Up']-df['Aroon Down']
    df.drop(labels = ['Aroon Up', 'Aroon Down'], axis = 1)
    #df = df.groupby(pd.Grouper(freq='M'))

    df = dict(tuple(df.groupby([df['Date'].dt.year,df['Date'].dt.month])))
    #df = df.set_index(keys='Date')
    return df

def plot(df, stock):
    '''
    Inputs:
    df = csv_clean(stock). Cleaned dataframe. 
    stock = Name of ticker (i.e. 'TSLA' for Tesla stock).
    
    Outputs:
    A plotted chart that is saved to the respective Gain, Loss, or Neutral folder. 
    
    '''
    date_lst = list(dict.keys(df)) 
    n = 0 
    threshold = .05 
    while n < len(date_lst)-1: 
        x = df[date_lst[n]]['Date'] 
        Price = df[date_lst[n]]['Adjusted_Close'] 
        Price_next_month = df[date_lst[n+1]]['Adjusted_Close']

        Price_change = [1 - x/y for x, y in zip(Price, Price[1:])]
        RSI = df[date_lst[n]]['RSI']
        Aroon = df[date_lst[n]]['Aroon']
        MFI = df[date_lst[n]]['MFI']
        DX = df[date_lst[n]]['DX']

        # Creates 6 subplots.
        fig = plt.figure(figsize=(4,4))
        ax1 = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,2,3)
        ax3 = fig.add_subplot(3,2,4)
        ax4 = fig.add_subplot(3,2,5)
        ax5 = fig.add_subplot(3,2,6)
        
        # Scales y axis of subplots.
        ax1.set(ylim = (-.10,.10))
        ax2.set(ylim = (0,1))
        ax3.set(ylim = (-1, 1))
        ax4.set(ylim = (0,100))
        ax5.set(ylim = (0,100))
        
        # Makes x axis invisible. 
        ax1.xaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
        ax4.xaxis.set_visible(False)
        ax5.xaxis.set_visible(False)
        
        # Plot data. 
        ax1.plot(Price_change)
        ax2.plot(RSI/100 , 'tab:orange')
        ax3.plot(Aroon/100, 'tab:green')
        ax4.plot(MFI, 'tab:red')
        ax5.plot(DX, 'tab:blue')

        #If the next month's average is 5% higher than this month's average.
        if (1-(Price.mean() / Price_next_month.mean())) > threshold:
            plt.savefig(f'Stock Gains/{stock}_{date_lst[n]}_Gain')
            plt.close()
            n+=1


        #If the next month's average is 5% lower than this month's average.
        elif (1-(Price.mean() / Price_next_month.mean())) < -threshold:
            plt.savefig(f'Stock Losses/{stock}_{date_lst[n]}_Loss')
            plt.close()
            n+=1


        #If the next month's average is between +/-5% than this month's average.
        elif -threshold <=(1-(Price.mean() / Price_next_month.mean())) <= threshold:
            plt.savefig(f'Stock Neutrals/{stock}_{date_lst[n]}_Neutral')
            plt.close()
            n+=1
            


# In[6]:


def create_folders(run = False):
    '''
    Create classifier folders.
    '''
    stock_list = ['Stock Gains', 'Stock Neutrals', 'Stock Losses']
    if True:
        for i in stock_list:
            try:
                if not os.path.exists(i):
                    os.makedirs(i)
            except OSError:
                print ('Error: Creating directory. ' +  i)
#create_folders(True)


# In[9]:


def create_images(run = False):
    '''
    Run through the entire S&P500 list, finds the csv files, cleans the csv files, plots the csv files, 
    and then saves it in the corresponding classifier folder. 
    '''
    if True:
        i = 0
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)
        with tqdm(total=len(tickers)) as pbar:
            while i < len(tickers):
                df = csv_clean(tickers[i])
                plot(df, tickers[i])
                i +=1
                pbar.update(1)
#create_images(run = True)


# In[9]:



#Make sure this you are in the directory that has a list of all the stock names.
with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
directory = os.listdir()
    
def find_missing_stocks(tickers, directory):
    
    missing_stocks = list(set(tickers)-set(directory))
    return missing_stocks
#find_missing_stocks(tickers, directory)


# In[11]:


with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
directory = os.listdir()
#Make sure this you are in the directory that has a list of all the stock names.

def get_missing_stocks(tickers, directory):
    missing_stocks = list(set(tickers)-set(directory))
    n = 0
    with tqdm(total=len(missing_stocks)) as pbar:
        while n < len(missing_stocks):
            rsi_dataframe(missing_stocks[n])
            time.sleep(60)
            n +=1
            pbar.update(1)
        return (f'{missing_stocks[n]} Complete')
#get_missing_stocks(tickers, directory)


# In[ ]:


cd /Users/Cliff/MEGAsync/Coding/Stocks


# In[6]:


path = pathlib.Path().absolute()
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train = '.', valid_pct=.2)
data.show_batch(rows=3, figsize=(7,8))


# In[3]:


classes = ['Stock Gains', 'Stock Neutrals', 'Stock Losses']


# In[49]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[48]:


learn = cnn_learner(data, models.resnet34, metrics = error_rate)


# In[1]:


#learn.fit_one_cycle(4)
#See Google Colab.


# In[ ]:




