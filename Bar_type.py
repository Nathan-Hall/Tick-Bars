import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import time
import math
import os.path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot


def get_start_time(days_ago):
    now = dt.now()
    minus = timedelta(days=days_ago)
    start = round(((now-minus).timestamp())*1000)
    return start


def split_callpoints(data_length):
    data_mins = int(data_length * 24 * 60 // 5)
    amount_of_calls = data_mins//1000 if data_mins % 1000 == 0 else (data_mins + 1000 - data_mins % 1000)//1000
    amount_of_calls = int(math.ceil(amount_of_calls))
    split = []
    b = lambda x: x*5*60*1000 #to milliseconds
    first = get_start_time(data_length)
    for i in range(amount_of_calls):
        if i == amount_of_calls-1:
            limit = 1000-((amount_of_calls*1000)-data_mins)
            ms = b(limit)
        else:
            ms=b(1000)
        split.append([first, first+ms])
        first += b(1000)
        
    return split, amount_of_calls
        

def get__OHLC_data(symbol, interval, split, call_no):
    #max 1000 per request so
    data=[]
    for i in range(call_no):
        first = split[i][0]
        second = split[i][1]
        url='https://api.binance.com/api/v3/klines?symbol=%s&interval=%s&startTime=%s&endTime=%s&limit=%d' % (symbol, interval, first, second, 1000)
        data_temp = requests.get(url).json()
        data.extend(data_temp)
        
    if len(data) > 1:
        df = pd.DataFrame(data)
        df = df.iloc[:, [0,1,2,3,4,5,8]]
        df.columns = ['Time', 'Open','High','Low','Close','Volume','No. of Trades']
        df = df.set_index('Time')
        df.index = pd.to_datetime(df.index, unit='ms')
        df = df.astype(float)
        return df


def plot_OHLC(df):
    x = np.arange(0,len(df))
    fig, ax = plt.subplots(1, figsize=(12,6))
    df.reset_index(inplace=True)
    for idx, val in df.iterrows():
        # high/low lines
        plt.plot([x[idx], x[idx]], 
                 [val['Low'], val['High']], 
                 color='black')
        # open marker
        plt.plot([x[idx], x[idx]-0.1], 
                 [val['Open'], val['Open']], 
                 color='black')
        # close marker
        plt.plot([x[idx], x[idx]+0.1], 
                 [val['Close'], val['Close']], 
                 color='black')
    plt.show()

def create_OHLC():
    symbol='BTCUSDT'
    interval='5m'
    data_length=.1 #amount of time (in days) to get data back to
    split, call_no = split_callpoints(data_length) #split days into 1000 call increments
    df = get_OHLC_data(symbol, interval, split, call_no)
    plot_OHLC(df)
    
    return named


def get_trades(symbol, split, call_no):
    data=[]
    t0 = time.time()
    
    #max hour between start and end, so if calling 24 hours back, call for each hour
    for i in range(call_no):
        start = split[i][0]
        end = split[i][1]
        j=0
        index=0
        print(start)
        #get all trades from start to end time
        while True: 
            url='https://api.binance.com/api/v3/aggTrades?symbol=%s&startTime=%s&endTime=%s&limit=%d' % (symbol, start, end, 1000)
            data_temp = requests.get(url).json()
            
            #There is no way to tell if call is the last call necessary, so 
            #messy way to tell if it is the last trade or not
            if not j==0:
                if data_temp[-1]['T'] == data[-1]['T']:
                    break
            else:
                j+=1
                
            data.extend(data_temp)
            start = data[-1]['T']
            
            index += 1
            t1 = time.time()
            #avoid limits
            if index > 995 and t1-t0 < 59:
                print(t1-t0)
                time.sleep(60-(t1-t0))
                t0 = time.time()
                index = 0
                
    return data
        

def get_tick_data():
    if os.path.isfile('crypto_tick_data.csv'):
        with open('crypto_tick_data.csv', "r") as f:
            for line in f: pass
            # recent = line.strip().split(',')[0]
    else:
        data_length=5
        symbol='BTCUSDT'
    
        start = get_start_time(data_length)
        end = int(time.time()*1000)
        
        split, call_no = split_hours(start, end)
        data = get_trades(symbol, split, call_no)
        
        df = pd.DataFrame(data)
        df = df.drop(labels=['f','l','M'], axis=1)
        df.columns = ['ID', 'Price','Quantity','Time', 'Buyer']
        df = df.set_index('Time')
        df=df.astype(float)
        df.to_csv('crypto_tick_data.csv')
        
    df = pd.read_csv("crypto_tick_data.csv")
    df = df.set_index('Time')
    
    return df

def split_hours(start, end):
    b = lambda x: x*60*1000 #to milliseconds
    
    #50min to ms (binance start end must be less than an hour)
    fifty_ms = b(55)
    call_no = int(math.ceil((end - start)/fifty_ms)) 
    split = []
    done=False
    while done==False:
        if start + fifty_ms < end:
            split.append([start, start+fifty_ms])
            start += fifty_ms
        else:
            split.append([start, end])
            done=True
    
    return split, call_no


def create_tickbar():
    ticks_per_bar = 100
    data = get_tick_data()
    
    data.reset_index(inplace=True)
    count = int(math.ceil(data.shape[0]/ticks_per_bar))
            
    index = [i*ticks_per_bar for i in range(count)]
    
    tick_df = data[['Time', 'Price']].iloc[index,:].copy()
    

    
    return tick_df
    
   
    
def create_volbar():
    vol_per_bar = 280
    data = get_tick_data()
    data.reset_index(inplace=True)
    
    sumvals = np.frompyfunc(lambda a,b: a+b if a+b <= vol_per_bar else (a+b-vol_per_bar),2,1)
    data['cumvol'] = sumvals.accumulate(data['Quantity'], dtype=np.object)
    
    result_df = data[data['cumvol'] > data['cumvol'].shift(periods=-1)]
    graph_type = 'Volume'
    
    return result_df


def create_dollarbar():
    dollars_per_bar = 10_000_000
    data = get_tick_data()
    data.reset_index(inplace=True)

    data['Dollars'] = data['Quantity']*data['Price']
    sumvals = np.frompyfunc(lambda a,b: a+b if a+b <= dollars_per_bar else (a+b-dollars_per_bar),2,1)
    data['cumdols'] = sumvals.accumulate(data['Dollars'], dtype=np.object)
    
    result_df = data[data['cumdols'] > data['cumdols'].shift(periods=-1)]

    graph_type = 'Dollar'
    
    return result_df
    

def plot_tickbars(data, graph_type):
    tick_data = create_tickbar()
    vol_data = create_volbar()
    dollar_data = create_dollarbar()
    
    data_list = [tick_data, vol_data, dollar_data]
    data_2 = []
    
    for i, df in enumerate(data_list):
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')  
        hour_df = pd.DataFrame(df.groupby([df.Time.dt.year, df.Time.dt.month, df.Time.dt.day, df.Time.dt.hour]).Time.count())
        hour_df.index = hour_df.index.set_names(['Year','Month','Day', 'Hour'])
        hour_df = hour_df.rename(columns={'Time': 'Count'})
        hour_df.reset_index(inplace=True)
        hour_df['Time'] = pd.to_datetime(hour_df[['Year','Month','Day','Hour']], format='%Y%m%d%H')
        hour_df = hour_df.drop(labels=['Year','Month','Day','Hour'],axis=1)
        hour_df = hour_df.set_index('Time')
        hour_df['Average'] = hour_df.rolling(window=24)['Count'].mean()
        hour_df['Average'].plot()
        data_2.append(hour_df)
        
        data_list[i].reset_index(inplace=True)
        data_list[i] = data_list[i].set_index('Time')
        data_list[i].index = pd.to_datetime(data_list[i].index)
    
    results_average = pd.DataFrame({'Tick bars':data_2[0]['Average'],
                         'Volume bars':data_2[1]['Average'],
                         'Dollar bars':data_2[2]['Average']})
    
    results_real = pd.DataFrame({'Tick bars':data_2[0]['Count'],
                     'Volume bars':data_2[1]['Count'],
                     'Dollar bars':data_2[2]['Count']})
    
    results_price = pd.DataFrame({'Tick bars':data_list[0]['Price'],
                     'Volume bars':data_list[1]['Price'],
                     'Dollar bars':data_list[2]['Price']})
    
    results_price = results_price.ffill()
    fig, ax = plt.subplots(1, figsize=(12, 6))
    
    results_average.plot(title="Average frequency of tick, volume and dollar bars",
                xlabel='Time', ylabel='Frequency')
    
    results_real.plot(title="Frequency of tick, volume and dollar bars",
            xlabel='Time', ylabel='Frequency')
    
    results_price.plot(title="Tick, Volume and Dollar bar ts",
         xlabel='Time', ylabel='Price')

    fig = sns.kdeplot(results_real['Tick bars'], color="r", fill=False, label='Tick bars')
    fig = sns.kdeplot(results_real['Volume bars'], color="b", fill=False, label='Volume bars')
    fig = sns.kdeplot(results_real['Dollar bars'], color="orange", fill=False, label='Dollar bars')
    plt.legend()
    plt.show()
    

def imbalance_bars():
    data = get_tick_data()
    data.reset_index(inplace=True)
    data['Time'] = pd.to_datetime(data['Time'], unit='ms')  

    data['diff'] = data['Price'].diff()
    data.loc[(data['diff'] > 0), 'diff'] = 1
    data.loc[(data['diff'] < 0), 'diff'] = -1
    data['diff'] = data['diff'].mask(data['diff'] == 0).ffill(downcast='infer')
    
    data = data.set_index('Time')
    data['Price'].plot()
    data['diff'].plot(style='bo')
    
    window = data.shape[0]//1000
    data['exp_b'] = data['diff']*data['Price']*data['Quantity']
    data['exp_b'] = data['exp_b'].ewm(100, adjust = True).mean()
    data['exp_b'][:20000].plot()
    
    #initial guess bc no t's yet
    tick_len = pd.DataFrame([40])
    diff_list = data['diff'].fillna(method='bfill').tolist()
    b_list = data['exp_b'].fillna(method='bfill').tolist()
    tick = [0]
    b_list
    cond = [int(tick_len.iloc[0]) * b_list[0]]
    theta_t = 0
    exp_t = 0
    cumsum_list = []
    
    for i, val in enumerate(diff_list[:1000000]):
        #print(val)
        theta_t += val
        cumsum_list.append(theta_t)
        cond.append(cond[-1])
        
        if abs(theta_t) >= abs(cond[-1]) or (i-tick[-1])>500:
            tick_len.loc[len(tick_len)] = i - tick[-1]
            exp_t = int(tick_len.ewm(20).mean().iloc[-1])
            del cond[-1]
            cond.append(exp_t * b_list[i])
            theta_t = 0
            tick.append(i)
        if (i % 1000) == 0:
            print(i)

    full = []
    
    for i, val in enumerate(tick):
        high = data.Price[tick[i-1]:val].max()
        low = data.Price[tick[i-1]:val].min()
        open_p = data.Price[tick[i-1]]
        close_p = data.Price[val]
        time = data.Time[val]
        
        full.append([time, val, high, low, open_p, close_p])
    
    del full[0]
    
    if len(cond) > len(cumsum_list):
        del cond[-1]
    
    samp_df = pd.DataFrame(list(zip(cumsum_list,cond)))
    samp_df.columns = ['Cumsum', 'Threshold']
    samp_df['Threshold'] = samp_df['Threshold'].abs()
    samp_df['_Threshold_neg'] = samp_df['Threshold']*-1
    
    fig, ax = plt.subplots()
    ax = samp_df[:4000].plot(color=['b','red','red'], ax=ax)
    ax.legend(['Cumulative sum of signed ticks', '_', 'Threshold'])

    
    tib_df = pd.DataFrame(full)    
    tib_df.columns = ['Time', 'Tick_no', 'High', 'Low', 'Open', 'Close']
    
    fig = go.Figure(data=[go.Candlestick(x=tib_df['Tick_no'][:10], 
                                         open=tib_df['Open'][:10], 
                                         high=tib_df['High'][:10], 
                                         low=tib_df['Low'][:10],
                                         close=tib_df['Close'][:10])])
    plot(fig)
    
    
def main():
    print()

if __name__ == "__main__":
    main()