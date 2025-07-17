import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

startingdate = "2020-01-31"    # Start date for historical data, setting this date so that the data doesn't get big 
enddate = datetime.date.today().strftime("%Y-%m-%d") # end date is today 
shortsmaperiod = int(input("Enter the Short SMA period ") or 20) # Short SMA period, default is 20
longsmaperiod = int(input("Enter the Long SMA period ") or 50) # Long SMA period, default is 50
startingcaptial = int(input("Enter the Starting Capital in USD ") or 10000) # Starting capital for backtesting, default is 10,000 USD
tickerforsp500 = "^GSPC" 

def get_stock_data(ticker, start, end):
    try: # this is to handle exceptions when a wrong ticker is being used
        data = yf.download(ticker, start=startingdate, end=enddate, auto_adjust=True) # ticker is scraped off of yfinance within the time bound stated and auto adjust is used to smoothe out things like stock splits 
        if data.empty:
            print(f"No data found for {ticker} in the specified date range.") # prints the ticker entered is wrong 
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns] # when you download the data it is usually in the form ('Close','TICKER') so we flatten it to just 'Close'
        print(f"Successfully downloaded data for {ticker}.") 
        # print("Downloaded data columns:", data.columns.tolist())
        return data
    except Exception as e: # this is for error handling 
        print(f"Error downloading data for {ticker}: {e}") # shows error message if there is an error
        return None

def apply_sma_crossover_strategy(data, short_window, long_window):
    data1 = data.copy(deep=True)  # making a copy of the data to apply sma
    data1['SMA_Short'] = data1['Close'].rolling(window=shortsmaperiod, min_periods=1).mean() #takes the close price for all the data frames and applies a rolling window of the duration of the period and takes the mean
    data1['SMA_Long'] = data1['Close'].rolling(window=longsmaperiod, min_periods=1).mean()
    if 'SMA_Short' in data1.columns and 'SMA_Long' in data1.columns:
        data1 = data1.dropna(subset=['SMA_Short', 'SMA_Long']).copy()
    else:
        print("Error: SMA columns not found in DataFrame after calculation.")
        return pd.DataFrame()
    data1['Signal'] = 0
    data1.loc[data1['SMA_Short'] > data1['SMA_Long'], 'Signal'] = 1 # buy signal 
    data1.loc[data1['SMA_Short'] < data1['SMA_Long'], 'Signal'] = -1 # sell signal 
    data1['Position'] = data1['Signal'].diff().fillna(0)
    print(f"Short SMA = {short_window} and Long SMA = {long_window}")
    return data1

def backtest_strategy(data1, startingcaptial, ticker_symbol):
    if 'Position' not in data1.columns:
        print("Error: Strategy not applied. 'Position' column missing.")
        return None
    portfolio = pd.DataFrame(index=data1.index) # Create a DataFrame to hold portfolio values
    portfolio['Holdings'] = 0.0
    portfolio['Cash'] = startingcaptial # Initialize cash with starting capital
    portfolio['Total'] = startingcaptial   # Initialize total portfolio value with starting capital
    if data1.empty:
        print("Empty strategy DataFrame provided for backtesting.")
        return None
    # Iterate through the DataFrame to simulate trades
    for i in range(1, len(data1)):
        today = data1.index[i]
        yesterday = data1.index[i-1]
        portfolio.loc[today, 'Cash'] = portfolio.loc[yesterday, 'Cash'] #loc is used to access row and column by label
        portfolio.loc[today, 'Holdings'] = portfolio.loc[yesterday, 'Holdings'] #transfer holdings from yesterday to today
        price = data1['Close'].iloc[i]
        signal = data1['Position'].iloc[i]
        if signal == 1 and portfolio.loc[today, 'Holdings'] == 0: #signal == 1 means buy signal and holdings is 0 means no shares are held
            shares = portfolio.loc[today, 'Cash'] // price #buy as many shares as possible with available cash, you can tweak this to make a sort of safety net 
            if shares > 0: # check if shares is greater than 0
                cost = shares * price # calculate cost of shares
                portfolio.loc[today, 'Holdings'] = shares # update holdings with number of shares bought
                portfolio.loc[today, 'Cash'] -= cost # deduct cost from cash
        elif signal == -1 and portfolio.loc[today, 'Holdings'] > 0:
            # Sell all held shares
            shares = portfolio.loc[today, 'Holdings']
            revenue = shares * price
            portfolio.loc[today, 'Cash'] += revenue
            portfolio.loc[today, 'Holdings'] = 0
        portfolio.loc[today, 'Total'] = portfolio.loc[today, 'Cash'] + portfolio.loc[today, 'Holdings'] * price
    portfolio['Daily_Return'] = portfolio['Total'].pct_change()
    portfolio['Cumulative_Return'] = (1 + portfolio['Daily_Return']).cumprod()
    portfolio['Cumulative_Return'].iat[0] = 1 # Set initial cumulative return to 1 (100%) to aviod propagation of 0
    final_value = portfolio['Total'].iloc[-1] # total value on last data frame 
    total_return = (portfolio['Cumulative_Return'].iloc[-1] - 1) * 100 # returns as percentage 
    print(f"\n--- Backtest Summary for {ticker_symbol} ---")
    print(f"Initial Capital: ${startingcaptial:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    return portfolio

def calculatesp500output(startingdate, enddate, portfolio_index):
    sp500_data = get_stock_data(tickerforsp500, startingdate, enddate) # getting s and p 500 data in the given time frame
    sp500_data['Daily_Return'] = sp500_data['Close'].pct_change() # this gives percentage change of all the close values 
    sp500_data['Cumulative_Return'] = (1 + sp500_data['Daily_Return']).cumprod() # calculates cumulative returns and increments by 1, for percentage change of increment 
    sp500_data['Cumulative_Return'].iat[0] = 1 # Set initial cumulative return to 1, as usually it is 0, to avoid propagation error
    aligned_sp500_returns = sp500_data['Cumulative_Return'].reindex(portfolio_index, method='ffill') # matched to cumulative values to subsiquent dates 
    return aligned_sp500_returns

def plot_strategy(data1, portfolio, ticker_symbol, sp500_cumulative_returns=None):
    if data1.empty or portfolio.empty: # no plot for empty data frame 
        print("Empty data: Cannot plot.")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True) # this creates two plots within the same graph 
    ax1.plot(data1['Close'], label='Close Price', alpha=0.7) # in plot one there is close price for the underlying that is mapped
    ax1.plot(data1['SMA_Short'], label=f'SMA {shortsmaperiod}', color='orange') # plot for short sma 
    ax1.plot(data1['SMA_Long'], label=f'SMA {longsmaperiod}', color='green') # plot for long sma 
    buys = data1[data1['Position'] == 1] # buys happen when the 1 and sells for -1 
    sells = data1[data1['Position'] == -1]
    buy_crossovers = data1[(data1['SMA_Short'] > data1['SMA_Long']) & (data1['SMA_Short'].shift(1) <= data1['SMA_Long'].shift(1))] # buying and selling for crossover
    sell_crossovers = data1[(data1['SMA_Short'] < data1['SMA_Long']) & (data1['SMA_Short'].shift(1) >= data1['SMA_Long'].shift(1))]
    for i, buy_date in enumerate(buy_crossovers.index): # drawing green and lines for instance of buying 
        label = 'Buy Crossover' if i == 0 else None
        ax1.axvline(buy_date, color='green', linestyle=':', alpha=0.5, label=label)

    for i, sell_date in enumerate(sell_crossovers.index): # drawing red lines for the instance of selling
        label = 'Sell Crossover' if i == 0 else None 
        ax1.axvline(sell_date, color='red', linestyle=':', alpha=0.5, label=label)

    
    ax1.set_title(f'{ticker_symbol}-SMA Crossover Strategy & Trade Signals') 
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(portfolio['Cumulative_Return']*100, label=f'{ticker_symbol} Strategy Cumulative Return (%)', color='purple', linewidth=2)

 
    ax2.plot(sp500_cumulative_returns * 100, label='S&P 500 Cumulative Return (%)', color='blue', linestyle='--', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7) # Horizontal line at 0% return
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Portfolio Performance vs. S&P 500')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    TICKER = input("Enter the Stock Ticker: ").strip().upper() or 'AAPL'
    print (f"\n")
    data = get_stock_data(TICKER, startingdate, enddate)
    if data is not None and not data.empty:
        strategydataforsma = apply_sma_crossover_strategy(data, shortsmaperiod, longsmaperiod)
        if not strategydataforsma.empty:
            portfolio = backtest_strategy(strategydataforsma, startingcaptial, TICKER)
            if portfolio is not None and not portfolio.empty:
                sp500_returns = calculatesp500output(startingdate, enddate, portfolio.index)
                plot_strategy(strategydataforsma, portfolio, TICKER, sp500_returns)
            else:
                print(f"Portfolio simulation failed.")
        else:
            print(f"Strategy DataFrame is empty. Cannot backtest.")
    else:
        print(f"Data download failed or no data available. Exiting.")
   
