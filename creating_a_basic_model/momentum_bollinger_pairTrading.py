import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

START_DATE = "2023-01-01" #limited the date for better visualization
END_DATE = datetime.date.today().strftime("%Y-%m-%d") #bascially extracting today's date.
INITIAL_CAPITAL = 10000.0 # fixing basic capital as for pair trading, as sometimes it needs sufficient capital. 

def get_stock_data(ticker): # defining a function to get the data of the required stock
    data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True) # using the Yahoo Finance library to directly scrape the data, important is autoadjust, where if a certain commodity has a stock split, historical data is adjusted, current share is shown 
    if isinstance(data.columns, pd.MultiIndex): #only accesses the data if it is multi index and makes, in this case a tuple
        data.columns = [col[0] for col in data.columns] # extracts the first element in the tuple, is of the form ('Stock_ticker' , 'Open')
    return data

def sma_crossover(data, short=20, long=50):
    data = data.copy() # make a copy of the data that we got in the previous function 
    data['SMA_Short'] = data['Close'].rolling(window=short).mean() # 
    data['SMA_Long'] = data['Close'].rolling(window=long).mean()
    data.dropna(inplace=True)
    data['Signal'] = 0
    data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1
    data.loc[data['SMA_Short'] < data['SMA_Long'], 'Signal'] = -1
    data['Position'] = data['Signal'].diff().fillna(0)
    return data

def momentum_strategy(data, lookback=10):
    data = data.copy()
    data['Momentum'] = data['Close'].pct_change(lookback)
    data.dropna(inplace=True)
    data['Signal'] = 0
    data.loc[data['Momentum'] > 0, 'Signal'] = 1
    data.loc[data['Momentum'] < 0, 'Signal'] = -1
    data['Position'] = data['Signal'].diff().fillna(0)
    return data

def bollinger_strategy(data, window=20, num_std=2):
    data = data.copy()
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
    data['Upper'] = data['MA'] + num_std * data['STD']
    data['Lower'] = data['MA'] - num_std * data['STD']
    data.dropna(inplace=True)
    data['Signal'] = 0
    data.loc[data['Close'] < data['Lower'], 'Signal'] = 1
    data.loc[data['Close'] > data['Upper'], 'Signal'] = -1
    data['Position'] = data['Signal'].diff().fillna(0)
    return data

def pair_trading_strategy(stock1, stock2, z_thresh=1.0):
    data = pd.DataFrame({
        'A': stock1['Close'],
        'B': stock2['Close']
    }).dropna()

    data['Spread'] = data['A'] - data['B']
    mean = data['Spread'].rolling(20).mean()
    std = data['Spread'].rolling(20).std()
    data['Z'] = (data['Spread'] - mean) / std

    data['Signal'] = 0
    data.loc[data['Z'] > z_thresh, 'Signal'] = -1  
    data.loc[data['Z'] < -z_thresh, 'Signal'] = 1  
    data['Position'] = data['Signal'].diff().fillna(0)

    data['Close'] = data['A']  
    return data

def backtest_strategy(data, initial_capital):
    data = data.copy()
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Holdings'] = 0.0
    portfolio['Cash'] = initial_capital
    portfolio['Total'] = initial_capital

    for i in range(1, len(data)):
        today = data.index[i]
        yesterday = data.index[i - 1]
        price = data['Close'].iloc[i]
        signal = data['Position'].iloc[i]

        portfolio.loc[today, 'Cash'] = portfolio.loc[yesterday, 'Cash']
        portfolio.loc[today, 'Holdings'] = portfolio.loc[yesterday, 'Holdings']

        if signal == 1 and portfolio.loc[today, 'Holdings'] == 0:
            shares = portfolio.loc[today, 'Cash'] // price
            cost = shares * price
            portfolio.loc[today, 'Cash'] -= cost
            portfolio.loc[today, 'Holdings'] = shares

        elif signal == -1 and portfolio.loc[today, 'Holdings'] > 0:
            shares = portfolio.loc[today, 'Holdings']
            revenue = shares * price
            portfolio.loc[today, 'Cash'] += revenue
            portfolio.loc[today, 'Holdings'] = 0

        portfolio.loc[today, 'Total'] = portfolio.loc[today, 'Cash'] + \
                                        portfolio.loc[today, 'Holdings'] * price

    portfolio['Daily_Return'] = portfolio['Total'].pct_change()
    portfolio['Cumulative_Return'] = (1 + portfolio['Daily_Return']).cumprod()
    portfolio['Cumulative_Return'].iat[0] = 1
    return portfolio

def calculate_metrics(portfolio):
    daily = portfolio['Daily_Return'].dropna()
    sharpe = np.sqrt(252) * daily.mean() / daily.std()
    cumulative = (1 + daily).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return {
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(drawdown.min() * 100, 2),
        "Final Value": round(portfolio['Total'].iloc[-1], 2),
        "Total Return (%)": round((portfolio['Cumulative_Return'].iloc[-1] - 1) * 100, 2)
    }

def plot_strategy(data, portfolio, strategy_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(data['Close'], label='Close Price', alpha=0.7)
    if 'SMA_Short' in data.columns:
        ax1.plot(data['SMA_Short'], label='Short')
    if 'SMA_Long' in data.columns:
        ax1.plot(data['SMA_Long'], label='Long')
    if 'Upper' in data.columns:
        ax1.plot(data['Upper'], label='Upper Band', linestyle='--', alpha=0.5)
    if 'Lower' in data.columns:
        ax1.plot(data['Lower'], label='Lower Band', linestyle='--', alpha=0.5)
    buys = data[data['Position'] == 1]
    sells = data[data['Position'] == -1]
    ax1.plot(buys.index, buys['Close'], '^', markersize=10, color='green', label='Buy')
    ax1.plot(sells.index, sells['Close'], 'v', markersize=10, color='red', label='Sell')
    ax1.set_title(f'{strategy_name} - Price & Signals')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(portfolio['Cumulative_Return'] * 100, color='purple')
    ax2.set_title('Cumulative Return (%)')
    ax2.set_ylabel('%')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    print("Downloading data...")
    aapl = get_stock_data("AAPL")
    msft = get_stock_data("MSFT")
    results = {}
    boll_data = bollinger_strategy(aapl)
    boll_portfolio = backtest_strategy(boll_data, INITIAL_CAPITAL)
    results['Bollinger'] = calculate_metrics(boll_portfolio)
    plot_strategy(boll_data, boll_portfolio, "Bollinger Bands")
    pair_data = pair_trading_strategy(aapl, msft)
    pair_portfolio = backtest_strategy(pair_data, INITIAL_CAPITAL)
    results['Pair Trading'] = calculate_metrics(pair_portfolio)
    plot_strategy(pair_data, pair_portfolio, "Pair Trading (AAPL-MSFT)")
    for name, metrics in results.items():
        print(f"\n{name} Strategy:")
        for key, val in metrics.items():
            print(f"  {key}: {val}")
