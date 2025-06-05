import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MeanReversionBacktester:
    def __init__(
        self, data, lookback=20, entry_z=-1.0, exit_z=0.0, slippage=0.0005, fee=0.0003,
        latency=1, volatility_lookback=60, capital=1e6, max_vol=0.02
    ):
        self.data = data.copy()
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.slippage = slippage
        self.fee = fee
        self.latency = latency
        self.volatility_lookback = volatility_lookback
        self.capital = capital
        self.max_vol = max_vol
        self._prepare_data()

    def _prepare_data(self):
        self.data['returns'] = self.data['close'].pct_change()
        self.data['mean'] = self.data['close'].rolling(self.lookback).mean()
        self.data['std'] = self.data['close'].rolling(self.lookback).std()
        self.data['zscore'] = (self.data['close'] - self.data['mean']) / self.data['std']
        self.data['volatility'] = (
            self.data['returns'].rolling(self.volatility_lookback).std() * np.sqrt(252 * 390)
        )
        self.data.dropna(inplace=True)

    def _generate_signals(self):
        self.data['signal'] = 0
        self.data.loc[self.data['zscore'] < self.entry_z, 'signal'] = 1
        self.data.loc[self.data['zscore'] > self.exit_z, 'signal'] = 0
        self.data['signal'] = self.data['signal'].ffill().shift(self.latency)
        self.data['signal'] = self.data['signal'].fillna(0)

    def _volatility_scaling(self):
        self.data['vol_scaled_position'] = (
            (self.capital * self.max_vol / self.data['volatility']).clip(upper=self.capital)
            / self.data['close']
        )
        self.data['vol_scaled_position'] = self.data['vol_scaled_position'].fillna(0)
        self.data['position'] = self.data['signal'] * self.data['vol_scaled_position']

    def _simulate_trades(self):
        self.data['position_change'] = self.data['position'].diff().fillna(0)
        trade_direction = np.sign(self.data['position_change'])
        self.data['trade_price'] = self.data['close'] * (
            1 + self.slippage * trade_direction
        )
        self.data['trade_price'] = self.data['trade_price'].fillna(self.data['close'])
        self.data['transaction_cost'] = (
            (self.data['trade_price'] * self.data['position_change'].abs()) * self.fee
        )
        self.data['holdings'] = self.data['position'] * self.data['close']
        self.data['cash'] = self.capital - (
            (self.data['trade_price'] * self.data['position_change']).cumsum() +
            self.data['transaction_cost'].cumsum()
        )
        self.data['portfolio'] = self.data['cash'] + self.data['holdings']
        self.data['strategy_return'] = self.data['portfolio'].pct_change().fillna(0)

    def run(self):
        self._generate_signals()
        self._volatility_scaling()
        self._simulate_trades()
        return self.data

    def evaluate_performance(self, plot=True):
        df = self.data.copy()
        cum_returns = (1 + df['strategy_return']).cumprod()
        daily_returns = df['strategy_return'].resample('1D').sum()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else np.nan
        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        max_drawdown = drawdown.min()
        days = (df.index[-1] - df.index[0]).days
        cagr = (cum_returns.iloc[-1]) ** (365 / days) - 1 if days > 0 else np.nan
        avg_gross_position = df['position'].abs().mean()
        total_traded = df['position_change'].abs().sum()
        turnover = total_traded / avg_gross_position if avg_gross_position > 0 else np.nan
        hit_rate = (df['strategy_return'] > 0).sum() / (df['strategy_return'] != 0).sum() if (df['strategy_return'] != 0).sum() > 0 else np.nan

        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"CAGR: {cagr:.2%}")
        print(f"Turnover: {turnover:.2f}")
        print(f"Hit Rate: {hit_rate:.2%}")

        if plot:
            self._plot_performance(cum_returns, drawdown, sharpe, cagr, turnover, hit_rate, max_drawdown)

        return {
            "Sharpe": sharpe,
            "Max Drawdown": max_drawdown,
            "CAGR": cagr,
            "Turnover": turnover,
            "Hit Rate": hit_rate
        }

    def _plot_performance(self, cum_returns, drawdown, sharpe, cagr, turnover, hit_rate, max_drawdown):
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(cum_returns.index, cum_returns, label="Equity Curve")
        plt.title("Cumulative Returns")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Returns")
        plt.grid()
        plt.legend()

        # Annotate metrics on plot
        textstr = (
            f"Sharpe: {sharpe:.2f}\n"
            f"CAGR: {cagr:.2%}\n"
            f"Turnover: {turnover:.2f}\n"
            f"Hit Rate: {hit_rate:.2%}\n"
            f"Max Drawdown: {max_drawdown:.2%}"
        )
        plt.gca().text(
            0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1)
        )

        plt.subplot(2, 1, 2)
        plt.plot(drawdown.index, drawdown, color='red', label="Drawdown")
        plt.title("Drawdown")
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


#Example

np.random.seed(123)
minutes = pd.date_range("2025-01-01 09:30", periods=1560, freq="T")  # 4 trading days

# Sinusoidal mean-reverting process with drift and noise
base = 100 + np.linspace(0, 2, len(minutes))  # upward drift
oscillation = 2 * np.sin(np.linspace(0, 20 * np.pi, len(minutes)))  # oscillation
noise = np.random.normal(0, 0.2, len(minutes))  # adding noise
price = base + oscillation + noise

data = pd.DataFrame(index=minutes)
data['close'] = price
data['open'] = data['close'].shift(1).fillna(method='bfill')
data['high'] = data[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, 0.05, size=len(minutes)))
data['low'] = data[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, 0.05, size=len(minutes)))
data['volume'] = np.random.randint(100, 1000, size=len(minutes))



bt = MeanReversionBacktester(data)
bt.run()
bt.evaluate_performance()
