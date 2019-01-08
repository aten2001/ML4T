import numpy as np
import pandas as pd

from util import get_data
from indicators_utils import daily_returns


def simple_moving_average(df_prices, lookback=14):
    sma = df_prices.rolling(window=lookback, min_periods=lookback).mean()
    return sma


def price_over_sma(df_prices, sma):
    psma = df_prices / sma
    return psma


def bollinger_band_percentage(df_prices, sma, lookback=14):
    rolling_std = df_prices.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)

    bbp = (df_prices - bottom_band) / (top_band - bottom_band)

    return bbp


def relative_strength_index(df_prices, lookback=14):
    daily_rets = daily_returns(df_prices=df_prices)

    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

    up_gain = df_prices.copy()
    up_gain.ix[:, :] = 0
    up_gain.values[lookback:, :] = up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]

    down_loss = df_prices.copy()
    down_loss.ix[:, :] = 0
    down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]

    rs = (up_gain / lookback) / (down_loss / lookback)
    rsi = 100 - (100 / (1 + rs))
    rsi.ix[:lookback, :] = np.nan

    rsi[rsi == np.inf] = 100

    return rsi


def momentum(df_prices, n):
    m = (df_prices / df_prices.shift(periods=n)) * 100

    return m


def stochastic_oscillator(sd, ed, symbol):
    df_high = get_data([symbol], pd.date_range(sd, ed), colname="High")
    del df_high["SPY"]
    df_low = get_data([symbol], pd.date_range(sd, ed), colname="Low")
    del df_low["SPY"]
    df_close = get_data([symbol], pd.date_range(sd, ed), colname="Close")
    del df_close["SPY"]

    df = pd.concat([df_high, df_low, df_close], axis=1)
    df.columns = ["High", "Low", "Close"]

    df['L14'] = df['Low'].rolling(window=14).min()
    df['H14'] = df['High'].rolling(window=14).max()

    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))

    df['%D'] = df['%K'].rolling(window=3).mean()

    return df
