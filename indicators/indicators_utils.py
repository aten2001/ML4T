import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indicators.indicators import simple_moving_average, price_over_sma, bollinger_band_percentage, \
    relative_strength_index, momentum, stochastic_oscillator
from util import get_data


def daily_returns(df_prices):
    daily_rets = df_prices.copy()
    daily_rets.values[1:, :] = df_prices.values[1:, :] - df_prices.values[:-1, :]
    daily_rets.values[0, :] = np.nan

    return daily_rets


def plot_indicators():
    symbol = "JPM"
    sd = dt.datetime(2008, 01, 01)
    ed = dt.datetime(2009, 12, 31)

    lookback = 37

    df_prices = get_data([symbol], pd.date_range(sd, ed))

    df_prices = df_prices / df_prices.iloc[0]

    # plot price over sma

    sma = simple_moving_average(df_prices=df_prices, lookback=lookback)
    psma = price_over_sma(df_prices=df_prices, sma=sma)

    f, (a) = plt.subplots(1, 1, figsize=(10, 8))
    a.plot(df_prices[symbol], label="Price")
    a.plot(sma[symbol], label="SMA")
    a.plot(psma[symbol], label="Price over SMA")
    a.legend()

    plt.xticks([sd, ed])
    xfmt = mdates.DateFormatter('%d-%m-%y')
    a.xaxis.set_major_formatter(xfmt)

    plt.savefig("price_over_sma.png")
    plt.clf()

    # plot bollinger band percentage

    rolling_std = df_prices.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)

    bbp = (df_prices - bottom_band) / (top_band - bottom_band)

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(10, 10))
    a0.plot(df_prices[symbol], label="Price")
    a0.plot(top_band[symbol], label="Top Band")
    a0.plot(bottom_band[symbol], label="Bottom Band")
    a0.legend()
    a1.plot(bbp[symbol], label="BB %B")
    a1.legend()

    plt.xticks([sd, ed])
    xfmt = mdates.DateFormatter('%d-%m-%y')
    a0.xaxis.set_major_formatter(xfmt)

    plt.savefig("bollinger_band_percentage.png")
    plt.clf()

    # plot momentum

    n = 10
    m = momentum(df_prices=df_prices, n=n)

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(8, 8))
    a0.plot(df_prices[symbol], label="Price")
    a0.legend()
    a1.plot(m[symbol], label="momentum")
    a1.legend()

    plt.xticks([sd, ed])
    xfmt = mdates.DateFormatter('%d-%m-%y')
    a0.xaxis.set_major_formatter(xfmt)

    plt.savefig("momentum.png")
    plt.clf()

    # plot stochastic oscillator

    stochastic_oscillator_df = stochastic_oscillator(sd, ed, symbol)

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(20, 8))
    a0.plot(df_prices[symbol], label="Price")
    a0.legend()
    a1.plot(stochastic_oscillator_df["%K"], label="%K")
    a1.plot(stochastic_oscillator_df["%D"], label="%D")
    a1.legend()

    plt.xticks([sd, ed])
    xfmt = mdates.DateFormatter('%d-%m-%y')
    a0.xaxis.set_major_formatter(xfmt)

    plt.savefig("stochastic_oscillator.png")
    plt.clf()


def build_orders(symbols, start_date, end_date, lookback=14):
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))

    sma = simple_moving_average(df_prices=df_prices, lookback=lookback)
    bbp = bollinger_band_percentage(df_prices=df_prices, lookback=lookback, sma=sma)
    sma = price_over_sma(df_prices=df_prices, sma=sma)
    rsi = relative_strength_index(df_prices=df_prices, lookback=lookback)
    m = momentum(df_prices=df_prices, n=10)

    orders = df_prices.copy()
    orders.ix[:, :] = np.nan

    spy_rsi = rsi.copy()
    spy_rsi.values[:, :] = spy_rsi.ix[:, ["SPY"]]

    sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
    sma_cross[sma >= 1] = 1

    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0

    orders[(sma < 0.95) & (bbp < 0) & (m < 100) & (spy_rsi > 30)] = 100
    orders[(sma > 1.05) & (bbp > 1) & (m > 100) & (spy_rsi < 70)] = -100

    orders[sma_cross != 0] = 0

    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)

    orders[1:] = orders.diff()
    orders.ix[0] = 0

    order_list = []

    for day in orders.index:
        for sym in symbols:
            if orders.ix[day, sym] > 0:
                order_list.append([day.date(), sym, "BUY", 100])
            elif orders.ix[day, sym] < 0:
                order_list.append([day.date(), sym, "SELL", 100])

    for order in order_list:
        print "    ".join(str(x) for x in order)
