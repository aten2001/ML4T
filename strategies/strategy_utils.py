import numpy as np
import pandas as pd

from util import get_data


def daily_returns(df_prices):
    daily_rets = df_prices.copy()
    daily_rets.values[1:, :] = df_prices.values[1:, :] - df_prices.values[:-1, :]
    daily_rets.values[0, :] = np.nan

    return daily_rets


def compute_portfolio_stats(port_val):
    cr = float((port_val.values[-1] / port_val.values[0]) - 1)

    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets.values[1:]

    adr = daily_rets.mean()

    sddr = daily_rets.std()

    return cr, adr, sddr


def compute_portvals_from_orders(orders_df, start_date, end_date, symbol, start_val=100000, commission=9.95,
                                 impact=0.005):
    df_prices = get_data([symbol], pd.date_range(start_date, end_date))
    del df_prices["SPY"]
    df_prices["Cash"] = np.ones(df_prices.shape[0])

    df_prices = df_prices.ffill()
    df_prices = df_prices.bfill()

    df_trades = pd.DataFrame(np.zeros(df_prices.shape), index=df_prices.index, columns=df_prices.columns)
    df_position = pd.DataFrame(np.zeros(df_prices.shape), index=df_prices.index, columns=df_prices.columns)

    current_position = 0

    for index, row in orders_df.iterrows():
        symbol, order, shares = row
        price = df_prices[symbol][index]
        cost = shares * price

        if order == "BUY":
            if current_position == 1:
                continue

            df_trades[symbol][index] += shares
            df_trades["Cash"][index] -= cost
            df_trades["Cash"][index] -= commission + cost * impact

            current_position += 1
        else:
            if current_position == -1:
                continue
            df_trades[symbol][index] -= shares
            df_trades["Cash"][index] += shares * price
            df_trades["Cash"][index] -= commission + cost * impact

            current_position -= 1

        df_position[symbol][index] = current_position

    df_trades.iloc[0]["Cash"] += start_val
    df_trades = df_trades.cumsum()

    df_value = df_prices * df_trades
    df_port_val = df_value.sum(axis=1)

    df_port_val = df_port_val.to_frame()
    df_port_val.columns = ["Portfolio"]

    return df_port_val, df_position


def compute_portvals_from_trades(df_trades, symbol, start_date, end_date, start_val=100000):
    df_prices = get_data([symbol], pd.date_range(start_date, end_date))
    del df_prices["SPY"]
    df_prices["Cash"] = np.ones(df_prices.shape[0])

    df_trades.iloc[0, 1] += start_val

    df_trades = df_trades.cumsum()
    df_value = df_prices * df_trades

    df_port_val = df_value.sum(axis=1)
    df_port_val = df_port_val.to_frame()

    df_port_val.columns = ["Portfolio"]

    return df_port_val
