import numpy as np
import pandas as pd

from util import get_data


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df.sort_index(inplace=True)

    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]

    symbols = set(orders_df["Symbol"])
    dates = pd.date_range(start_date, end_date)

    df_prices = get_data(list(symbols), dates)
    del df_prices["SPY"]
    df_prices["Cash"] = np.ones(df_prices.shape[0])

    df_prices = df_prices.ffill()
    df_prices = df_prices.bfill()

    index = df_prices.index
    columns = df_prices.columns

    df_trades = pd.DataFrame(np.zeros(df_prices.shape), index=index, columns=columns)

    for index, row in orders_df.iterrows():
        symbol, order, shares = row
        price = df_prices[symbol][index]
        cost = shares * price

        if order == "BUY":
            df_trades[symbol][index] += shares
            df_trades["Cash"][index] -= cost
            df_trades["Cash"][index] -= commission + cost * impact
        else:
            df_trades[symbol][index] -= shares
            df_trades["Cash"][index] += shares * price
            df_trades["Cash"][index] -= commission + cost * impact

    df_trades.loc[start_date]["Cash"] += start_val
    df_trades = df_trades.cumsum()

    df_value = df_prices * df_trades
    df_port_val = df_value.sum(axis=1)

    return df_port_val
