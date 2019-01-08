import numpy as np
import pandas as pd

from indicators.indicators import simple_moving_average, price_over_sma, bollinger_band_percentage, momentum
from util import get_data


class ManualStrategy:

    def __init__(self):
        pass

    def test_policy(self, symbol, start_date, end_date):
        df_prices = get_data([symbol], pd.date_range(start_date, end_date))

        lookback = 37
        n = 10

        sma = simple_moving_average(df_prices=df_prices, lookback=lookback)
        bbp = bollinger_band_percentage(df_prices=df_prices, lookback=lookback, sma=sma)
        sma = price_over_sma(df_prices=df_prices, sma=sma)
        m = momentum(df_prices=df_prices, n=n)

        orders = df_prices.copy()
        orders.ix[:, :] = np.nan

        spy_m = m.copy()
        spy_m.values[:, :] = spy_m.ix[:, ["SPY"]]

        sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
        sma_cross[sma >= 1] = 1

        sma_cross[1:] = sma_cross.diff()
        sma_cross.ix[0] = 0

        orders[(sma < 0.95) & (bbp < 0) & (m < 90) & (spy_m > 90)] = 1
        orders[(sma > 1.05) & (bbp > 1) & (m > 125) & (spy_m < 125)] = -1

        orders[sma_cross != 0] = 0

        orders.ffill(inplace=True)
        orders.fillna(0, inplace=True)

        orders[1:] = orders.diff()
        orders.ix[0] = 0

        order_list = []

        for day in orders.index:
            if orders.ix[day, symbol] > 0:
                order_list.append([day.date(), symbol, "BUY", 1000])
            elif orders.ix[day, symbol] < 0:
                order_list.append([day.date(), symbol, "SELL", 1000])

        order_df = pd.DataFrame(order_list, columns=["Date", "Symbol", "Order", "Shares"])
        order_df = order_df.set_index(order_df['Date'], inplace=False).drop('Date', 1)

        return order_df
