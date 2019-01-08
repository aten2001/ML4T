import numpy as np
import pandas as pd

from strategy_utils import daily_returns
from util import get_data


class TheoreticallyOptimalStrategy:

    def __init__(self):
        pass

    def test_policy(self, symbol, start_date, end_date):
        df_prices = get_data([symbol], pd.date_range(start_date, end_date))
        del df_prices["SPY"]

        trades = df_prices.copy()
        trades.ix[:, :] = 0
        trades["Cash"] = np.zeros(trades.shape[0])

        daily_rets = daily_returns(df_prices)

        current_position = 0

        for i in range(1, df_prices.shape[0]):
            if daily_rets.iloc[i, 0] > 0 and current_position != 1:
                if current_position == -1:
                    trades.iloc[i - 1, 0] += 2000
                    trades.iloc[i - 1, 1] -= 2000 * df_prices.iloc[i - 1, 0]
                    current_position = 1
                else:
                    trades.iloc[i - 1, 0] += 1000
                    trades.iloc[i - 1, 1] -= 1000 * df_prices.iloc[i - 1, 0]
                    current_position = 1

            elif daily_rets.iloc[i, 0] < 0 and current_position != -1:
                if current_position == 1:
                    trades.iloc[i - 1, 0] -= 2000
                    trades.iloc[i - 1, 1] += 2000 * df_prices.iloc[i - 1, 0]
                    current_position = -1
                else:
                    trades.iloc[i - 1, 0] -= 1000
                    trades.iloc[i - 1, 1] += 1000 * df_prices.iloc[i - 1, 0]
                    current_position = -1

        return trades
