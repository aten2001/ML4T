import datetime as dt

import pandas as pd

import util as ut
from indicators.indicators import simple_moving_average, price_over_sma, bollinger_band_percentage, momentum
from learners.QLearner import QLearner
from strategy_utils import compute_portvals_from_trades


class StrategyLearner(object):

    def __init__(self):
        self.learner = None

        self.bins = None

    def get_indicators(self, df_prices):
        del df_prices['SPY']

        lookback = 37
        n = 14

        sma = simple_moving_average(df_prices=df_prices, lookback=lookback)
        bbp = bollinger_band_percentage(df_prices=df_prices, lookback=lookback, sma=sma)
        sma = price_over_sma(df_prices=df_prices, sma=sma)
        m = momentum(df_prices=df_prices, n=n)

        df = pd.concat([sma, bbp, m], axis=1)

        df.index = df_prices.index
        df.columns = ['sma', 'bbp', 'm']

        df.dropna(inplace=True)

        return df

    def discretize(self, df_indicators, steps):
        columns = df_indicators.columns
        df = pd.DataFrame(data=None, index=df_indicators.index, columns=columns)

        bins = []

        for c in columns:
            a, b = pd.qcut(df_indicators[c], steps, labels=False, retbins=True, duplicates='drop')

            df[c] = a

            bins.append(b)

        self.bins = bins

        df = df[columns[0]] * 10000 + df[columns[1]] * 100 + df[columns[0]]

        return df

    def discretize_test_policy(self, df_indicators):
        columns = df_indicators.columns
        df = pd.DataFrame(data=None, index=df_indicators.index, columns=columns)

        for i in range(len(columns)):
            df[columns[i]] = pd.cut(df_indicators[columns[i]], bins=self.bins[i], labels=False, include_lowest=True)

        df = df[columns[0]] * 10000 + df[columns[1]] * 100 + df[columns[0]]

        return df.fillna(0)

    def get_reward(self, curr_day, prev_day, position, df_prices):
        return float(((df_prices.ix[curr_day] / df_prices.ix[prev_day]) - 1)) * position

    def addEvidence(self, symbol="IBM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000):

        df_prices = ut.get_data([symbol], pd.date_range(sd, ed))

        df_indicators = self.get_indicators(df_prices)

        steps = 100
        discretized = self.discretize(df_indicators, steps)

        self.learner = QLearner(num_states=steps ** 4, num_actions=3)

        i = 0
        min_iter = 5
        max_iter = 100

        dates = df_indicators.index

        cum_ret = []

        while i < max_iter:
            position = 0
            df_trades = pd.DataFrame(data=None, index=df_prices.index, columns=[symbol, 'Cash'])

            s = discretized[dates[0]] * 10 + position
            self.learner.querysetstate(s)

            for j in range(1, len(dates)):
                date = dates[j]

                s = discretized[date] * 10 + position

                r = self.get_reward(date, dates[j - 1], position, df_prices)

                a = self.learner.query(s, r) - 1

                if a == 1 and position != 1:
                    if position == -1:
                        df_trades.ix[date, 0] = 2000
                        df_trades.ix[date, 1] = - 2000 * df_prices.ix[date, 0]
                    else:
                        df_trades.ix[date, 0] = 1000
                        df_trades.ix[date, 1] = - 1000 * df_prices.ix[date, 0]

                elif a == -1 and position != -1:
                    if position == 1:
                        df_trades.ix[date, 0] = - 2000
                        df_trades.ix[date, 1] = 2000 * df_prices.ix[date, 0]
                    else:
                        df_trades.ix[date, 0] = -1000
                        df_trades.ix[date, 1] = 1000 * df_prices.ix[date, 0]

                elif a == 0:
                    if position == 1:
                        df_trades.ix[date, 0] = - 1000
                        df_trades.ix[date, 1] = 1000 * df_prices.ix[date, 0]

                    elif position == -1:
                        df_trades.ix[date, 0] = 1000
                        df_trades.ix[date, 1] = - 1000 * df_prices.ix[date, 0]

                position = a

            df_trades.fillna(0, inplace=True)
            port_val = compute_portvals_from_trades(df_trades, symbol, sd, ed, sv)

            cr = float((port_val.values[-1] / port_val.values[0]) - 1)

            cum_ret.append(cr)

            if i > 1:

                converged = cum_ret[-1] == cum_ret[-2]

                if converged and i > min_iter:
                    break

            i += 1

    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1)):

        df_prices = ut.get_data([symbol], pd.date_range(sd, ed))

        df_indicators = self.get_indicators(df_prices)

        discretized = self.discretize_test_policy(df_indicators)

        dates = df_indicators.index

        position = 0
        df_trades = pd.DataFrame(data=None, index=df_prices.index, columns=[symbol, 'Cash'])

        for j in range(len(dates)):
            date = dates[j]

            s = int(discretized[date] * 10 + position)

            a = self.learner.querysetstate(s)

            if a == 1 and position != 1:
                if position == -1:
                    df_trades.ix[date, 0] = 2000
                    df_trades.ix[date, 1] = - 2000 * df_prices.ix[date, 0]
                elif position == 0:
                    df_trades.ix[date, 0] = 1000
                    df_trades.ix[date, 1] = - 1000 * df_prices.ix[date, 0]

            elif a == -1 and position != -1:
                if position == 1:
                    df_trades.ix[date, 0] = - 2000
                    df_trades.ix[date, 1] = 2000 * df_prices.ix[date, 0]
                elif position == 0:
                    df_trades.ix[date, 0] = -1000
                    df_trades.ix[date, 1] = 1000 * df_prices.ix[date, 0]

            elif a == 0:
                if position == 1:
                    df_trades.ix[date, 0] = - 1000
                    df_trades.ix[date, 1] = 1000 * df_prices.ix[date, 0]
                elif position == -1:
                    df_trades.ix[date, 0] = 1000
                    df_trades.ix[date, 1] = - 1000 * df_prices.ix[date, 0]

            position = a

        df_trades.fillna(0, inplace=True)

        return df_trades
