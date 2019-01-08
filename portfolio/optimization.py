import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis import compute_portfolio_stats
from util import get_data, plot_data


def f(x, normed, rfr=0.0, sf=252.0):
    alloced = normed * x
    port_val = alloced.sum(axis=1)

    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]

    sddr = daily_returns.std()

    sr = sf ** 0.5 * (daily_returns - rfr).mean() / sddr

    return - sr


def optimize_portfolio(sd, ed, syms, gen_plot):
    """

    :param sd: dt.datetime
    :param ed: dt.datetime
    :param syms: list of strings
    :param gen_plot: boolean
    :return: tuple (allocs, cr, adr, sddr, sr)
    """
    n = len(syms)

    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices = prices_all[syms]
    prices_SPY = prices_all['SPY']

    prices = prices.ffill()
    prices = prices.bfill()

    normed = prices / prices.iloc[0]

    x0 = np.ones(n) / n

    bounds = [(0.0, 1.0) for i in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})

    allocs = minimize(f, x0, args=(normed, 0.0, 252.0), method='SLSQP', bounds=bounds,
                      constraints=constraints).x

    alloced = normed * allocs
    port_val = alloced.sum(axis=1)

    cr, adr, sddr, sr = compute_portfolio_stats(port_val=port_val)

    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp / df_temp.iloc[0]
        plot_data(df_temp, "Daily Portfolio Value and SPY")

    return allocs, cr, adr, sddr, sr
