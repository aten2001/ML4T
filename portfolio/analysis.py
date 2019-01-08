import datetime as dt

import pandas as pd

from util import get_data, plot_data


def compute_portfolio_stats(port_val, rfr, sf):
    cr = (port_val[-1] / port_val[0]) - 1

    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]

    adr = daily_returns.mean()

    sddr = daily_returns.std()

    sr = sf ** 0.5 * (daily_returns - rfr).mean() / sddr

    return cr, adr, sddr, sr


def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                     syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
                     allocs=[0.1, 0.2, 0.3, 0.4],
                     sv=1000000, rfr=0.0, sf=252.0,
                     gen_plot=False):
    if sum(allocs) != 1:
        raise Exception('Allocations to the equities should sum to 1')

    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices = prices_all[syms]
    prices_SPY = prices_all['SPY']

    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis=1)

    cr, adr, sddr, sr = compute_portfolio_stats(port_val=port_val, rfr=rfr, sf=sf)

    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp / df_temp.iloc[0]
        plot_data(df_temp)

    ev = port_val[-1]

    return cr, adr, sddr, sr, ev
