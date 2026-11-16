"""
Retirement Stress Test - synthetic market paths + Monte Carlo prototype (Problem 5, parts 3 and 4)

Generates synthetic future market paths by resampling real historical years (block bootstrap),
then runs a client's retirement through each path and finds the highest safe spending.

Run (after data_pull.py has made market_history.csv):  python simulate.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)  # fixed seed so results can be reproduced


def synthetic_paths(history, n_paths=10_000, years=30, block=5, rng=RNG):
    """
    Build synthetic future paths by stitching together random 5-year blocks of real history.

    Why this distribution: instead of assuming returns are normal (bell-shaped), we reuse real
    years, so crashes like 1931, 1974 and 2008 show up as often as they really happened, and
    blocks keep the link between stocks, bonds and inflation in the same year (e.g. the 1970s).
    """
    data = history[["stocks", "t_bonds", "inflation"]].to_numpy()
    n_hist = len(data)
    n_blocks = int(np.ceil(years / block))
    starts = rng.integers(0, n_hist - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :years]
    paths = data[idx]  # shape: paths x years x 3
    return paths[:, :, 0], paths[:, :, 1], paths[:, :, 2]


def run_retirement(savings, spending, stock_share, stocks, bonds, inflation,
                   social_security=0.0, fee=0.005):
    """
    Simulate each path year by year. Spending and Social Security grow with inflation.
    Returns the ending balance per path (0 = ran out) and the year money ran out (or None).
    """
    n_paths, years = stocks.shape
    balance = np.full(n_paths, float(savings))
    need = np.full(n_paths, float(spending - social_security))
    ran_out_year = np.full(n_paths, np.nan)
    for t in range(years):
        balance -= need  # withdraw at the start of the year
        broke = (balance <= 0) & np.isnan(ran_out_year)
        ran_out_year[broke] = t + 1
        balance = np.maximum(balance, 0)
        growth = stock_share * stocks[:, t] + (1 - stock_share) * bonds[:, t] - fee
        balance *= 1 + growth
        need *= 1 + inflation[:, t]
    return balance, ran_out_year


def success_rate(*args, **kwargs):
    balance, ran_out = run_retirement(*args, **kwargs)
    return np.mean(np.isnan(ran_out))


def max_safe_spending(savings, stock_share, stocks, bonds, inflation,
                      target=0.85, social_security=0.0):
    """Optimization step: binary search for the highest spending with success >= target."""
    lo, hi = 0.0, savings * 0.15
    for _ in range(40):
        mid = (lo + hi) / 2
        ok = success_rate(savings, mid, stock_share, stocks, bonds, inflation,
                          social_security=social_security) >= target
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


if __name__ == "__main__":
    history = pd.read_csv("market_history.csv")
    stocks, bonds, infl = synthetic_paths(history)

    print("First 10 synthetic years of path #1 (decimals):")
    print(pd.DataFrame({"year": range(1, 11), "stocks": stocks[0, :10],
                        "t_bonds": bonds[0, :10], "inflation": infl[0, :10]}).round(4).to_string(index=False))

    # Example client: $1.2M saved, wants $60k/yr, 60% stocks, $24k/yr Social Security, 30-year horizon
    client = dict(savings=1_200_000, stock_share=0.60, social_security=24_000)
    p = success_rate(client["savings"], 60_000, client["stock_share"], stocks, bonds, infl,
                     social_security=client["social_security"])
    best = max_safe_spending(client["savings"], client["stock_share"], stocks, bonds, infl,
                             social_security=client["social_security"])
    print(f"\nChance $60,000/yr lasts 30 years: {p:.1%}")
    print(f"Highest spending with an 85% success rate: ${best:,.0f}/yr")
