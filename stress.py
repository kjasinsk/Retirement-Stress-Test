"""
Retirement Stress Test - simulation engine for the app (saving years + retirement years on the same paths).

Same rules as simulate.py, plus the years before retirement:
  * before retiring: add the yearly contribution (grows with inflation) at the start of each year, then grow
  * after retiring: withdraw (spending - Social Security, both grow with inflation) at the start of each year
  * portfolio rebalanced yearly to the stock/bond mix; 0.5% yearly fee; plan runs to age 95
  * optional "worse future": subtract a fixed amount from every year's stock return
  * optional flat tax rate on money taken out of the portfolio (withdrawal is grossed up: need / (1 - tax))
  * if Social Security covers all spending, nothing is withdrawn (the extra is NOT added to savings)
"""
import numpy as np
import pandas as pd

from simulate import synthetic_paths

FEE = 0.005
END_AGE = 95

# J.P. Morgan Asset Management, 2027 Long-Term Capital Market Assumptions (published Oct 19, 2026; estimates as of
# Sep 30, 2026; 10-15 year compound annual returns in USD). Update these each year when the new edition comes out.
LTCMA = {"edition": "J.P. Morgan 2027 Long-Term Capital Market Assumptions",
         "stocks": 0.075,     # U.S. large cap equities
         "bonds": 0.046}      # U.S. intermediate Treasuries


def _geo_mean(r):
    return float(np.prod(1 + r) ** (1 / len(r)) - 1)


def _shift_to(r, target):
    """Constant shift d so that the compound (geometric) average of r + d equals target (bisection)."""
    lo, hi = -0.5, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _geo_mean(r + mid) < target else (lo, mid)
    return (lo + hi) / 2


def forecast_history(history, stock_return=LTCMA["stocks"], bond_return=LTCMA["bonds"]):
    """Copy of the 1928-2025 history with every year's stock and bond return moved by one constant amount, so the
    long-run compound averages match a forecast (e.g. J.P. Morgan's LTCMA). Crashes, streaks, volatility and the
    link between stocks, bonds and inflation stay exactly as they happened; inflation is unchanged."""
    out = history.copy()
    out["stocks"] = out["stocks"] + _shift_to(out["stocks"].to_numpy(), stock_return)
    out["t_bonds"] = out["t_bonds"] + _shift_to(out["t_bonds"].to_numpy(), bond_return)
    return out


def make_paths(history, years, n_paths=25_000, seed=42, stock_haircut=0.0):
    stocks, bonds, infl = synthetic_paths(history, n_paths=n_paths, years=years, rng=np.random.default_rng(seed))
    return stocks - stock_haircut, bonds, infl


def run_plan(paths, current_age, retire_age, savings, contribution, spending, social_security, stock_share,
             tax_rate=0.0):
    """Year-by-year balances in today's dollars (paths x years+1), and the age money ran out (NaN = never)."""
    stocks, bonds, infl = paths
    n, years = stocks.shape
    bal = np.full(n, float(savings))
    add = np.full(n, float(contribution))
    need = np.full(n, max(float(spending - social_security), 0.0) / (1 - tax_rate))  # pre-tax withdrawal
    deflator = np.ones(n)
    ran_out_age = np.full(n, np.nan)
    real = [bal.copy()]
    for t in range(years):
        age = current_age + t
        if age < retire_age:
            bal = bal + add
        else:
            bal = bal - need
            broke = (bal <= 0) & np.isnan(ran_out_age)
            ran_out_age[broke] = age
            bal = np.maximum(bal, 0)
        bal = bal * (1 + stock_share * stocks[:, t] + (1 - stock_share) * bonds[:, t] - FEE)
        add *= 1 + infl[:, t]
        need *= 1 + infl[:, t]
        deflator *= 1 + infl[:, t]
        real.append(bal / deflator)
    return np.array(real), ran_out_age


def success(paths, **plan):
    _, out = run_plan(paths, **plan)
    return float(np.mean(np.isnan(out)))


def max_safe_spending(paths, target=0.85, **plan):
    """Binary search: highest yearly spending (today's dollars) with success >= target."""
    plan = dict(plan)
    lo, hi = 0.0, max(plan["savings"], 1) * 0.25 + plan["social_security"] + plan.get("contribution", 0) * 10
    for _ in range(30):
        mid = (lo + hi) / 2
        plan["spending"] = mid
        lo, hi = (mid, hi) if success(paths, **plan) >= target else (lo, mid)
    return lo


def input_problems(current_age, retire_age, savings, contribution, spending, social_security, stock_share,
                   tax_rate=0.0):
    """Check what the user typed. Returns (errors that stop the run, warnings to show)."""
    errors, warnings = [], []
    values = dict(current_age=current_age, retire_age=retire_age, savings=savings, contribution=contribution,
                  spending=spending, social_security=social_security)
    missing = [k.replace("_", " ") for k, v in values.items() if v is None]
    if missing:
        return [f"Fill in: {', '.join(missing)}."], []
    if not 18 <= current_age < END_AGE - 1:
        errors.append(f"Current age must be between 18 and {END_AGE - 2}.")
    if retire_age < current_age:
        errors.append("Retirement age can't be before your current age.")
    if retire_age >= END_AGE:
        errors.append(f"Retirement age must be before {END_AGE}.")
    if min(savings, contribution, spending, social_security) < 0:
        errors.append("Dollar amounts can't be negative.")
    if not 0 <= tax_rate < 0.6:
        errors.append("Tax rate must be between 0% and 60%.")
    if errors:
        return errors, warnings
    if savings == 0 and contribution == 0:
        warnings.append("No savings and no yearly saving: the plan depends only on Social Security.")
    if spending <= social_security:
        warnings.append("Social Security covers all of the spending, so nothing comes out of savings.")
    if spending > 0 and savings > 0 and (spending - social_security) / savings > 0.10 and retire_age == current_age:
        warnings.append("Taking more than 10% of savings in the first year is very high; check the numbers.")
    if savings > 50_000_000 or spending > 2_000_000:
        warnings.append("These amounts are unusually large; check for an extra zero.")
    if retire_age - current_age > 40:
        warnings.append("Retirement is more than 40 years away, so the results are very uncertain.")
    if stock_share >= 0.95:
        warnings.append("Almost all stocks: big swings, especially early in retirement.")
    if retire_age > current_age and contribution == 0:
        warnings.append("Saving per year is $0 until retirement; was that on purpose?")
    return errors, warnings
