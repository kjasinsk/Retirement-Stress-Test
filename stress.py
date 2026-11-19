"""
Retirement Stress Test - simulation engine for the app (saving years + retirement years on the same paths).

Same rules as simulate.py, plus the years before retirement:
  * before retiring: add the yearly contribution (grows with inflation) at the start of each year, then grow
  * after retiring: withdraw (spending - Social Security, both grow with inflation) at the start of each year
  * portfolio rebalanced yearly to the stock/bond mix; 0.5% yearly fee; plan runs to age 95
  * optional "worse future": subtract a fixed amount from every year's stock return
  * optional flat tax rate on money taken out of the portfolio (withdrawal is grossed up: need / (1 - tax))
  * if Social Security covers all spending, nothing is withdrawn (the extra is NOT added to savings)
  * optional Social Security claiming age (62-70): the benefit is entered at full retirement age (67) and moved up
    or down with the SSA rules; checks received before retiring are added to savings
  * optional flexible spending: after a year when the portfolio lost money, spend flex_cut % less the next year
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


FULL_RETIREMENT_AGE = 67


def ss_factor(claim_age):
    """Share of the full-retirement-age (67) benefit received when claiming at claim_age (SSA rules):
    -5/9% a month for the first 36 months early, -5/12% a month beyond that (62 -> 70%); +8% a year after 67
    up to 70 (70 -> 124%)."""
    if claim_age < FULL_RETIREMENT_AGE:
        months = (FULL_RETIREMENT_AGE - claim_age) * 12
        return 1 - min(months, 36) * 5 / 900 - max(months - 36, 0) * 5 / 1200
    return 1 + 0.08 * (min(claim_age, 70) - FULL_RETIREMENT_AGE)


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
             tax_rate=0.0, ss_age=None, flex_cut=0.0):
    """Year-by-year balances in today's dollars (paths x years+1), and the age money ran out (NaN = never).
    ss_age=None: Social Security starts at retirement at the amount entered (the original rule).
    ss_age=62..70: social_security is the benefit at 67, adjusted with ss_factor, starting at ss_age."""
    stocks, bonds, infl = paths
    n, years = stocks.shape
    ss_start = retire_age if ss_age is None else ss_age
    bal = np.full(n, float(savings))
    add = np.full(n, float(contribution))
    spend = np.full(n, float(spending))
    ss = np.full(n, float(social_security) * (1.0 if ss_age is None else ss_factor(ss_age)))
    down_last_year = np.zeros(n, dtype=bool)
    deflator = np.ones(n)
    ran_out_age = np.full(n, np.nan)
    real = [bal.copy()]
    for t in range(years):
        age = current_age + t
        if age < retire_age:
            bal = bal + add + (ss if age >= ss_start else 0.0)   # early Social Security checks are saved
        else:
            this_spend = np.where(down_last_year, spend * (1 - flex_cut), spend) if flex_cut else spend
            income = ss if age >= ss_start else 0.0
            need = np.maximum(this_spend - income, 0.0) / (1 - tax_rate)   # pre-tax withdrawal
            bal = bal - need
            broke = (bal <= 0) & np.isnan(ran_out_age)
            ran_out_age[broke] = age
            bal = np.maximum(bal, 0)
        growth = stock_share * stocks[:, t] + (1 - stock_share) * bonds[:, t] - FEE
        down_last_year = growth < 0
        bal = bal * (1 + growth)
        add = add * (1 + infl[:, t])
        spend = spend * (1 + infl[:, t])
        ss = ss * (1 + infl[:, t])
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
                   tax_rate=0.0, ss_age=None, flex_cut=0.0):
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
    if ss_age is not None and not 62 <= ss_age <= 70:
        errors.append("Social Security can start between ages 62 and 70.")
    if not 0 <= flex_cut < 1:
        errors.append("Flexible spending cut must be between 0% and 99%.")
    if errors:
        return errors, warnings
    if savings == 0 and contribution == 0:
        warnings.append("No savings and no yearly saving: the plan depends only on Social Security.")
    ss_paid = social_security * (1.0 if ss_age is None else ss_factor(ss_age))
    if spending <= ss_paid:
        warnings.append("Social Security covers all of the spending, so nothing comes out of savings.")
    if ss_age is not None and ss_age < retire_age and social_security > 0:
        warnings.append("Social Security starts before retiring: the app adds those checks to savings "
                        "(while working, the earnings test may hold some back).")
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
