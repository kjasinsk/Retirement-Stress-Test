"""
Retirement Stress Test - verification checks (Problem 4, item 6)

Runs the checks from our evaluation plan against simulate.py and saves them to verify_results.csv:
  1. Hand calculation: one simple path computed by hand must match the model exactly
  2. Sanity bounds: $0 spending always succeeds; more spending never helps; more savings never hurts
  3. Known answer: Trinity-Study setup (4% first-year withdrawal, inflation-adjusted, 30 years)
  4. Stability: success rate across different random seeds
  5. Historical backtest: real retirements that started in 1929, 1966 and 2000
  6. The app's engine (stress.py): matches simulate.py; tax, worse future, saving and input checks behave sensibly
  7. J.P. Morgan LTCMA forecast option: hits the forecast averages exactly and keeps the real swings

Run (after data_pull.py has made market_history.csv):  python verify.py
"""
import numpy as np
import pandas as pd

from simulate import synthetic_paths, run_retirement, success_rate, max_safe_spending

history = pd.read_csv("market_history.csv")
results = []


def check(name, passed, detail):
    results.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")


# 1. Hand calculation: 1 path, 2 years, $100k, $10k spending, 100% stocks, +10% then -20%, 0% inflation, no fee
s, b, i = np.array([[0.10, -0.20]]), np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])
bal, _ = run_retirement(100_000, 10_000, 1.0, s, b, i, fee=0.0)
by_hand = ((100_000 - 10_000) * 1.10 - 10_000) * 0.80          # = 71,200
check("Hand calculation (2-year path)", abs(bal[0] - by_hand) < 0.01,
      f"model ${bal[0]:,.2f} vs. by hand ${by_hand:,.2f}")

# Shared paths for the remaining checks (fixed seed 42, 10,000 paths x 30 years)
stocks, bonds, infl = synthetic_paths(history, rng=np.random.default_rng(42))
client = dict(savings=1_200_000, stock_share=0.60, social_security=24_000)

# 2. Sanity bounds
p0 = success_rate(client["savings"], 0, client["stock_share"], stocks, bonds, infl)
check("$0 spending always succeeds", p0 == 1.0, f"success {p0:.1%}")

levels = [40_000, 60_000, 80_000, 100_000, 120_000]
by_spend = [success_rate(client["savings"], x, 0.6, stocks, bonds, infl, social_security=24_000) for x in levels]
check("More spending never raises success", all(a >= b for a, b in zip(by_spend, by_spend[1:])),
      ", ".join(f"${x // 1000}k: {p:.1%}" for x, p in zip(levels, by_spend)))

savings_levels = [800_000, 1_000_000, 1_200_000, 1_400_000]
by_sav = [success_rate(x, 75_000, 0.6, stocks, bonds, infl, social_security=24_000) for x in savings_levels]
check("More savings never lowers success", all(a <= b for a, b in zip(by_sav, by_sav[1:])),
      ", ".join(f"${x / 1e6:.1f}M: {p:.1%}" for x, p in zip(savings_levels, by_sav)))

# 3. Known answer: Trinity Study (Cooley, Hubbard & Walz, 1998) - 4% withdrawal, 30 years, ~95% success
for share in (0.50, 0.75):
    p = success_rate(1_000_000, 40_000, share, stocks, bonds, infl, fee=0.0)
    check(f"Trinity Study, {share:.0%} stocks, 4% withdrawal", abs(p - 0.95) <= 0.05,
          f"our model {p:.1%} vs. published ~95% (pass if within 5 points)")

# 4. Stability across random seeds
rates = []
for seed in (1, 7, 42, 123, 2026):
    st, bo, inf = synthetic_paths(history, rng=np.random.default_rng(seed))
    rates.append(success_rate(1_200_000, 75_000, 0.6, st, bo, inf, social_security=24_000))
spread = (max(rates) - min(rates)) * 100
check("Stable across 5 random seeds (10,000 paths)", spread < 1.0,
      f"success at $75k ranges {min(rates):.1%} to {max(rates):.1%} ({spread:.2f} points)")
# Fix found by the check above: near the 85% decision point, 10,000 paths is borderline -> use 25,000
rates = []
for seed in (1, 7, 42, 123, 2026):
    st, bo, inf = synthetic_paths(history, n_paths=25_000, rng=np.random.default_rng(seed))
    rates.append(success_rate(1_200_000, 75_000, 0.6, st, bo, inf, social_security=24_000))
spread = (max(rates) - min(rates)) * 100
check("Stable across 5 random seeds (25,000 paths)", spread < 1.0,
      f"success at $75k ranges {min(rates):.1%} to {max(rates):.1%} ({spread:.2f} points)")

# 5. Historical backtest: real sequence of years, 5% first-year withdrawal, 60% stocks, 0.5% fee
for start in (1929, 1966, 2000):
    h = history[history["year"] >= start].head(30)
    st, bo, inf = (h[c].to_numpy()[None, :] for c in ("stocks", "t_bonds", "inflation"))
    bal, out = run_retirement(1_000_000, 50_000, 0.60, st, bo, inf)
    real_end = bal[0] / np.prod(1 + inf[0])                       # ending balance in start-year dollars
    years = len(h)
    if np.isnan(out[0]):
        detail = f"{years} years ({start}-{start + years - 1}): money lasted, ending at ${real_end:,.0f} in {start} dollars"
        high_risk = real_end < 500_000                           # lost more than half of its real value
    else:
        detail = f"money ran out in year {int(out[0])} ({start + int(out[0]) - 1})"
        high_risk = True
    check(f"Backtest {start} start, 5% withdrawal flagged high risk", high_risk, detail)

# 6. The app's engine (stress.py: saving years, tax, worse future) agrees with simulate.py and behaves sensibly
from stress import input_problems, make_paths, run_plan, success as app_success, max_safe_spending as app_safe
app_paths = make_paths(history, 30, n_paths=10_000, seed=42)                  # same paths as above
base = dict(current_age=65, retire_age=65, savings=1_200_000, contribution=0, spending=60_000,
            social_security=24_000, stock_share=0.6)
same_p = app_success(app_paths, **base)
same_safe = app_safe(app_paths, **base)
ref_safe = max_safe_spending(1_200_000, 0.6, stocks, bonds, infl, social_security=24_000)
check("App engine matches simulate.py (no saving years)",
      abs(same_p - success_rate(1_200_000, 60_000, 0.6, stocks, bonds, infl, social_security=24_000)) < 1e-12
      and abs(same_safe - ref_safe) < 1,
      f"success {same_p:.2%} both; highest safe ${same_safe:,.0f} vs. ${ref_safe:,.0f}")

saver_paths = make_paths(history, 37, n_paths=10_000, seed=42)
saver = dict(current_age=58, retire_age=63, savings=900_000, contribution=20_000, spending=70_000,
             social_security=24_000, stock_share=0.6)
by_tax = [app_success(saver_paths, **saver, tax_rate=t) for t in (0, 0.1, 0.2, 0.3)]
check("Higher tax never raises success", all(a >= b for a, b in zip(by_tax, by_tax[1:])),
      ", ".join(f"{t:.0%} tax: {p:.1%}" for t, p in zip((0, 0.1, 0.2, 0.3), by_tax)))
by_cut = [app_success(make_paths(history, 37, n_paths=10_000, seed=42, stock_haircut=c / 100), **saver)
          for c in (0, 1, 2, 3)]
check("Worse future (lower stock returns) never raises success", all(a >= b for a, b in zip(by_cut, by_cut[1:])),
      ", ".join(f"-{c} pts: {p:.1%}" for c, p in zip((0, 1, 2, 3), by_cut)))
by_add = [app_success(saver_paths, **dict(saver, contribution=c)) for c in (0, 10_000, 20_000, 40_000)]
check("Saving more before retiring never lowers success", all(a <= b for a, b in zip(by_add, by_add[1:])),
      ", ".join(f"${c // 1000}k/yr: {p:.1%}" for c, p in zip((0, 10_000, 20_000, 40_000), by_add)))
rich_ss, _ = run_plan(saver_paths, **dict(saver, spending=20_000))       # Social Security > spending
no_spend, _ = run_plan(saver_paths, **dict(saver, spending=0))
check("Social Security above spending: nothing withdrawn, nothing extra added",
      np.allclose(rich_ss, no_spend), "balances identical to the $0-spending plan (bug found and fixed in stress.py)")
bad_inputs = [dict(saver, retire_age=50), dict(saver, current_age=97), dict(saver, savings=-5),
              dict(saver, spending=None), dict(saver, tax_rate=0.9)]
caught = sum(bool(input_problems(**b)[0]) for b in bad_inputs)
check("Input checks stop impossible inputs", caught == len(bad_inputs) and not input_problems(**saver)[0],
      f"{caught} of {len(bad_inputs)} impossible inputs stopped; the normal plan runs")

# 7. J.P. Morgan LTCMA forecast option: averages hit the forecast exactly, swings stay the same
from stress import LTCMA, forecast_history, _geo_mean
fh = forecast_history(history)
geo_s, geo_b = _geo_mean(fh["stocks"].to_numpy()), _geo_mean(fh["t_bonds"].to_numpy())
check("LTCMA forecast: compound averages match J.P. Morgan's numbers",
      abs(geo_s - LTCMA["stocks"]) < 1e-6 and abs(geo_b - LTCMA["bonds"]) < 1e-6,
      f"stocks {geo_s:.2%} (target {LTCMA['stocks']:.1%}), bonds {geo_b:.2%} (target {LTCMA['bonds']:.1%}); "
      f"history was {_geo_mean(history['stocks'].to_numpy()):.2%} and {_geo_mean(history['t_bonds'].to_numpy()):.2%}")
check("LTCMA forecast keeps the real ups and downs",
      abs(fh["stocks"].std() - history["stocks"].std()) < 1e-12 and fh["inflation"].equals(history["inflation"])
      and int(fh.loc[fh["stocks"].idxmin(), "year"]) == 1931,
      f"stock volatility {fh['stocks'].std():.1%} both; worst year still 1931; inflation unchanged")
p_hist = app_success(saver_paths, **saver)
p_jpm = app_success(make_paths(fh, 37, n_paths=10_000, seed=42), **saver)
check("Lower forecast returns lower the chance of success", p_jpm < p_hist,
      f"same client: {p_hist:.1%} on history vs. {p_jpm:.1%} on the J.P. Morgan forecast")

# Record the example client's headline numbers too
p60 = success_rate(1_200_000, 60_000, 0.6, stocks, bonds, infl, social_security=24_000)
best = max_safe_spending(1_200_000, 0.6, stocks, bonds, infl, social_security=24_000)
print(f"\nExample client: $60,000/yr succeeds in {p60:.1%} of 10,000 futures; max safe spending at 85% = ${best:,.0f}")

pd.DataFrame(results).to_csv("verify_results.csv", index=False)
print(f"\n{sum(r['result'] == 'PASS' for r in results)} of {len(results)} checks passed -> verify_results.csv")
