"""
Retirement Stress Test - shared results helpers used by the app AND by the AI tests (eval_ai.py).

  summarize_run(paths, plan, target)  run the model once and collect every number the app shows
  facts_for(result)                   the numbers, already formatted by Python, that the LLM may use
  report_html(result, summary, ...)   a one-page client report (download button in the app)
"""
import html
from datetime import date

import numpy as np

from stress import END_AGE, FEE, max_safe_spending, run_plan, ss_factor, success


def summarize_run(paths, plan, target=0.85, haircut=0.0, scenario=None, years=(1928, 2025)):
    real, out = run_plan(paths, **plan)
    safe = max_safe_spending(paths, target=target, **plan)
    ran_out = out[~np.isnan(out)]
    # saving years: contributions grow with inflation, so in today's dollars the total is simply amount x years
    contrib_years = max(int(plan["retire_age"]) - int(plan["current_age"]), 0)
    return dict(plan=plan, paths=paths, real=real, out=out, success=float(np.mean(np.isnan(out))), safe=safe,
                target=target, haircut=haircut, scenario=scenario, years=tuple(years),
                contrib_years=contrib_years, contrib_total=float(plan["contribution"]) * contrib_years,
                retire_p50=float(np.percentile(real[contrib_years], 50)),
                p10=float(np.percentile(real[-1], 10)), p50=float(np.percentile(real[-1], 50)),
                p90=float(np.percentile(real[-1], 90)),
                runout_age=int(np.percentile(ran_out, 10)) if len(ran_out) else None)


def facts_for(r):
    """Every number the LLM may use, formatted by Python (the LLM copies them, it never calculates)."""
    plan = r["plan"]
    facts = {
        "Current age": str(plan["current_age"]),
        "Retirement age": str(plan["retire_age"]),
        "Planned yearly spending (today's dollars)": f"${plan['spending']:,.0f}",
        "Chance the money lasts to age 95": f"{r['success']:.1%}",
        f"Highest safe yearly spending for a {r['target']:.0%} chance": f"${r['safe']:,.0f}",
        "Money left at 95 in a bad future (10th percentile, today's dollars)": f"${r['p10']:,.0f}",
        "Money left at 95 in a typical future (median, today's dollars)": f"${r['p50']:,.0f}",
    }
    if r["runout_age"] is not None and r["success"] < 0.999:
        facts["Among the futures where the money runs out, 1 in 10 run out by age"] = str(r["runout_age"])
    if r.get("contrib_years"):
        facts["Total the client puts in before retiring (today's dollars)"] = f"${r['contrib_total']:,.0f}"
        facts["Typical savings at retirement (median, today's dollars)"] = f"${r['retire_p50']:,.0f}"
    if tuple(r.get("years", (1928, 2025))) != (1928, 2025):
        facts["Historical years the market futures were drawn from"] = f"{r['years'][0]}-{r['years'][1]}"
    if plan.get("ss_age") is not None:
        facts["Social Security starts at age"] = str(plan["ss_age"])
        facts["Social Security per year from that age (today's dollars)"] = \
            f"${plan['social_security'] * ss_factor(plan['ss_age']):,.0f}"
    if plan.get("flex_cut"):
        facts["Flexible spending: spending cut after a losing year"] = f"{plan['flex_cut']:.0%}"
    if r.get("scenario"):
        facts["Market returns assumed"] = r["scenario"]
    if plan.get("tax_rate"):
        facts["Tax rate assumed on withdrawals"] = f"{plan['tax_rate']:.0%}"
    if r["haircut"]:
        facts["Worse-future test: stock returns cut by (percentage points per year)"] = f"{r['haircut']:.1f}"
    return facts


def _money(x):
    return f"${x:,.0f}"


def report_html(r, summary=None, age_rows=None, plans=None):
    """Self-contained HTML report a client can open or print (no internet needed)."""
    plan = r["plan"]
    y0, y1 = r.get("years", (1928, 2025))
    rows_in = [("Current age", plan["current_age"]), ("Retirement age", plan["retire_age"]),
               ("Savings today", _money(plan["savings"])),
               ("Saving per year until retirement", _money(plan["contribution"])),
               ("Yearly spending in retirement (today's $)", _money(plan["spending"])),
               ("Social Security per year" + (" at 67" if plan.get("ss_age") is not None else ""),
                _money(plan["social_security"])),
               ("Social Security starts at", f"{plan['ss_age']} ({_money(plan['social_security'] * ss_factor(plan['ss_age']))}/yr)"
                if plan.get("ss_age") is not None else f"retirement ({plan['retire_age']})"),
               ("Flexible spending", f"spend {plan['flex_cut']:.0%} less after a losing year" if plan.get("flex_cut")
                else "off"),
               ("Stocks / bonds", f"{plan['stock_share']:.0%} / {1 - plan['stock_share']:.0%}"),
               ("Tax rate on withdrawals", f"{plan.get('tax_rate', 0):.0%}"),
               ("Target chance of success", f"{r['target']:.0%}"),
               ("Historical years used", f"{y0}-{y1}"),
               ("Market returns", r.get("scenario") or f"Historical {y0}-{y1} returns"),
               ("Worse-future test", f"stock returns cut {r['haircut']:.1f} points/yr" if r["haircut"] else "off")]
    rows_out = [("Total put in before retiring (today's $)",
                 f"{_money(r.get('contrib_total', 0))} over {r.get('contrib_years', 0)} years"),
                ("Typical savings at retirement (median)", _money(r.get("retire_p50", plan["savings"]))),
                ("Chance the money lasts to age 95", f"{r['success']:.1%}"),
                (f"Highest safe yearly spending ({r['target']:.0%} target)", _money(r["safe"])),
                ("Money left at 95: bad future (10th pct.)", _money(r["p10"])),
                ("Money left at 95: typical future (median)", _money(r["p50"])),
                ("Money left at 95: good future (90th pct.)", _money(r["p90"]))]
    ages = plan["current_age"] + np.arange(r["real"].shape[0])
    pct = np.percentile(r["real"], [10, 50, 90], axis=1)
    path_rows = [(int(a), _money(pct[0, i]), _money(pct[1, i]), _money(pct[2, i]))
                 for i, a in enumerate(ages) if (a % 5 == 0 or i == 0 or a == END_AGE)]

    def table(rows, head=None):
        h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head) if head else ""
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
        return f"<table>{'<tr>' + h + '</tr>' if h else ''}{body}</table>"

    summary_html = ""
    if summary and summary.get("text"):
        badge = ("Number check: PASS, every number came from the model" if summary.get("bad") == []
                 else f"Number check: FAIL ({summary['bad']})" if summary.get("bad") else "")
        summary_html = f"<h2>Summary</h2><p>{html.escape(summary['text'])}</p><p class='badge'>{html.escape(badge)}</p>"
    plans_html = ""
    if plans is not None and len(plans):
        plans_html = "<h2>Plans compared</h2>" + table(plans.values.tolist(), list(plans.columns))
    ages_html = ""
    if age_rows:
        cols = list(age_rows[0].keys())
        ages_html = "<h2>Retire earlier or later?</h2>" + table([[row[c] for c in cols] for row in age_rows], cols)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Retirement Stress Test report</title>
<style>
 body {{ font-family: "Segoe UI", Arial, sans-serif; color: #23303D; max-width: 760px; margin: 32px auto; padding: 0 16px; }}
 h1 {{ font-family: Georgia, serif; margin-bottom: 0; }} h2 {{ font-family: Georgia, serif; color: #8C6A47; margin-top: 28px; }}
 table {{ border-collapse: collapse; width: 100%; }} td, th {{ border-bottom: 1px solid #D5DCE2; padding: 6px 8px; text-align: left; }}
 th {{ background: #0F2236; color: white; }} .big {{ display: flex; gap: 12px; }}
 .tile {{ flex: 1; background: #F7F5F1; border-radius: 8px; padding: 12px 14px; }} .tile b {{ font-size: 26px; display: block; }}
 .badge {{ color: #2E6B4F; font-weight: 600; }} .note {{ color: #5A6B7B; font-size: 13px; }}
</style></head><body>
<h1>Retirement Stress Test</h1><p class="note">Report created {date.today():%B %d, %Y}</p>
<div class="big">
 <div class="tile">Chance the money lasts to {END_AGE}<b>{r['success']:.1%}</b></div>
 <div class="tile">Highest safe yearly spending<b>{_money(r['safe'])}</b></div>
 <div class="tile">Money left at {END_AGE}, bad future<b>{_money(r['p10'])}</b></div>
</div>
{summary_html}
<h2>Your inputs</h2>{table(rows_in)}
<h2>Results</h2>{table(rows_out)}
{ages_html}
{plans_html}
<h2>Balance over time (today's dollars)</h2>{table(path_rows, ["Age", "Bad future (10th pct.)", "Typical (median)", "Good future (90th pct.)"])}
<h2>How this was calculated</h2>
<p class="note">25,000 possible market futures were built from random 5-year stretches of real {y0}-{y1} stock, bond and
inflation data (Damodaran, NYU Stern; BLS CPI-U). Each year: contributions before retiring; after retiring, spending
minus Social Security is withdrawn (grossed up for the tax rate); both grow with inflation; {FEE:.1%} yearly fee;
rebalanced to the stock/bond mix; plan runs to age {END_AGE}. The highest safe spending is found by binary search.
{"The summary above was written by an AI model from numbers Python calculated, and every number in it was checked against the model. " if summary_html else ""}Taxes are a single flat rate. This is an educational illustration, not financial advice.</p>
</body></html>"""
