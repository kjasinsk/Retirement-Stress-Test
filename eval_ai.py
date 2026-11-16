"""
Retirement Stress Test - AI accuracy tests (evaluation plan, Problem 5 part 7)

  Test 1  AI input parsing:   20 client descriptions with known answers.      Pass if at least 18 of 20 are fully right.
  Test 2  AI summary accuracy: 20 random client plans -> model -> LLM summary.  Pass if 100% of the numbers in the
                               summaries came from the model (no invented or changed numbers).

Uses the SAME functions as the app (llm_helpers.parse_inputs / write_summary / check_numbers, report.facts_for),
so this tests exactly what users get. Needs the Groq key (c:/MGT4170/MyKeys/.env).

Run:  python eval_ai.py      ->  ai_eval_parse.csv, ai_eval_summary.csv, ai_eval_results.txt
"""
import sys
import time

import numpy as np
import pandas as pd

import llm_helpers
from report import facts_for, summarize_run
from stress import END_AGE, make_paths

sys.stdout.reconfigure(encoding="utf-8")
FIELDS = ["current_age", "retirement_age", "savings", "annual_contribution", "annual_spending", "social_security",
          "stock_share"]
N = None

# (client text, expected values in FIELDS order). None = the client didn't say it, so the AI must return null.
# A tuple means any of those answers is accepted.
PARSE_CASES = [
    ("I'm 58 and have about $900k saved. I put away $20k a year, want to retire at 63 and spend around $55k a year. "
     "I expect about $2,000 a month from Social Security, and I'm invested roughly 60/40.",
     [58, 63, 900_000, 20_000, 55_000, 24_000, 0.60]),
    ("Age 45, $350,000 in my 401k, saving $1,500 a month. Plan to retire at 65, spend $70k/yr, Social Security about "
     "$30k a year, 80% stocks.", [45, 65, 350_000, 18_000, 70_000, 30_000, 0.80]),
    ("We're both 62 with $1.2 million saved. Retiring next year at 63, living on $85,000 a year. Social Security will "
     "be $3,100 per month combined. Portfolio is 50/50.", [62, 63, 1_200_000, N, 85_000, 37_200, 0.50]),
    ("I'm 35, have 120k saved, put away 10k per year, want to retire at 60 spending 60k annually, all in stocks.",
     [35, 60, 120_000, 10_000, 60_000, N, 1.0]),
    ("Sixty years old, two million dollars in savings, retiring at 62. Want to spend $9,000 a month. Expect "
     "$2,800/month from Social Security. 70 percent stocks, 30 percent bonds.",
     [60, 62, 2_000_000, N, 108_000, 33_600, 0.70]),
    ("I just turned 67 and retired this year. I have $650k and need $45k a year. Social Security pays me $28,800 a "
     "year. I keep 40% in stocks.", [67, 67, 650_000, N, 45_000, 28_800, 0.40]),
    ("I'm 50. Savings: $500k. Contribution: $25k/yr. Retirement age: 67. Spending goal: $60k. SS: $2,500/mo. "
     "Allocation: 65/35.", [50, 67, 500_000, 25_000, 60_000, 30_000, 0.65]),
    ("My husband and I are 55, we have about $1.5M and add $40,000 a year. We'd like to retire at 60 and spend $100k "
     "per year. We haven't figured out Social Security yet. Mostly stocks, like 75%.",
     [55, 60, 1_500_000, 40_000, 100_000, N, 0.75]),
    ("I am 40 years old with $80,000 saved. I want to retire at 65.", [40, 65, 80_000, N, N, N, N]),
    ("Retire at 70? I'm 64, have $400k, spend $3,500 a month, and get $1,900 monthly in Social Security. I'm "
     "conservative: 30% stocks.", [64, 70, 400_000, N, 42_000, 22_800, 0.30]),
    ("I'm 52 and I have $750k. I save $2,000 every month. I'd retire at 62 and spend $5,000 a month. My Social "
     "Security estimate is $26k a year. I'm 90/10 stocks/bonds.", [52, 62, 750_000, 24_000, 60_000, 26_000, 0.90]),
    ("Hi! 29 years old, 45k in savings, putting in 500 a month, hoping to retire at 55 on 50k a year, 100% stock "
     "index funds.", [29, 55, 45_000, 6_000, 50_000, N, 1.0]),
    ("I'm 61, retiring at 66 with $1,050,000. I'll keep saving $15k a year until then. I'd like $72,000 a year to "
     "live on and Social Security should be about $2,400 a month. Split 55% stocks, 45% bonds.",
     [61, 66, 1_050_000, 15_000, 72_000, 28_800, 0.55]),
    ("I'm 70, already retired, with $300k left. I spend $40k a year, and Social Security covers $32k of that. "
     "Everything is in bonds.", [70, (70, N), 300_000, (N, 0), 40_000, 32_000, 0.0]),
    ("Age 48. 401(k) $410k, IRA $90k. I contribute $23,000 a year. Retire at 62, spend $65k, Social Security $27k, "
     "70/30.", [48, 62, 500_000, 23_000, 65_000, 27_000, 0.70]),
    ("I'm fifty-seven. I have $2.3M. I'd like to stop working at 59 and spend $150,000 a year. I'll get $3,500 a "
     "month from Social Security. 60% equities.", [57, 59, 2_300_000, N, 150_000, 42_000, 0.60]),
    ("Currently 44, saved $260k, I add $12k annually. Retirement at 67, expected expenses $55k per year, Social "
     "Security $24k per year, portfolio 85% stocks 15% bonds.", [44, 67, 260_000, 12_000, 55_000, 24_000, 0.85]),
    ("I have $600,000 and want to spend $40k per year when I retire at 65. I'm 59. Stocks 50%.",
     [59, 65, 600_000, N, 40_000, N, 0.50]),
    ("I'm 63 and plan to retire in 2 years. Savings are $820k, spending would be $4,000/month, Social Security "
     "$2,200/month, and I'm in a 60/40 fund.", [63, 65, 820_000, N, 48_000, 26_400, 0.60]),
    ("Thirty-eight, with three hundred thousand dollars saved, contributing twenty thousand a year, I'd like to retire "
     "at sixty and spend eighty thousand a year. No Social Security assumed. 100 percent stocks.",
     [38, 60, 300_000, 20_000, 80_000, (0, N), 1.0]),
]


def same(got, want):
    """True if the AI's value matches the expected one (money within 1%, stock share within 1 point)."""
    if isinstance(want, tuple):
        return any(same(got, w) for w in want)
    if want is None:
        return got is None
    try:
        got = float(got)
    except (TypeError, ValueError):
        return False
    return abs(got - want) <= (0.011 if want <= 1 else max(0.01 * abs(want), 0.5))


def call(fn, *args):
    """LLM call with retries (Groq's free tier sometimes rate-limits)."""
    for attempt in range(4):
        try:
            return fn(*args)
        except Exception as e:                                       # noqa: BLE001
            err = e
            time.sleep(8 * (attempt + 1))
    raise err


# ---------------- Test 1: parsing ----------------
print(f"Test 1: AI input parsing on {len(PARSE_CASES)} client descriptions (model {llm_helpers.MODEL_NAME})\n")
parse_rows = []
for i, (text, want) in enumerate(PARSE_CASES, 1):
    try:
        got = call(llm_helpers.parse_inputs, text)
    except Exception as e:                                            # noqa: BLE001
        got = {"error": f"{type(e).__name__}: {e}"[:200]}
    wrong = [f for f, w in zip(FIELDS, want) if not same(got.get(f), w)]
    parse_rows.append({"case": i, "text": text, "expected": dict(zip(FIELDS, want)), "ai_output": got,
                       "wrong_fields": ", ".join(wrong), "result": "PASS" if not wrong else "FAIL"})
    print(f"  {i:2d}  {'PASS' if not wrong else 'FAIL'}  " + (f"wrong: {', '.join(wrong)}  got {got}" if wrong else ""))
    time.sleep(2)
n_parse = sum(r["result"] == "PASS" for r in parse_rows)
fields_right = sum(len(FIELDS) - (len(r["wrong_fields"].split(", ")) if r["wrong_fields"] else 0) for r in parse_rows)
pd.DataFrame(parse_rows).to_csv("ai_eval_parse.csv", index=False)

# ---------------- Test 2: summaries ----------------
print("\nTest 2: AI summaries for 20 random client plans (25,000 futures each)\n")
history = pd.read_csv("market_history.csv")
rng = np.random.default_rng(2026)
summary_rows = []
for i in range(1, 21):
    age = int(rng.integers(30, 71))
    plan = dict(current_age=age, retire_age=int(min(age + rng.integers(0, 16), 75)),
                savings=float(rng.integers(5, 300) * 10_000), contribution=float(rng.integers(0, 41) * 1_000),
                spending=float(rng.integers(30, 151) * 1_000), social_security=float(rng.integers(0, 41) * 1_000),
                stock_share=float(rng.integers(4, 21) * 0.05), tax_rate=float(rng.choice([0, 0, 0.10, 0.20])))
    target, haircut = float(rng.choice([0.80, 0.85, 0.90])), float(rng.choice([0, 0, 0, 1.0, 2.0]))
    r = summarize_run(make_paths(history, END_AGE - age, stock_haircut=haircut / 100), plan, target, haircut)
    facts = facts_for(r)
    try:
        text = call(llm_helpers.write_summary, facts)
        bad = llm_helpers.check_numbers(text, facts) if text.strip() else ["empty summary"]
    except Exception as e:                                            # noqa: BLE001
        text, bad = f"ERROR {type(e).__name__}: {e}"[:200], ["error"]
    key_numbers = [facts["Chance the money lasts to age 95"],
                   facts[f"Highest safe yearly spending for a {target:.0%} chance"]]
    mentions = all(k in text for k in key_numbers)
    n_nums = len(llm_helpers._numbers(text))
    summary_rows.append({"case": i, **plan, "target": target, "worse_future_pts": haircut,
                         "facts": facts, "summary": text, "numbers_in_summary": n_nums,
                         "numbers_not_from_model": bad, "uses_both_key_numbers": mentions,
                         "result": "PASS" if not bad else "FAIL"})
    print(f"  {i:2d}  {'PASS' if not bad else 'FAIL'}  {n_nums} numbers"
          + ("" if mentions else "  (left out a key number)") + (f"  invented: {bad}" if bad else ""))
    time.sleep(2)
n_sum = sum(r["result"] == "PASS" for r in summary_rows)
total_nums = sum(r["numbers_in_summary"] for r in summary_rows)
bad_nums = sum(len(r["numbers_not_from_model"]) for r in summary_rows)
n_key = sum(r["uses_both_key_numbers"] for r in summary_rows)
pd.DataFrame(summary_rows).to_csv("ai_eval_summary.csv", index=False)

# ---------------- Scorecard ----------------
lines = [
    f"AI accuracy tests, model {llm_helpers.MODEL_NAME} via Groq, run {pd.Timestamp.now():%Y-%m-%d %H:%M}",
    f"Test 1  Input parsing: {n_parse} of {len(PARSE_CASES)} descriptions fully right "
    f"({fields_right} of {len(PARSE_CASES) * len(FIELDS)} fields). Goal 18+: {'PASS' if n_parse >= 18 else 'FAIL'}",
    f"Test 2  Summary numbers: {n_sum} of 20 summaries had no invented numbers "
    f"({total_nums - bad_nums} of {total_nums} numbers matched the model). Goal 100%: {'PASS' if n_sum == 20 else 'FAIL'}",
    f"        Summaries that state both the chance of success and the highest safe spending: {n_key} of 20",
]
for r in parse_rows:
    if r["result"] == "FAIL":
        lines.append(f"  Parse case {r['case']} wrong on {r['wrong_fields']}: {r['ai_output']}")
for r in summary_rows:
    if r["result"] == "FAIL":
        lines.append(f"  Summary case {r['case']} numbers not from the model: {r['numbers_not_from_model']}")
open("ai_eval_results.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))
