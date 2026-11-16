"""
Retirement Stress Test - GenAI component (Problem 4, item 5)

Pipeline:  plain English  --LLM-->  model inputs (JSON)  --Python simulation-->  results  --LLM-->  client summary
Rule: the LLM never does math. Every number in the summary must come from the results table; the script
checks that and prints PASS/FAIL. Prompts and raw LLM outputs are saved to llm_demo_output.txt.

Uses LangChain + Groq like our class examples (key in c:/MGT4170/MyKeys/.env).
Run (after data_pull.py):  python llm_inputs.py
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from simulate import synthetic_paths, run_retirement, max_safe_spending

sys.stdout.reconfigure(encoding="utf-8")         # Windows console: LLM text can contain characters like '\u2011'
load_dotenv(Path("c:/MGT4170/MyKeys/.env"))
model = ChatGroq(model="openai/gpt-oss-20b", api_key=os.environ["GROQ_API_KEY"], temperature=0)

CLIENT_TEXT = ("I'm 58 and have about $900k saved for retirement. I want to retire at 63 and spend around "
               "$55k a year. I expect about $2,000 a month from Social Security, and I'm invested roughly "
               "60/40 stocks and bonds.")

# ---- 1. Plain English -> model inputs ------------------------------------------------------------
parse_prompt = PromptTemplate.from_template(
    """Extract retirement-planning inputs from the client's message. Return ONLY a JSON object:
{{"current_age": int, "retirement_age": int, "savings": number (US dollars),
  "annual_spending": number (US dollars per year), "social_security": number (US dollars per YEAR),
  "stock_share": number between 0 and 1}}
Convert monthly amounts to yearly and "k" to thousands. Use null for anything the client did not say.
Do not calculate anything else.

Client message: {text}""")
inputs = (parse_prompt | model | JsonOutputParser()).invoke({"text": CLIENT_TEXT})
print("LLM-parsed inputs:", inputs)

expected = {"current_age": 58, "retirement_age": 63, "savings": 900_000, "annual_spending": 55_000,
            "social_security": 24_000, "stock_share": 0.6}
parse_ok = all(inputs.get(k) is not None and abs(float(inputs[k]) - v) < 1e-6 for k, v in expected.items())
print("Parsed inputs match the known answers:", "PASS" if parse_ok else "FAIL")

# ---- 2. Python does the math ------------------------------------------------------------------
history = pd.read_csv("market_history.csv")
years = 95 - int(inputs["retirement_age"])                      # plan to age 95
stocks, bonds, infl = synthetic_paths(history, n_paths=25_000, years=years, rng=np.random.default_rng(42))
args = (inputs["savings"], inputs["annual_spending"], inputs["stock_share"], stocks, bonds, infl)
bal, out = run_retirement(*args, social_security=inputs["social_security"])
real_end = bal / np.prod(1 + infl, axis=1)                      # ending balance in today's dollars
results = {
    "years_planned": years,
    "success_rate_pct": round(float(np.mean(np.isnan(out))) * 100, 1),
    "max_safe_spending_at_85pct": int(round(max_safe_spending(inputs["savings"], inputs["stock_share"], stocks, bonds,
                                                              infl, social_security=inputs["social_security"]), -2)),
    "ending_balance_10th_pct": int(round(np.percentile(real_end, 10), -3)),
    "ending_balance_median": int(round(np.percentile(real_end, 50), -3)),
    "ending_balance_90th_pct": int(round(np.percentile(real_end, 90), -3)),
    "run_out_year_in_worst_10pct_of_failures": (int(np.nanpercentile(out, 10)) if np.isnan(out).mean() < 1 else None),
}
print("Simulation results:", results)

# ---- 3. Results -> client summary -------------------------------------------------------------
summary_prompt = PromptTemplate.from_template(
    """You are writing for a financial advisor's client. Write 3 short sentences in plain English that explain
these retirement simulation results. Use ONLY numbers that appear in the results below, written exactly as
given (dollars with commas, percentages with one decimal). Do not calculate, round or invent any number.
The client plans to retire at {retirement_age} and spend ${spending:,.0f} a year.

Results (25,000 simulated market futures built from 1928-2025 history):
{results}""")
summary = (summary_prompt | model | StrOutputParser()).invoke(
    {"retirement_age": inputs["retirement_age"], "spending": inputs["annual_spending"],
     "results": json.dumps(results, indent=2)})
print("\nLLM summary:\n" + summary)

# ---- 4. Check every number in the summary against the model ---------------------------------
allowed = {float(v) for v in list(results.values()) + list(inputs.values()) if isinstance(v, (int, float))}
allowed |= {95.0, 25000.0, 1928.0, 2025.0, 10.0, 50.0, 90.0, 85.0}   # plan age, path count, data years, percentile labels
found = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", summary)]
bad = [x for x in found if x not in allowed]
print("\nNumbers in summary:", found)
print("Every number comes from the model:", "PASS" if not bad else f"FAIL (not from the model: {bad})")

with open("llm_demo_output.txt", "w", encoding="utf-8") as f:
    f.write("=== Client message ===\n" + CLIENT_TEXT + "\n\n")
    f.write("=== Prompt 1 (parse) ===\n" + parse_prompt.format(text=CLIENT_TEXT) + "\n\n")
    f.write("=== LLM output 1 ===\n" + json.dumps(inputs, indent=2) + f"\nmatch known answers: {parse_ok}\n\n")
    f.write("=== Simulation results (Python) ===\n" + json.dumps(results, indent=2) + "\n\n")
    f.write("=== LLM output 2 (summary) ===\n" + summary + "\n\n")
    f.write(f"=== Number check ===\nnumbers found: {found}\nnot from the model: {bad}\n")
print("Saved llm_demo_output.txt")
