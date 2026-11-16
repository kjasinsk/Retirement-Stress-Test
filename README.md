# Retirement Stress Test

MGT4160 Final Project. Team: Kaitlyn Jasinski, [teammate], [teammate]

A Shiny for Python app that answers one question for a client and their financial advisor: **how much can I safely spend each year in retirement, and when can I retire?**

The app runs 25,000 possible market futures, built from real 1928-2025 stock, bond and inflation data. It shows:
- the chance the money lasts to age 95
- the highest yearly spending that keeps that chance at a target (for example 85%)
- what is left in a bad, typical and good future

A **market returns** switch lets you use either the real 1928-2025 history or **J.P. Morgan Asset Management's 2027 Long-Term Capital Market Assumptions** (U.S. large-cap stocks 7.5%, U.S. intermediate Treasuries 4.6% a year over 10-15 years). With the forecast, every historical year is moved by the same amount so the long-run averages match, while the real crashes and swings stay. The numbers are in `LTCMA` at the top of `stress.py`, so they are easy to update each October.

An LLM (LangChain + Groq) does two jobs. It turns a plain-English description into the model inputs, and it writes a client summary. **The LLM never does math**: every number in its summary is checked against the model.

## What's in the folder

| File | What it does |
|---|---|
| `app.py` | The Shiny app: inputs, AI fill, results, 4 chart/table tabs, AI summary, client report download |
| `stress.py` | Simulation engine: saving years + retirement years, tax rate, "worse future", input checks |
| `simulate.py` | The original Monte Carlo model from Problem 5 (block bootstrap + binary search) |
| `report.py` | Shared results helpers and the downloadable client report |
| `llm_helpers.py` | The two LLM jobs (parse inputs, write summary) and the number check |
| `data_pull.py` | Downloads Damodaran returns + CPI inflation (FRED, BLS backup) into `market_history.csv` |
| `market_history.csv` | Cleaned data, 1928-2025 (so the app never depends on a live download) |
| `verify.py` | 20 model checks: hand calculation, sanity, Trinity Study, seed stability, history backtests, app engine, J.P. Morgan forecast |
| `eval_ai.py` | AI tests: 20 descriptions with known answers, 20 summaries checked number by number |
| `www/plotly.min.js` | Chart library served locally (charts work offline) |

## Run it on your computer

```
pip install -r requirements.txt
shiny run app.py
```
Then open http://127.0.0.1:8000. On Kaitlyn's PC, just double-click `run_app.bat` and open http://127.0.0.1:8001.

The AI buttons need a Groq API key. Put it in a file named `.env` next to `app.py`:
```
GROQ_API_KEY=your-key-here
```
`.env` is in `.gitignore`, so it is never uploaded to GitHub. Without a key, everything still works except the two AI buttons.

## Tests
```
python verify.py     # model checks -> verify_results.csv
python eval_ai.py    # AI checks   -> ai_eval_results.txt, ai_eval_parse.csv, ai_eval_summary.csv
```
On Windows: double-click `run_tests.bat` (output in `tests_log.txt`).

## Put it online (Posit Connect Cloud, free)
1. Put this folder in a **public GitHub repository**. Make sure `.env` is NOT included.
2. Go to https://connect.posit.cloud and sign in with GitHub.
3. Click **Publish**, choose **Shiny**, then pick the repository, the branch and `app.py` as the primary file.
4. Under **Advanced settings → Configure variables**, add a secret named `GROQ_API_KEY` with your key.
5. Click **Publish**. You get a public link to share.

Another option is shinyapps.io: `pip install rsconnect-python`, then `rsconnect deploy shiny . --name <account> --title retirement-stress-test`. Set `GROQ_API_KEY` in the app's settings there.

## Assumptions and limits
- The future looks like a reshuffled version of 1928-2025, or of that history adjusted to J.P. Morgan's forecast. The "worse future" slider tests lower stock returns. J.P. Morgan's forecast covers 10-15 years; the app applies it to the whole plan.
- Spending, Social Security and contributions grow with inflation. 0.5% yearly fee. Rebalanced yearly. Plan runs to age 95.
- Taxes are a single flat rate on withdrawals. Real taxes depend on account type and state.
- Educational illustration, not financial advice.
