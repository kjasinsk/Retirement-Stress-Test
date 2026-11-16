"""
Retirement Stress Test - Data Evidence (Problem 5, part 3)

Pulls the two public data sources the app needs and prints the first 10 rows of each:
  1. Damodaran (NYU Stern): yearly returns for stocks, T-bills and 10-year T-bonds, 1928 to today
  2. FRED: CPI-U inflation index (series CPIAUCNS, starts 1913), turned into yearly inflation rates

Setup:  pip install pandas xlrd requests
Run:    python data_pull.py
"""

import io
import time

import pandas as pd
import requests

DAMODARAN_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"  # backup inflation source
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,text/html,application/vnd.ms-excel,*/*",
}


def get(url, tries=3, **kwargs):
    """requests.get with a few retries, since public sites sometimes drop the connection."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  attempt {attempt} for {url[:50]}... failed: {type(e).__name__}")
            time.sleep(3 * attempt)
    raise ConnectionError(f"Could not download {url}")


def find_col(columns, *options):
    """First column whose name contains any option (case-insensitive, spaces ignored)."""
    squash = lambda x: x.lower().replace(" ", "")
    for opt in options:
        for c in columns:
            if squash(opt) in squash(c):
                return c
    raise KeyError(f"No column matching {options} in {columns}")


def load_damodaran_returns(url=DAMODARAN_URL):
    """Return a table of yearly returns: year, stocks, t_bills, t_bonds (as decimals, 0.10 = 10%)."""
    raw = get(url).content
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)

    for sheet in sheets.values():
        # The returns table starts on the row whose first cell says "Year"
        first_col = sheet.iloc[:, 0].astype(str).str.strip()
        for h in first_col.index[first_col == "Year"]:
            table = sheet.iloc[h + 1:].copy()
            table.columns = [" ".join(str(c).split()) for c in sheet.iloc[h]]
            table = table[pd.to_numeric(table["Year"], errors="coerce").between(1928, 2100)]
            if len(table) > 50:  # found the real yearly table
                cols = list(table.columns)
                with open("damodaran_columns.txt", "w") as f:  # record what the file contained
                    f.write("\n".join(cols))
                # Pick columns by name, not position (the file also has small caps, corporate bonds, etc.)
                out = pd.DataFrame({
                    "year": table["Year"].astype(int),
                    "stocks": pd.to_numeric(table[find_col(cols, "S&P")], errors="coerce"),
                    "t_bills": pd.to_numeric(table[find_col(cols, "T.Bill", "Bill")], errors="coerce"),
                    "t_bonds": pd.to_numeric(table[find_col(cols, "T.Bond", "Treasury Bond", "10-year")], errors="coerce"),
                })
                return out.dropna().reset_index(drop=True)
    raise ValueError("Could not find the yearly returns table in the Damodaran file")


def december_inflation(cpi):
    """cpi: DataFrame with date + cpi. Returns year, December CPI, Dec-to-Dec inflation."""
    december = cpi[cpi["date"].dt.month == 12].copy()  # matches calendar-year returns
    december["year"] = december["date"].dt.year
    december["inflation"] = december["cpi"].pct_change()
    return december[["year", "cpi", "inflation"]].dropna().reset_index(drop=True)


def load_fred_inflation(url=FRED_URL):
    """Main source: FRED CPI-U (CPIAUCNS, monthly since 1913), no API key needed."""
    cpi = pd.read_csv(io.StringIO(get(url).text))
    cpi.columns = ["date", "cpi"]  # FRED names them observation_date (or DATE) and CPIAUCNS
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi["cpi"] = pd.to_numeric(cpi["cpi"], errors="coerce")
    return december_inflation(cpi)


def load_bls_inflation(first=1913, last=None):
    """Backup source: the same CPI-U series straight from the BLS API (v1 = no key, 10 years per call)."""
    last = last or pd.Timestamp.today().year
    rows = []
    for start in range(first, last + 1, 10):
        body = {"seriesid": ["CUUR0000SA0"], "startyear": str(start),
                "endyear": str(min(start + 9, last))}
        r = requests.post(BLS_URL, json=body, headers=HEADERS, timeout=60)
        for obs in r.json()["Results"]["series"][0]["data"]:
            if obs["period"] == "M12":
                rows.append({"date": pd.Timestamp(int(obs["year"]), 12, 1), "cpi": float(obs["value"])})
        time.sleep(1)
    cpi = pd.DataFrame(rows).sort_values("date")
    return december_inflation(cpi)


def load_inflation():
    try:
        return load_fred_inflation(), "FRED (CPIAUCNS)"
    except Exception as e:
        print(f"FRED did not respond ({type(e).__name__}); using the BLS API instead.")
        return load_bls_inflation(), "BLS API (CUUR0000SA0, same CPI-U index)"


if __name__ == "__main__":
    pd.set_option("display.width", 120)

    returns = load_damodaran_returns()
    print(f"Damodaran yearly returns: {len(returns)} years "
          f"({returns.year.min()}-{returns.year.max()})")
    print(returns.head(10).to_string(index=False))

    inflation, source = load_inflation()
    print(f"\n{source} yearly inflation: {len(inflation)} years "
          f"({inflation.year.min()}-{inflation.year.max()})")
    print(inflation.head(10).to_string(index=False))

    # Join them into the one table the simulation samples from
    history = returns.merge(inflation[["year", "inflation"]], on="year")
    history.to_csv("market_history.csv", index=False)
    returns.to_csv("damodaran_returns.csv", index=False)
    inflation.to_csv("fred_inflation.csv", index=False)
    print(f"\nSaved market_history.csv: {len(history)} years with returns + inflation")
