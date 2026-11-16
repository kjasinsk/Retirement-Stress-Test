"""
Retirement Stress Test - Shiny for Python app

Sidebar: describe your situation in plain English (the LLM fills the inputs), or type the inputs directly.
Main area: chance of success, highest safe spending, bad-case money left, balance fan chart, spending curve,
retirement-age comparison and an AI-written client summary whose numbers are checked against the model.
All math is done in Python (stress.py); the LLM (llm_helpers.py, LangChain + Groq) only reads and writes words.

Run:  shiny run app.py   then open http://127.0.0.1:8000   (or double-click run_app.bat on Windows)
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.offline
import pandas as pd
from shiny import App, reactive, render, req, ui

import llm_helpers
from report import facts_for, report_html, summarize_run
from stress import END_AGE, LTCMA, forecast_history, input_problems, make_paths, max_safe_spending, success

HERE = Path(__file__).parent
HISTORY = pd.read_csv(HERE / "market_history.csv")
BLUE, RED, GOLD, INK = "#0F2236", "#A3412F", "#8C6A47", "#23303D"   # navy, brick, bronze, ink
# Plotly's JavaScript is served from ./www, so the charts work without an internet connection
WWW = Path(__file__).with_name("www")
if not (WWW / "plotly.min.js").exists():
    WWW.mkdir(exist_ok=True)
    (WWW / "plotly.min.js").write_text(plotly.offline.get_plotlyjs(), encoding="utf-8")


def chart(fig, height=330):
    """Plotly figure -> HTML that Shiny inserts into the page (hover shows exact values)."""
    fig.update_layout(height=height, margin=dict(l=55, r=20, t=10, b=45), template="simple_white",
                      font=dict(size=12, family="Segoe UI, Helvetica Neue, Arial", color="#23303D"),
                      paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.08, x=0))
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}))

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.head_content(ui.tags.script(src="plotly.min.js"), ui.tags.link(rel="stylesheet", href="style.css")),
        ui.div("Describe the client", class_="section-label"),
        ui.input_text_area("story", None, rows=4,
                           value="I'm 58 and have about $900k saved. I put away $20k a year, want to retire at 63 and "
                                 "spend around $55k a year. I expect about $2,000 a month from Social Security, and "
                                 "I'm invested roughly 60/40."),
        ui.input_action_button("fill", "Fill inputs from description (AI)", class_="btn-outline-primary"),
        ui.output_ui("fill_status"),
        ui.div("Client profile", class_="section-label"),
        ui.layout_columns(ui.input_numeric("age", "Current age", 58, min=30, max=80),
                          ui.input_numeric("retire_age", "Retire at", 63, min=40, max=80)),
        ui.input_numeric("savings", "Savings today ($)", 900_000, step=50_000),
        ui.input_numeric("contribution", "Saving per year until retirement ($)", 20_000, step=1_000),
        ui.input_numeric("spending", "Yearly spending in retirement, today's $", 55_000, step=1_000),
        ui.input_numeric("ss", "Social Security per year ($)", 24_000, step=1_000),
        ui.div("Portfolio and assumptions", class_="section-label"),
        ui.input_radio_buttons("scenario", "Market returns", {
            "history": "Historical (1928-2025)",
            "ltcma": "J.P. Morgan 2027 LTCMA forecast"}, selected="history"),
        ui.panel_conditional(
            "input.scenario === 'ltcma'",
            ui.layout_columns(
                ui.input_numeric("ltcma_stocks", "U.S. stocks (%/yr)", LTCMA["stocks"] * 100, step=0.1),
                ui.input_numeric("ltcma_bonds", "U.S. bonds (%/yr)", LTCMA["bonds"] * 100, step=0.1)),
            ui.p("Source: J.P. Morgan Asset Management, 2027 Long-Term Capital Market Assumptions (Oct 2026): "
                 "U.S. large cap equities 7.5%, U.S. intermediate Treasuries 4.6%, 10-15 year compound returns. "
                 "Historical years are shifted so their averages match; crashes and swings stay.",
                 class_="msg-note")),
        ui.input_slider("stocks", "Stocks in the portfolio (%)", 0, 100, 60, step=5),
        ui.input_slider("target", "Target chance of success (%)", 70, 99, 85),
        ui.input_slider("haircut", "Worse future: cut stock returns by (points per year)", 0.0, 4.0, 0.0, step=0.5),
        ui.input_slider("tax", "Tax rate on withdrawals (%)", 0, 40, 0, step=1),
        ui.output_ui("problems"),
        ui.input_action_button("run", "Run stress test", class_="btn-primary"),
        width=370,
    ),
    ui.div(ui.h2("Retirement Plan Stress Test"), ui.div(ui.output_text("subtitle"), class_="sub"), class_="page-head"),
    ui.layout_columns(
        ui.value_box("Chance the money lasts to age 95", ui.output_text("vb_success")),
        ui.value_box("Highest safe yearly spending", ui.output_text("vb_safe")),
        ui.value_box("Left at 95 in a bad future (today's $)", ui.output_text("vb_bad")),
    ),
    ui.navset_card_underline(
        ui.nav_panel("Wealth over time", ui.output_ui("fan")),
        ui.nav_panel("Spending vs. chance of success", ui.output_ui("curve")),
        ui.nav_panel("Retire earlier or later", ui.output_data_frame("ages")),
        ui.nav_panel("Methodology", ui.markdown(
            "- **Market futures:** 25,000 paths, each stitched from random 5-year stretches of real 1928-2025 "
            "stock, bond and inflation data (keeps crashes and streaks like the 1970s).\n"
            "- **Each year:** contributions before retiring; after retiring, spending minus Social Security is "
            "withdrawn; both grow with inflation; 0.5% yearly fee; rebalanced to the stock/bond mix.\n"
            "- **J.P. Morgan forecast option:** J.P. Morgan Asset Management's 2027 Long-Term Capital Market "
            "Assumptions expect 7.5% a year for U.S. large-cap stocks and 4.6% for U.S. intermediate Treasuries over "
            "10-15 years (history averaged 10.0% and 4.5%). Every historical year's return is moved by the same amount "
            "so the long-run average matches the forecast; crashes, swings and inflation stay as they happened.\n"
            "- **Highest safe spending:** binary search for the most the client can spend with at least the target chance.\n"
            "- **AI:** reads the description into inputs and writes the summary from numbers Python formatted. "
            "It never does math, and every number in its summary is checked.\n"
            "- **Taxes:** one flat rate on money taken out of savings (the withdrawal is grossed up so the client still "
            "gets their spending after tax). Real taxes depend on account type and state."))),
    ui.card(ui.card_header("Client summary"),
            ui.layout_columns(
                ui.input_action_button("summarize", "Write AI summary", class_="btn-outline-primary btn-sm"),
                ui.download_button("report", "Download client report", class_="btn-outline-secondary btn-sm")),
            ui.output_ui("summary")),
    ui.div("Educational illustration based on historical market data, not investment advice. Past performance does "
           "not guarantee future results. Taxes are simplified to one flat rate.", class_="disclaimer"),
    title=ui.span("Retirement Stress Test", ui.span("Wealth planning \u00b7 Monte Carlo analysis", class_="tagline"),
                  class_="brand"),
    fillable=False,
)


def server(input, output, session):
    fill_msg = reactive.value("")
    summary_state = reactive.value(None)

    # ---- AI: plain English -> inputs (user sees and can change them before running) ----
    @reactive.effect
    @reactive.event(input.fill)
    def _fill():
        try:
            p = llm_helpers.parse_inputs(input.story())
        except Exception as e:                                    # no key, no network, bad JSON ...
            fill_msg.set(f"AI could not read the description: {type(e).__name__}. Type the inputs instead.")
            return
        updates = {"age": "current_age", "retire_age": "retirement_age", "savings": "savings",
                   "contribution": "annual_contribution", "spending": "annual_spending", "ss": "social_security"}
        filled = 0
        for input_id, key in updates.items():
            if p.get(key) is not None:
                ui.update_numeric(input_id, value=p[key])
                filled += 1
        if p.get("stock_share") is not None:
            ui.update_slider("stocks", value=round(float(p["stock_share"]) * 100))
            filled += 1
        fill_msg.set(f"AI filled {filled} of 7 inputs. Check them, then press Run stress test.")

    @render.ui
    def fill_status():
        return ui.p(fill_msg(), class_="msg-ok") if fill_msg() else None

    # ---- input checks (live, as the user types) ----
    def current_plan():
        return dict(current_age=input.age(), retire_age=input.retire_age(), savings=input.savings(),
                    contribution=input.contribution(), spending=input.spending(), social_security=input.ss(),
                    stock_share=input.stocks() / 100, tax_rate=input.tax() / 100)

    @reactive.calc
    def checks():
        return input_problems(**current_plan())

    @render.ui
    def problems():
        errors, warnings = checks()
        items = [ui.p("\u26d4 " + e, class_="msg-bad") for e in errors]
        items += [ui.p("\u26a0 " + w, class_="msg-warn") for w in warnings]
        return ui.div(*items) if items else None

    # ---- the model (runs on the button, and once at start) ----
    @reactive.calc
    @reactive.event(input.run, ignore_none=False)
    def sim():
        req(not checks()[0])                                      # don't run on impossible inputs
        plan = current_plan()
        plan["current_age"], plan["retire_age"] = int(plan["current_age"]), int(plan["retire_age"])
        history, scenario = HISTORY, None
        if input.scenario() == "ltcma":
            s_ret, b_ret = (input.ltcma_stocks() or LTCMA["stocks"] * 100), (input.ltcma_bonds() or LTCMA["bonds"] * 100)
            history = forecast_history(HISTORY, s_ret / 100, b_ret / 100)
            scenario = f"J.P. Morgan 2027 LTCMA forecast: U.S. stocks {s_ret:.1f}%, U.S. bonds {b_ret:.1f}% a year"
        paths = make_paths(history, END_AGE - plan["current_age"], stock_haircut=input.haircut() / 100)
        return summarize_run(paths, plan, target=input.target() / 100, haircut=input.haircut(), scenario=scenario)

    @reactive.effect
    def _reset_summary():
        sim()                                                     # an old summary no longer matches new results
        summary_state.set(None)

    @render.text
    def subtitle():
        r = sim()
        worse = f" Stock returns cut by {r['haircut']:.1f} points a year." if r["haircut"] else ""
        worse += f" Withdrawals taxed at {r['plan']['tax_rate']:.0%}." if r["plan"]["tax_rate"] else ""
        basis = ("1928-2025 history adjusted to J.P. Morgan's 2027 forecast" if r["scenario"]
                 else "1928-2025 history")
        return (f"25,000 market futures from {basis}, age {r['plan']['current_age']} to {END_AGE}, "
                f"retiring at {r['plan']['retire_age']}.{worse}")

    @render.text
    def vb_success():
        return f"{sim()['success']:.1%}"

    @render.text
    def vb_safe():
        return f"${sim()['safe']:,.0f}"

    @render.text
    def vb_bad():
        return f"${sim()['p10']:,.0f}"

    @render.ui
    def fan():
        r = sim()
        ages = r["plan"]["current_age"] + np.arange(r["real"].shape[0])
        p10, p50, p90 = np.percentile(r["real"], [10, 50, 90], axis=1) / 1e6
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ages, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ages, y=p10, fill="tonexty", fillcolor="rgba(140,106,71,0.16)", line=dict(width=0),
                                 name="10th-90th percentile", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ages, y=p50, name="Typical future (median)", line=dict(color=BLUE, width=3),
                                 hovertemplate="Age %{x}: $%{y:.2f}M<extra>median</extra>"))
        fig.add_trace(go.Scatter(x=ages, y=p10, name="Bad future (10th pct.)", line=dict(color=RED, width=2),
                                 hovertemplate="Age %{x}: $%{y:.2f}M<extra>10th pct.</extra>"))
        top = float(p90.max()) * 1.05
        fig.add_trace(go.Scatter(x=[r["plan"]["retire_age"]] * 2, y=[0, top], mode="lines+text", text=["  retire", ""],
                                 textposition="top right", line=dict(color=INK, dash="dot", width=1.5),
                                 showlegend=False, hoverinfo="skip"))
        fig.update_xaxes(title="Age"); fig.update_yaxes(title="$ millions, today's dollars", rangemode="tozero")
        return chart(fig)

    @render.ui
    def curve():
        r = sim()
        plan = dict(r["plan"])
        grid = np.linspace(max(r["safe"] * 0.5, 1000), r["safe"] * 1.6 + 1000, 14)
        rates = []
        for x in grid:
            plan["spending"] = x
            rates.append(success(r["paths"], **plan) * 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=rates, mode="lines+markers", line=dict(color=BLUE, width=3),
                                 name="Chance the money lasts", hovertemplate="$%{x:,.0f}/yr: %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[grid[0], grid[-1]], y=[r["target"] * 100] * 2, mode="lines",
                                 line=dict(color=GOLD, dash="dash", width=1.5), name=f"Target {r['target']:.0%}",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r["safe"]], y=[r["target"] * 100], mode="markers+text", marker=dict(color=GOLD, size=12),
                                 text=[f"highest safe: ${r['safe']:,.0f}"], textposition="bottom left", name="Highest safe"))
        fig.add_trace(go.Scatter(x=[r["plan"]["spending"]], y=[r["success"] * 100], mode="markers+text",
                                 marker=dict(color=RED, size=12), text=[f"your plan: {r['success']:.1%}"],
                                 textposition="bottom left", name="Your plan"))
        fig.update_layout(showlegend=True)
        fig.update_xaxes(title="Yearly spending in retirement (today's dollars)", tickprefix="$", tickformat=",.0f")
        fig.update_yaxes(title="Chance (%)", range=[0, 105])
        return chart(fig)

    @reactive.calc
    def age_rows():
        r = sim()
        rows = []
        for retire in sorted({r["plan"]["retire_age"] + d for d in (-2, 0, 2)}):
            if not (r["plan"]["current_age"] <= retire < END_AGE):
                continue
            plan = dict(r["plan"], retire_age=retire)
            rows.append({"Retire at": retire,
                         "Chance the money lasts": f"{success(r['paths'], **plan):.1%}",
                         f"Highest safe spending ({r['target']:.0%} target)":
                             f"${max_safe_spending(r['paths'], target=r['target'], **plan):,.0f}"})
        return rows

    @render.data_frame
    def ages():
        return render.DataGrid(pd.DataFrame(age_rows()), width="100%")

    # ---- AI summary from Python-formatted numbers, then a number check ----
    @reactive.effect
    @reactive.event(input.summarize)
    def _summarize():
        facts = facts_for(sim())
        try:
            text = llm_helpers.write_summary(facts)
            summary_state.set({"text": text, "bad": llm_helpers.check_numbers(text, facts)})
        except Exception as e:
            summary_state.set({"text": f"AI summary unavailable ({type(e).__name__}).", "bad": None})

    @render.ui
    def summary():
        st = summary_state()
        if st is None:
            return ui.p("Press 'Write AI summary' after running the stress test.", style="color: #6B7480;")
        if st["bad"] is None:
            badge = ui.span("")
        elif st["bad"]:
            badge = ui.span(f"Number check: FAIL ({st['bad']} not from the model)", class_="check-fail")
        else:
            badge = ui.span("\u2713 Number check: PASS, every number came from the model", class_="check-pass")
        return ui.div(ui.p(st["text"], class_="summary-text"), badge)


    # ---- downloadable one-page client report ----
    @render.download(filename=lambda: f"retirement_stress_test_{pd.Timestamp.today():%Y-%m-%d}.html")
    def report():
        yield report_html(sim(), summary_state(), age_rows())


app = App(app_ui, server, static_assets=WWW)
