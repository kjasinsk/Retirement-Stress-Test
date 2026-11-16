"""
Retirement Stress Test - the two LLM jobs used by the app (LangChain + Groq, like our class examples).

  parse_inputs(text)   plain English -> dict of model inputs (the user confirms them before anything runs)
  write_summary(facts) client summary from numbers Python has ALREADY formatted ("$61,293", "95.0%")
  check_numbers(...)   every number in the summary must be one of the numbers we gave it
The LLM never does math.
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

for env in (Path("c:/MGT4170/MyKeys/.env"), Path(__file__).with_name(".env")):
    if env.exists():
        load_dotenv(env)

MODEL_NAME = "openai/gpt-oss-20b"


def _model():
    from langchain_groq import ChatGroq                     # imported here so the app still runs without it
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No GROQ_API_KEY found (c:/MGT4170/MyKeys/.env or a .env next to app.py)")
    return ChatGroq(model=MODEL_NAME, api_key=key, temperature=0)


PARSE_PROMPT = """Extract retirement-planning inputs from the client's message. Return ONLY a JSON object:
{{"current_age": int, "retirement_age": int, "savings": number (US dollars),
  "annual_contribution": number (US dollars saved per year until retirement),
  "annual_spending": number (US dollars per year in retirement), "social_security": number (US dollars per YEAR),
  "stock_share": number between 0 and 1}}
Convert monthly amounts to yearly and "k" to thousands. Use null for anything the client did not say.
Do not calculate anything else.

Client message: {text}"""

SUMMARY_PROMPT = """You are writing for a financial advisor's client. Write 3 short sentences in plain English that
explain these retirement stress-test results. Copy every number EXACTLY as written below (same symbols, commas and
decimals). Do not calculate, round, combine or invent any number, and do not give investment advice.

{facts}"""


def parse_inputs(text: str) -> dict:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    return (PromptTemplate.from_template(PARSE_PROMPT) | _model() | JsonOutputParser()).invoke({"text": text})


def write_summary(facts: dict) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    chain = PromptTemplate.from_template(SUMMARY_PROMPT) | _model() | StrOutputParser()
    for _ in range(3):                       # the model occasionally returns an empty answer; ask again
        text = chain.invoke({"facts": lines}).strip()
        if text:
            return text
    raise ValueError("the AI returned an empty summary 3 times")


def _numbers(text):
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text.replace("\u2011", "-"))}


def check_numbers(summary: str, facts: dict) -> list:
    """Numbers in the summary that are not in the facts we gave the LLM (empty list = PASS)."""
    allowed = _numbers(" ".join(str(k) + " " + str(v) for k, v in facts.items()))
    return sorted(_numbers(summary) - allowed)
