# Instamart Customer Intelligence & Support Copilot

A reproducible, offline-first Python capstone project connecting data quality, customer analytics, privacy-aware support triage, policy retrieval, and a tool-using refund agent. Uses synthetic data and an offline mock LLM; no API key or paid model is required.

> **Important:** This is an educational simulation. It is not affiliated with Swiggy/Instamart. All customers, orders, tickets, policies, and refunds are synthetic. Refund actions only update a simulated ledger and do not move real money.

## Project modules

- **Part A — Data Foundation:** raw data cleaning, SQLite database, SQL analyses, data validation and exploratory analysis.
- **Part B — Customer Intelligence:** RFM analysis, K-means segmentation, PCA, lapse prediction, model tuning and retention shortlist.
- **Part C — Support Ticket Triage:** PII redaction, ticket classification, retries, output validation, token budgets and customer-aware routing.
- **Part D — Policy RAG & Refund Agent:** TF-IDF retrieval, grounded policy answers, tool-calling agent, refund ledger, safety limits and crash recovery.

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python run_all.py
```

Keep the supplied `generate_data.py` and `mock_llm.py` unchanged, as required by the project brief. See the source tree and scripts for detailed run commands and validation.

## Expected project layout

```text
.
├── generate_data.py
├── mock_llm.py
├── run_all.py
├── requirements.txt
├── src/
├── sql/
├── data/
├── outputs/
├── state/
└── README.md
```

Generated datasets and model artifacts may be reproducible outputs; follow the project brief about which files must be committed. Never commit credentials, access tokens, or personal data.
