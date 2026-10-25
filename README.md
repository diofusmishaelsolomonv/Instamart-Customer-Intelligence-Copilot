# Instamart Customer Intelligence & Support Copilot

[![Reproducible project validation](https://github.com/diofusmishaelsolomonv/Instamart-Customer-Intelligence-Copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/diofusmishaelsolomonv/Instamart-Customer-Intelligence-Copilot/actions)

An offline-first Python capstone demonstrating data cleaning, customer analytics, privacy-aware support-ticket triage, policy retrieval, and a bounded tool-using refund agent. The project uses deterministic synthetic data and a local mock LLM; no API key or paid inference service is required.

> **Educational simulation only.** This project is not affiliated with Swiggy/Instamart. All customers, orders, tickets, policies, and refund scenarios are fictional. Refund actions update a simulated ledger only; they do not move real money.

## Project modules

| Module | What it does | Main outputs |
|---|---|---|
| **Part A — Data foundation** | Audits and cleans CSV data, persists clean tables in SQLite, runs seven SQL analyses, and performs exploratory analysis. | `data/clean/`, `data/instamart.db`, `outputs/q1_result.csv`–`q7_result.csv`, audit summaries |
| **Part B — Customer intelligence** | Builds RFM features, evaluates K-means clusters and PCA, trains/evaluates lapse models, and prepares a retention shortlist. | Segment profiles, cluster metrics, lapse-model summary and saved model |
| **Part C — Support triage** | Redacts PII before inference, skips empty/oversized tickets, retries transient failures, validates classifications, and tracks token budgets. | Triage results, retry log, zero-shot/few-shot and budget demonstrations |
| **Part D — Policy RAG and refund agent** | Retrieves policy text with TF-IDF, evaluates grounded answers, and runs a bounded, resumable agent with simulated refund actions. | Policy evaluation, agent run logs, refund ledger, part summary and checkpoints |

The implementation is intended for learning and demonstration. Synthetic-data results and mock-LLM accuracy should not be interpreted as real customer or production-model performance.

## Requirements

- Python **3.11 or newer**
- pip and internet access for the initial dependency installation
- Windows, macOS, or Linux

After dependencies are installed, data generation and the pipeline run locally without API credentials or hosted LLM calls.

## Quick start

Run from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python run_all.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python run_all.py
```

A successful full run ends with `PIPELINE FINISHED`. The Part A validator prints `ALL CHECKPOINTS PASSED` when all clean-data assertions pass. The generator prints a summary of generated customer, product, order, and ticket counts.

## Repository layout

```text
.
├── .github/workflows/ci.yml  # automated reproducibility check
├── generate_data.py          # deterministic synthetic data/policy generator
├── mock_llm.py               # offline mock model interface
├── run_all.py                # runs Parts A through D in order
├── validate_data.py          # independent Part A checkpoint validator
├── requirements.txt
├── sql/queries.sql
├── src/
│   ├── part_a.py
│   ├── part_b.py
│   ├── part_c.py
│   └── part_d.py
├── data/                     # generated at runtime
├── outputs/                  # generated at runtime
└── state/                    # generated agent checkpoints at runtime
```

Generated datasets, SQLite databases, trained models, reports, and agent state are reproducible runtime artifacts and are not all committed to `main`. The CI workflow generates them from source and uploads them as a downloadable artifact for each successful run. Download the artifact from the relevant run's **Artifacts** section if you need the generated files.

## Automated validation

GitHub Actions runs on pushes to `main`, pull requests targeting `main`, and manual dispatch. It:

1. Sets up Python 3.11 and installs the pinned dependencies.
2. Compiles the project Python files to catch syntax errors.
3. Generates the synthetic raw datasets and policy documents.
4. Executes the full Part A–D pipeline.
5. Runs the independent clean-data validator.
6. Uploads generated data and outputs as a workflow artifact.

Check the [Actions page](https://github.com/diofusmishaelsolomonv/Instamart-Customer-Intelligence-Copilot/actions) for the latest run status and downloadable artifacts.

## Safety, privacy, and reproducibility notes

- Data and policy content are synthetic and fictional.
- Email addresses, phone numbers, and order identifiers in ticket text are redacted before the mock inference step.
- The refund agent is bounded and uses a simulated refund ledger; it does not initiate payments.
- Do not add API keys, access tokens, real customer information, or other secrets to the repository.
- Model scores and policy evaluation results describe this deterministic educational simulation only.

## Troubleshooting

- **PowerShell blocks activation:** use the activation guidance shown by your local PowerShell policy, or run the environment's Python executable directly.
- **Missing generated files:** run `python generate_data.py` before `python run_all.py`.
- **Dependency or Python version errors:** use Python 3.11+ in a fresh virtual environment and reinstall from `requirements.txt`.
- **Need generated results from CI:** open a successful Actions run and download `instamart-copilot-run-outputs` under **Artifacts**.
