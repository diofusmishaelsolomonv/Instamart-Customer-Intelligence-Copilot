# Instamart Customer Intelligence & Support Copilot

[![Reproducible project validation](https://github.com/diofusmishaelsolomonv/Instamart-Customer-Intelligence-Copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/diofusmishaelsolomonv/Instamart-Customer-Intelligence-Copilot/actions)

## Overview

An offline-first Python capstone joining four stages: (A) clean and audit synthetic order data, store it in SQLite and answer business questions; (B) segment customers and predict lapse; (C) redact and triage support tickets with privacy, retry, validation and budget guardrails; and (D) retrieve policy passages and run a resumable tool-using refund agent. All people, orders, tickets and policies are synthetic; this is not real Instamart data or policy. Refund actions only update a local simulated ledger. The supplied `generate_data.py` and `mock_llm.py` are used as offline dependencies and do not require a hosted model or API key.

## Reproduce

Run these commands from the repository root. Use Python 3.11 or newer.

**Windows PowerShell**

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`
7. `python validate_data.py`

**macOS / Linux**

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`
7. `python validate_data.py`

The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The pipeline runs Parts A–D in order. The validator must print exactly `ALL CHECKPOINTS PASSED`. GitHub Actions repeats generation, the full pipeline, validation, and expected-file checks; see the [Actions runs](https://github.com/diofusmishaelsolomonv/Instamart-Customer-Intelligence-Copilot/actions).

## Part A

### Raw-data audit

The three tables were loaded with pandas. Missing percentages below are calculated over each table's row count; the explicit blank/invalid audit is separate from pandas' parsed missing-value count.

| Table | Shape | Column | pandas dtype | Missing count | Missing % |
|---|---:|---|---|---:|---:|
| customers.csv | 1,200 × 5 | customer_id | object | 0 | 0.00% |
| | | city | object | 0 | 0.00% |
| | | signup_date | object | 0 | 0.00% |
| | | age | float64 | 61 | 5.08% |
| | | gender | object | 30 | 2.50% |
| products.csv | 40 × 4 | product_id | object | 0 | 0.00% |
| | | product_name | object | 0 | 0.00% |
| | | category | object | 0 | 0.00% |
| | | price_inr | int64 | 0 | 0.00% |
| orders.csv | 13,084 × 9 | order_id | object | 0 | 0.00% |
| | | customer_id | object | 0 | 0.00% |
| | | product_id | object | 0 | 0.00% |
| | | quantity | int64 | 0 | 0.00% |
| | | order_ts | object | 0 | 0.00% |
| | | status | object | 0 | 0.00% |
| | | discount_pct | int64 | 0 | 0.00% |
| | | order_amount | object | 53 | 0.41% |
| | | delivery_minutes | float64 | 2,541 | 19.42% |

| Raw audit checkpoint | Result |
|---|---:|
| Order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id values | 12,309 |
| Non-numeric order_amount values | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank gender values | 30 |

### Cleaning and storage

Rules are applied in order and the change counts below come from the reproducible pipeline.

- **Customers:** trimmed and title-cased city (476 values standardised; exactly six cities remain); mapped gender variants to `male`, `female`, or `unknown` (950 values standardised); set 6 impossible ages outside 10–100 to missing, then filled all 67 missing/invalid ages with each city's median using `groupby('city').transform('median')`; converted `signup_date` to ISO date text for storage.
- **Orders:** converted `order_ts` to datetime; removed 119 exact duplicate rows; kept the latest timestamp per `order_id`, removing 656 earlier status records; converted amounts to numeric and imputed 112 remaining missing amounts from catalogue price × quantity × (1 − discount/100), rounded to 2 decimals; cleared 105 negative delivery values remaining after duplicate/final-record resolution; imputed 774 missing delivery values on completed orders with the completed-order median for the product category. Cancelled-order delivery times remain missing because those orders were not delivered.
- **Products:** converted `price_inr` to numeric. Clean tables are written to `data/clean/{customers,products,orders}.csv`; SQLite `data/instamart.db` contains `customers`, `products`, and `orders`. Stored timestamps use `YYYY-MM-DD HH:MM:SS`.

| Clean checkpoint | Result |
|---|---:|
| Customers / products / final orders | 1,200 / 40 / 12,309 |
| Customers by city | Pune 227; Delhi 210; Bengaluru 197; Chennai 191; Hyderabad 191; Mumbai 184 |
| Customers by gender | female 610; male 560; unknown 30 |
| Missing ages after cleaning | 0; all ages are 18–60 |
| Completed / cancelled orders | 11,110 / 1,199 |
| Pending rows / missing order amounts | 0 / 0 |
| Missing delivery minutes: completed / cancelled | 0 / 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

### SQL and pandas analysis

All seven queries are saved in `sql/queries.sql` and executed with `pandas.read_sql_query` against the SQLite database.

| Query | Business question / technique | Result checkpoint |
|---|---|---|
| Q1 | Completed orders with amount > INR 500; group by customer and HAVING sum > INR 8,000 | 30 customers |
| Q2 | Completed revenue by city and product category; three-table join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 | Products priced above average catalogue price; scalar subquery | 11 products |
| Q4 | Customers without completed orders; LEFT JOIN | 25 customers |
| Q5 | Above-average customer completed spend; two CTEs | 550 customers; average customer spend INR 2,552.06 |
| Q6 | Per-city spend ranking with RANK, DENSE_RANK and ROW_NUMBER | 1,175 customers; 8 rows have RANK different from ROW_NUMBER |
| Q7 | Daily completed revenue and cumulative window sum | 183 days; final cumulative revenue INR 2,998,668.80 |

Q6's top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10). Ties can share a RANK value, whereas ROW_NUMBER still assigns a unique sequential number, so the rankings differ on tied rows.

The pandas pivot-table cross-check of Q2 has the same INR 2,998,668.80 grand total. Per-city completed-order stats:

| City | Sum (INR) | Mean (INR) | Count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**Exploratory analysis (completed-order amount):** mean INR 269.91; median INR 135.00; skewness 4.61; Q1 INR 64.80; Q3 INR 297.00; lower fence −INR 283.50; upper fence INR 645.30; 0 lower and 897 upper outliers. The data describes synthetic grocery and small-electronics order amounts, discounts, quantities and delivery times. The mean is much higher than the median and skewness is strongly positive, indicating a long right tail. There are 897 high-side outliers by the 1.5×IQR rule, and no low-side outliers. The largest absolute correlation is order_amount with discount_pct (−0.095), followed by order_amount with quantity (+0.087); these relationships are weak, and correlation of this size does not establish causation.

## Part B

### RFM and K-means segmentation

RFM is calculated from completed orders as of 2025-10-01 00:00:00. The resulting table contains 1,175 customers: recency median 19 days (minimum 0, maximum 180), frequency mean 9.46 (maximum 33), monetary mean INR 2,552.06 (maximum INR 14,766.55). The three RFM features are standardised before K-means; each K uses `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 is selected: it has the best silhouette in this sweep, a clear inertia reduction, and four interpretable campaign groups. Inertia normally falls as K rises because adding clusters lets points sit closer to their assigned centroids; K=8's lower inertia than K=2 is therefore expected and alone does not show that eight useful, stable business segments exist.

| Segment name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---:|---:|---:|---:|---|
| Premium | 138 | 24.12 | 6.77 | 7,129.70 | Offer relevant cross-category bundles and protect service quality |
| Frequent Essentials | 419 | 8.77 | 19.40 | 3,410.02 | Encourage repeat baskets with low-friction loyalty rewards |
| Occasional Essentials | 436 | 28.86 | 3.77 | 1,057.62 | Use personalised replenishment reminders and targeted offers |
| At-Risk / Low-Activity | 182 | 109.04 | 2.20 | 685.99 | Run a low-cost win-back campaign and monitor response |

The segment with the highest mean monetary value is marked `is_premium=True`. Cluster IDs are arbitrary labels, not a ranked scale. PCA on the scaled RFM matrix explains 0.6343 and 0.2180 of variance in its first two components (total 0.8523). Output: `outputs/customer_segments.csv`.

### Lapse model

The cutoff is 2025-09-01. The cohort contains customers with a completed order before the cutoff; `lapsed=1` means no completed order during 2025-09-01 through 2025-09-30. The cohort has 1,156 customers, 443 lapsed (38.32%), and a stratified 924/232 train/test split. The always-not-lapsed baseline has 61.64% accuracy and lapsed-class F1 = 0.

Features are built only from pre-cutoff orders and the clean customer table: recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite category, city, age, gender and tenure. Whole-window segment labels are not model features. Numeric imputation, scaling and categorical imputation/one-hot encoding are inside a scikit-learn preprocessing pipeline fitted only on training rows; it is applied unchanged to test rows. No order dated 2025-09-01 or later enters the features.

| Model | Grid | Best setting | CV F1 | Test precision | Test recall | Test F1 | Test accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| Logistic regression | C = 0.01, 0.1, 1, 10, 100 | C = 10 | 0.6565 | 0.5938 | 0.6404 | 0.6162 | 0.6940 |
| Random forest | max_depth = 3, 5, 8; min_samples_leaf = 5, 10, 20 | depth = 8, leaf = 5 | 0.6845 | 0.6129 | 0.6404 | 0.6264 | 0.7069 |

The random forest is selected for the highest cross-validated F1 and slightly stronger held-out F1/precision; it meets the required 0.55 F1 checkpoint. Missing a lapsing customer loses a chance to retain them, while targeting a loyal customer wastes campaign cost, so recall and precision both matter. The top five random-forest features are frequency (0.2566), monetary (0.2544), recency (0.1161), average delivery (0.0685) and average discount (0.0597). Logistic-regression coefficient signs: frequency −1.589 (more completed orders associate with lower lapse odds), monetary +0.199 (higher spend associates with higher lapse odds conditional on the other features), and Delhi city −0.353 versus the dropped reference city.

The selected model is saved and reloaded from `outputs/lapse_model.joblib`; test predictions match exactly after reload. The Premium-only test-set retention shortlist is `outputs/retention_list.csv`; 1 customer in that list truly lapsed. **The retention list is illustrative, not a leakage-free evaluation, because the Premium flag is derived from whole-window RFM including the September label window.**

Class-balance experiment: oversampling was applied only to training rows (with replacement and seed 42); the original test set remained unchanged. Logistic-regression precision/recall/F1 changed from 0.5938/0.6404/0.6162 to 0.5954/0.8764/0.7091. Recall moved most, increasing by about 0.236, so the team would catch more lapsing customers at the cost of more false-positive outreach.

## Part C

Tickets are processed in source-file order. Empty and over-400-token tickets are not sent to the model; the user message contains only the redacted ticket text. Redaction order is email `[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]`, phone `\b[6-9]\d{9}\b` → `[PHONE]`, then `ORD-\d{6}` → `[ORDER_ID]`. Assertions check that none of the three patterns remains in redacted text.

| Guardrail / result | Value |
|---|---:|
| Empty tickets skipped | 4 |
| Overlong tickets flagged (>400 estimated tokens) | 4 (about 734 tokens each) |
| Email / phone / order-ID substitutions | 55 / 91 / 132 |
| Total substitutions / remaining PII matches | 278 / 0 |
| Transient failures recovered by retry | 19 / 19 |
| Full-run status counts | 135 classified_json; 12 classified_keyword; 9 flagged_invalid_output; 4 flagged_long; 4 skipped_empty |

The zero-shot and few-shot system prompts both specify CLASSIFY, the four allowed categories, and JSON-only output. The few-shot prompt includes `EXAMPLES:` and labelled examples including fraud. Every model call is retried up to three times with exponential backoff and jitter (BASE_DELAY 0.05 s); the retry log records ticket id, retry number and wait. Output is accepted only when valid JSON has an allowed category or a non-JSON response contains exactly one allowed category word extracted with `re.findall(r"[a-z]+", text.lower())`; all other outputs go to human review.

### Prompt comparison and confusion table

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 10,180 | 69.25 |
| Few-shot | 147 | 100% (147/147) | 14,548 | 98.97 |

Few-shot used 4,368 additional tokens overall (about 29.72 more per classified ticket) and raised this deterministic mock's accuracy by 14.3 percentage points. In this simulation, the extra context is worthwhile where a wrong fraud/billing classification is costly; in a production system the trade-off should be measured against real, reviewed labels.

| Actual \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

With the zero-shot prompt, the mock misclassifies 11 fraud and 10 technical tickets as billing. The 6000-token zero-shot budget demonstration is saved to `outputs/triage_budget_demo.csv`; it ends 64 tickets as `budget_exhausted` (80 JSON-classified, 7 keyword-classified, 5 invalid-output, 4 empty, 4 long). The final few-shot run uses a budget of 100000 and writes `outputs/triage_results.csv`. Invalid, empty, long and budget-exhausted tickets route to `human_review` at NORMAL priority; valid fraud tickets route to `security_team` at HIGH priority; other valid premium customers route to `premium_desk` at HIGH priority; remaining valid tickets route to `support_queue` at NORMAL priority. Only the accepted outputs are rule-validated classifications; all guardrail/invalid-output cases are handed to people.

## Part D

### Policy RAG

Each of the five policy documents is split into 40-word chunks with 10-word overlap (15 total: refund 4, delivery 3, payment 3, escalation 3, privacy 2). TF-IDF retrieval uses cosine similarity and returns the top three chunks with source, score and text. Term frequency reflects how often a term occurs in a chunk; inverse document frequency downweights terms common across chunks. TF-IDF is lexical, so semantically similar questions phrased with different words can rank poorly.

| Evaluation | Expected document hit (12 answerable) | Expected fact in answer (12 answerable) | Fallback on unanswerable (2) |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the answer to “What delivery time makes a completed order eligible for a full refund?” and “How soon is money returned when a payment fails but is deducted?” breaks because the relevant fact crosses a chunk boundary. The assistant uses the supplied context only and falls back with “I could not find that information in the policy documents.” when the mock cannot find an answer.

### Refund agent and scenarios

The agent uses five tools through a `TOOLS_SPEC` and `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. Only completed orders delivered in more than 45 minutes are eligible for a full refund; orders delivered within 45 minutes are denied, and amounts above INR 1,000 are escalated. Refunds and escalations are logged to a local simulated ledger only.

| Scenario | Status | Refund / escalated amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 962 |
| S2 | REFUND | 135.00 | 5 | 962 |
| S3 | ESCALATE | 1,999.00 (human approval required) | 5 | 949 |
| S4 | DENIED | 0.00 | 4 | 657 |
| S5 | DENIED | 0.00 | 3 | 376 |
| S6 | NOT_FOUND | 0.00 | 2 | 98 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 / no refund issued | 6 | 3,078 |

All seven scenarios match their expected outcomes and refund amounts. S7 uses all six model calls and safely falls back at the iteration limit. The S1 token-limit demonstration ends with `FALLBACK_TOKEN_LIMIT` at limit 200; the zero-second timeout demonstration ends with `FALLBACK_TIMEOUT` before the first model call. The crash-and-resume demonstration crashes after its second tool observation is checkpointed, then resumes to REFUND INR 128.25. Across the crash and resume, the exact tool order is `get_order → search_policy → calculate_refund → issue_refund`; no tool is repeated. Checkpoints are written under `state/`, each scenario is logged to `outputs/agent_runs.jsonl`, and simulated issue/escalation actions are written to `outputs/refund_ledger.jsonl`.

### ReAct versus plan-and-execute

The measured ReAct run uses 5 model calls for S1 (refund), 3 for S5 (cancelled order) and 2 for S6 (unknown order); the seven scenarios range from 2 to 6 calls. A plan-and-execute design could ask the model for a complete sequence once and invoke tools without a model call after each observation, using fewer model calls and usually fewer tokens on the happy path; this design was not separately benchmarked here. ReAct responds naturally to observations: it stops when the order is missing, denies cancelled orders, and escalates high-value refunds. A one-shot plan is more efficient but needs strict validation and replanning on unexpected tool results to avoid executing stale steps. For refund handling, I prefer bounded ReAct because every tool result can change the next action and the implementation has explicit limits, checkpoints and human fallbacks.

## Limitations and Next Steps

- All data, policies, classifications and refund actions are synthetic; the mock LLM is deterministic and is not a measure of production performance.
- The retention list is illustrative, not a leakage-free evaluation, because its Premium flag comes from whole-window RFM that includes the September label window.
- The refund ledger simulates decisions only; it does not issue real payments. A production implementation would need authentication, authorization, audit controls, human approval, monitoring and tests against real reviewed data.
- Model quality, segmentation stability and policy retrieval should be validated on representative real-world data before any operational use.
