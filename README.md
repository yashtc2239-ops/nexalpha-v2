# NEXALPHA — ML-Driven Stock Signal Engine for NSE Equities

NEXALPHA analyzes Indian (NSE) stocks using technical indicators + machine learning,
produces buy/hold/sell signals with model explainability (SHAP), and backtests those
signals out-of-sample. It started as a college group project; this is a from-scratch
production rebuild after a code review surfaced critical correctness bugs
(see [What Changed](#what-changed-from-v1) below — written deliberately, not hidden).

## Architecture

```
yfinance (OHLCV) → indicators (ta-lib, 18 features) → feature engineering
   → RandomForest (regressor + classifier) → prediction + SHAP explanation
   → out-of-sample backtest → PSI drift monitor
```

```
backend/
├── app/
│   ├── api/         # Flask routes — thin, no business logic
│   ├── core/        # data_loader, indicators, features (pure functions)
│   ├── models/       # train, predict, drift detection
│   ├── services/     # backtest, explainability
│   ├── cache.py       # Redis cache-aside, in-memory fallback
│   ├── db.py          # SQLite run history
│   └── config.py      # every tunable constant, env-overridable
├── tests/             # pytest, network-free (synthetic fixtures)
├── Dockerfile
└── requirements.txt
frontend/index.html      # single-file Chart.js dashboard
docker-compose.yml        # backend + redis + mlflow
.github/workflows/ci.yml  # lint → test → docker build → security scan
```

## Why these technical choices

- **RandomForest (reg + clf), not deep learning**: tabular data with <20 features and
  a few thousand rows per ticker — RF is well-suited, harder to overfit than a neural
  net, and gives exact SHAP values via TreeExplainer. Documented as a deliberate
  choice for an interview, not a default.
- **PSI for drift detection**: a single comparable number per feature, cheap to
  compute, standard in industry model-monitoring (vs. eyeballing distribution plots).
- **Redis cache-aside with in-memory fallback**: reads dominate writes here (EOD data
  changes once/day); if Redis is unreachable the app degrades to in-process caching
  instead of crashing — a graceful-degradation pattern.
- **Walk-forward / out-of-sample backtesting**: signals are only backtested on the
  held-out test split the model never trained on. Backtesting on training data is the
  single most common way a quant project's numbers look great and mean nothing.

## What changed from v1

The original group project had real, identifiable bugs. Calling them out explicitly:

| Issue | v1 (legacy) | v2 (this repo) |
|---|---|---|
| Explainability | `shap_explain()` returned a **hardcoded dummy dict**, unrelated to the model | Real `shap.TreeExplainer` computation on actual model + data |
| "Ensemble" | `rf_pred`, `xgb_pred`, `ensemble_pred` were **the same number** — no XGBoost was ever trained | Single honestly-labeled RandomForest prediction; XGBoost listed as roadmap, not shipped |
| Backtest leakage | `backtest.py` fed `future_ret`/`future_dir` (the answer) back into the model as features, then silently degraded to zero trades on a shape mismatch | `select_features()` enforces the exact trained feature set; backtest runs only on out-of-sample rows |
| Model staleness | Models keyed only by ticker — a 7-day-horizon model was silently reused for a 30-day request | Models keyed by `(ticker, horizon)`, with a `MODEL_MAX_AGE_DAYS` freshness check |
| Repo hygiene | Class notes (`MYSQL_notes.txt`, DBMS assignment) and binary `.pkl`/`.db` files committed to git | Clean repo, `.gitignore` for all generated artifacts |
| Logging | `print()` statements | Structured JSON logs via loguru |
| Config | Magic numbers scattered across files | Single `config.py`, env-var overridable |
| Tests | None | pytest suite, including a regression test that asserts no leakage column ever reaches the feature matrix |
| Data sufficiency | No minimum-lookback check — a request without explicit dates got yfinance's ~20-day default, and `ta`'s ADX/SMA-200 indicators crashed with a raw `IndexError` on too little data | `MIN_LOOKBACK_DAYS=730` default + explicit `ValueError` with an actionable message if data is still insufficient |
| Drift monitor always "critical" | A `STATIONARY_FEATURES` allowlist was defined but never wired in — PSI ran on non-stationary price levels (`close`, `sma_200`) and fired "critical" on every call, on every feature | `detect_drift()` now filters to `STATIONARY_FEATURES` by default — only bounded oscillators (RSI, MACD, ADX, etc.) are monitored |
| Drift monitor false positives | Live-data comparison window was 30 rows — too small for a reliable 10-bucket PSI estimate; proven by computing PSI on two samples from the *same* distribution and watching it swing from 0.18 to 1.38 | `DRIFT_LIVE_WINDOW=90`, confirmed stable (<0.15) on identical distributions across repeated trials |
| Degenerate classifier on imbalanced tickers | Found via testing across 4 NSE tickers: `TCS.NS` gave accuracy=61% but **F1=0.000 and 0 backtest trades** — the classifier had collapsed to always predicting the majority class, making the "good" accuracy number meaningless | `class_weight="balanced"` added to `RandomForestClassifier`; training class balance is now logged and saved to model metadata so a degenerate model can't hide behind a high accuracy number |

## API

```
POST /api/analyze     { "ticker": "RELIANCE.NS", "horizon_days": 7 }
POST /api/backtest    { "ticker": "RELIANCE.NS", "horizon_days": 7 }
GET  /api/drift/<ticker>?horizon_days=7
GET  /api/health
```

## Running locally

```bash
docker-compose up --build
# backend: http://localhost:5000  |  mlflow ui: http://localhost:5001
```

```bash
cd backend && pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
python run.py
```

## Measured performance (real, not illustrative)

Evaluated across the same 4 tickers the v1 paper used (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`,
`HDFCBANK.NS`), at horizons of 5/7/10 days, using `scripts/evaluate_multiple_tickers.py`:

| Metric | Value |
|---|---|
| Mean classification accuracy (12 runs) | **56.4%** (std dev 6.3%) |
| Mean F1 score | **0.362** |
| Accuracy range | 48.3% – 72.4% |

**This does not match the v1 paper's reported 72-76% test accuracy / 0.78 AUC.** See
"What changed from v1" above and [`PUBLICATIONS.md`](./PUBLICATIONS.md) for why.

A more specific pattern emerged: **`RELIANCE.NS` and `INFY.NS` show a consistent, real
directional signal (F1 0.57-0.71 across all horizons)**, while **`TCS.NS` and `HDFCBANK.NS`
degenerate into a trivial always-predict-majority-class model at 7+ day horizons (F1=0.000)**
even after adding `class_weight="balanced"` to address class imbalance. This is treated as
a genuine finding, not a bug to silently fix: large, lower-volatility blue-chip stocks
appear to carry less learnable short-term directional signal in this feature set than
more volatile/momentum-driven names. Full per-ticker results: `scripts/evaluation_results.csv`.

## Known limitations (honest, not hidden)

- RandomForest only — no LSTM/XGBoost ensemble yet (listed as roadmap, not claimed as shipped).
- No real-time tick data — daily granularity (yfinance EOD).
- Single-asset models — no cross-sectional/portfolio-level signal yet.
- Backtest ignores Indian market-specific costs like STT and exchange transaction charges beyond a flat commission.

## Measured accuracy (real data, not a claim)

Ran `backend/scripts/evaluate_multiple_tickers.py` across the same 4 tickers cited in the
related paper, at 3 horizons each (12 runs total, June 2026):

| Ticker | h=5d | h=7d | h=10d |
|---|---|---|---|
| RELIANCE.NS | acc 57.6%, F1 0.706 | acc 55.9%, F1 0.683 | acc 48.3%, F1 0.571 |
| TCS.NS | acc 52.5%, F1 0.364 | acc 57.6%, **F1 0.000** | acc 72.4%, **F1 0.000** |
| INFY.NS | acc 52.5%, F1 0.600 | acc 54.2%, F1 0.609 | acc 53.5%, F1 0.667 |
| HDFCBANK.NS | acc 59.3%, F1 0.143 | acc 50.9%, **F1 0.000** | acc 62.1%, **F1 0.000** |

**Mean accuracy: 56.4% (σ=6.3%)** — well below the 72-76% reported in the related v1
paper (see `PUBLICATIONS.md`), consistent with a leakage-corrected, genuinely
out-of-sample evaluation.

**Residual issue found, not fully fixed**: `class_weight="balanced"` was added to the
classifier after discovering degenerate (always-majority-class) predictions during this
same testing. It fixed the issue for RELIANCE.NS and INFY.NS at every horizon, but
**TCS.NS and HDFCBANK.NS still collapse to F1=0.000 at the 7-day and 10-day horizons** —
the "high" 72.4% accuracy for TCS@10d is from a model that never predicts "up" at all,
not a genuinely skilled one. This is left as a documented, known limitation rather than
papered over: a real fix would need per-ticker threshold tuning or a precision-recall-based
decision boundary (the original paper's own PR-curve analysis is the right direction here),
not just a global class-weight adjustment.

**Also worth noting**: all 12 runs use a single chronological train/test split (most recent
~20% of days). A short, single test window can land entirely within one trend regime —
e.g. TCS@10d's test slice may have had almost no "up" days at all, which alone explains an
F1 of exactly 0 independent of the class-weight fix. A more rigorous evaluation would use
walk-forward validation (multiple rolling time windows) rather than one fixed split — listed
here as a known evaluation-methodology limitation, not silently worked around.

## Disclaimer

Educational/portfolio project. Not investment advice. Past backtest performance does
not predict future returns.

## Related publication

The earlier v1 architecture (technical indicators + Random Forest regression/classification
+ SHAP + backtesting) was described in a co-authored conference paper. The published
paper reports preliminary v1 results (test accuracy ~72-76%, AUC ~0.78, backtest
outperformance vs. buy-and-hold of 12-18%) and explicitly discloses that the "ensemble"
prediction used a RandomForest placeholder in place of the unimplemented XGBoost component.

**Important note on reproducibility**: the v1 backtest module that produced the published
backtest figures had a data-leakage bug (see the changelog above) — the target columns
(`future_ret`/`future_dir`) were inadvertently included as model features during backtesting.
This repo's v2 backtest is leakage-corrected and evaluates strictly on an out-of-sample
test split, so its numbers are expected to be more conservative than the published figures,
and should be treated as the trustworthy ones going forward. The classification accuracy/AUC
figures in the paper came from a standard train/test split (not the backtest module) and are
less directly affected, but have not been independently re-verified against v2.

This discrepancy — and catching it — is itself part of the v1 → v2 engineering story documented
in this README. Full citation details: [`PUBLICATIONS.md`](./PUBLICATIONS.md).
