# Polymarket Weather Autonomous Trading Bot (Phase 1 Production)

Production-grade automated trading system for Polymarket daily **Highest Temperature** bucket markets across global hubs (**London, Seoul, NYC, Beijing, Miami**).

Powered by multi-model ensemble forecasting (`ECMWF IFS`, `GFS`, `ICON`, `GEM`), calibrated with a **Student-$t$ heavy-tail distribution**, dynamic lead-time scaling, and strict fail-closed institutional risk gates.

Designed for automated execution via **GitHub Actions** or standalone daemons.

---

## 1. Architecture & Strategy Core

* **Ensemble Model**: Queries ~100+ members via Open-Meteo across NOAA, ECMWF, DWD, and CMC.
* **Student-$t$ Distribution ($\nu = 5.0$)**: Eliminates Gaussian exponential tail underestimation, correctly pricing extreme weather tails.
* **Lead-Time Scaling**: Forecast error smear dynamically expands with lead distance: $\sigma(L) = \sigma_1 \sqrt{1 + 0.65(L - 1)}$.
* **Asymmetric Leg Construction**:
  * **Convex Cheap YES Accumulation** ($p \le \$0.15$): Captures underpriced longshots with up to 20:1 payouts and bounded loss.
  * **Filtered NO Insurance** ($p \ge \$0.93$): High win rate yield protected by an asymmetric tail-safety margin ($\frac{1 - \text{price}}{\text{Student-}t \, P(YES)} \ge 1.50$).
* **Official Ground Truth**: Settles from Iowa State Mesonet ASOS archives matching exact Polymarket resolution rules:
  * **NYC**: KLGA (LaGuardia) — $F$, whole degree
  * **London**: EGLC (London City) — $C$, whole degree
  * **Miami**: KMIA (Miami Intl) — $F$, whole degree
  * **Beijing**: ZBAA (Capital Airport) — $C$, whole degree
  * **Seoul**: RKSI (Incheon Airport) settled, sampled terrestrially via Gimpo (`37.56, 126.79`)

---

## 2. Directory Layout

```
polymarket-weather-bot/
├── .github/workflows/
│   ├── weather_bot.yml      # Scheduled GitHub Actions runner (4x daily + auto-commit)
│   └── ci_selftest.yml      # CI workflow running unit tests on push/PR
├── src/
│   ├── config.py            # Calibrations, station metadata, risk constants
│   ├── fees.py              # Polymarket dynamic fee model (0.05 * p * (1-p))
│   ├── weather_model.py     # Student-t CDF, lead scaling, tail risk filter
│   ├── station_truth.py     # Iowa Mesonet & HKO ground-truth scrapers
│   ├── risk.py              # Categorical multi-bucket sizing, caps, HALT
│   ├── paper.py             # SQLite ledger, Brier scoring, T6 gatekeeper
│   ├── scanner.py           # Gamma API & CLOB book fetcher, batch grader
│   ├── executor.py          # Paper vs py-clob-client live order router
│   └── main.py              # Single CLI entry point
├── tests/
│   ├── test_model.py        # Student-t heavy tail & lead scaling tests
│   ├── test_paper.py        # 14/14 fail-closed gate tests
│   └── test_risk.py         # 14/14 risk allocation & circuit breaker tests
├── data/
│   └── paper.db             # Local SQLite state (committed back by GitHub Actions)
├── CAPITAL_STRATEGY.md      # Mathematical portfolio allocation rules
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. GitHub Actions Automated Deployment

This repository is pre-configured to run automatically in GitHub Actions on a schedule matching the publication of global weather supercomputer runs:

### Synoptic Schedule
* **03:45 UTC** (`09:15 AM IST`) — NOAA 00z model drop
* **09:45 UTC** (`03:15 PM IST`) — NOAA 06z model drop
* **16:00 UTC** (`09:30 PM IST`) — ECMWF 12z primary model drop
* **21:45 UTC** (`03:15 AM IST`) — NOAA 18z model drop

### Deployment Steps on GitHub

1. **Initialize Git & Push**:
   ```bash
   cd C:\Users\Mr\Desktop\polymarket-weather-bot
   git init
   git add .
   git commit -m "Initial production release: Polymarket weather bot Phase 1"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Configure GitHub Repository Secrets (Required for LIVE=1)**:
   Navigate to **Settings -> Secrets and variables -> Actions -> Secrets**:
   * `POLY_PRIVATE_KEY`: Your Polygon wallet private key (hex string).
   * `POLY_API_KEY`: Polymarket CLOB API Key.
   * `POLY_API_SECRET`: Polymarket CLOB API Secret.
   * `POLY_API_PASSPHRASE`: Polymarket CLOB Passphrase.

3. **Configure Repository Variables (Optional Overrides)**:
   Navigate to **Settings -> Secrets and variables -> Actions -> Variables**:
   * `LIVE`: Set to `0` for Paper Trading (default) or `1` for Live execution.
   * `FORCE_LIVE`: Set to `0` (enforces T6 50-trade gate) or `1` (manual override).
   * `MIN_EDGE_AFTER_FEE`: Default `0.04` (4¢ net edge).

4. **Enable Workflow Permissions**:
   Under **Settings -> Actions -> General -> Workflow permissions**, select:
   * **Read and write permissions** (allows the workflow to commit `data/paper.db` state updates back to git).

5. **Manual Trigger (On-Demand)**:
   You can trigger a run at any time from the **Actions** tab -> **Polymarket Weather Autonomous Trading Bot** -> **Run workflow**.

---

## 4. Local CLI Usage

Run commands locally from the project root:

```bash
# 1. Run offline verification suite (17/17 tests must pass):
python src/main.py selftest

# 2. Run market scan in paper trading mode:
python src/main.py scan

# 3. Grade settled contracts against official station truth:
python src/main.py grade

# 4. View cumulative paper performance and T6 gate status:
python src/main.py report

# 5. Run full pipeline (Grade -> Scan -> Report):
python src/main.py
```

---

## 5. Risk Safeguards & Circuit Breakers

* **T6 Paper Gatekeeper**: Blocks live order execution until $\ge 50$ paper trades, $\ge 14$ days, Model Brier < Market Brier, and positive paper P&L.
* **Correlated Market Cap**: Sizing automatically pools all buckets for the same city-day, limiting total exposure across correlated legs to **5.0% of bankroll**.
* **Single-Trade Cap**: Strictly limits any individual position to **2.0% of bankroll**.
* **Daily Drawdown Stop**: If realized trailing 24h loss exceeds **5.0%**, the engine automatically trips `data/HALT` and blocks execution.
* **Rolling Brier Circuit Breaker**: Evaluates the trailing 100 graded trades; trips `data/HALT` if model calibration degrades to market levels.

## 6. Live Data

* **Source of truth**: `data/paper.db` (SQLite, table `paper_trades`), committed to `main` by the workflow `git-auto-commit` step after each run.
* **Supabase mirror**: `src/supabase_sync.py` upserts every `paper_trades` row (on `id`) into `public.weather_trades` over PostgREST (`POST {SUPABASE_URL}/rest/v1/weather_trades?on_conflict=id`). Requires repo secrets/vars `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`; without them the sync is silently skipped.
* **Query it**:
  ```bash
  curl "$SUPABASE_URL/rest/v1/weather_trades?select=id,ts,market,side,price,pnl&order=id.desc&limit=20"     -H "apikey: $SUPABASE_KEY" -H "Authorization: Bearer $SUPABASE_KEY"
  ```
  Or locally: `sqlite3 data/paper.db "select * from paper_trades order by id desc limit 20"`.
* **Stalled trading?** Runs can succeed while placing nothing if `data/HALT` exists (the scanner prints `Trading HALTED`). Delete the file and commit once the cause (daily loss or Brier breaker) is understood.

