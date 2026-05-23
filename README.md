# Stock Portfolio Managing Agent

## Overview
I built this repository as an experimental stock portfolio research and backtesting workspace. It includes data fetching, local data storage, rule-based and ML-oriented agents, portfolio simulation, performance metrics, notebooks, and tests.

## Motivation
I use this project to connect data science, software engineering, and finance-oriented experimentation. I use it to practice modular code organization, backtesting concepts, agent design, and evaluation discipline.

## Project Structure
```text
src/
  agents/
  api/
  data/
  metrics/
  ml/
  portfolio/
notebooks/
scripts/
tests/
data/
README.md
requirements.txt
```

## Methods
- I fetch and store OHLCV stock data.
- I prototype rule-based and ML-based signals.
- I simulate portfolio behavior through backtests.
- I compute performance metrics.
- I experiment with API schemas and endpoints.

## Results
I have not finalized a single benchmark metric for this repo yet because the project is still an active engineering workspace. The current value is in the structure, tests, and backtesting pipeline rather than a polished performance claim.

## Key Insights
- The codebase already has a useful modular structure for future finance experiments.
- The next important step is a clean reproducible backtest command with fixed sample data.
- This project would be stronger under the repository name `stock-portfolio-managing-agent`.

## Limitations
- I do not intend this as investment advice.
- Backtests can be misleading without transaction costs, walk-forward validation, and leakage checks.
- Some notebooks and data files still need cleanup.
- The current repository name undersells the project.

## Future Improvements
- Rename the repository to `stock-portfolio-managing-agent`.
- Add a minimal reproducible backtest example.
- Add architecture notes in `docs/`.
- Add stronger tests for metrics, data loading, and portfolio simulation.

## How to Run
```bash
git clone https://github.com/BobbY-24/pythonProject1.git
cd pythonProject1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_backtest.py
```
