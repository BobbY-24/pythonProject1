# Stock Portfolio Managing Agent

## Overview
This project is an experimental stock portfolio research and backtesting workspace. It includes data fetching, local data storage, rule-based and ML-oriented agents, portfolio simulation, performance metrics, notebooks, and tests. The project is still in progress, but it has a more substantial software structure than the repository name suggests.

## Motivation
This repo is valuable as a bridge between data science, software engineering, and finance-oriented experimentation. It demonstrates modular code organization, test scaffolding, backtesting concepts, and early agent design. With continued cleanup, it could become a strong applied ML systems project.

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
- Fetch and store OHLCV stock data.
- Generate rule-based and ML-based signals.
- Simulate portfolio behavior through backtests.
- Compute performance metrics.
- Prototype API schemas and endpoints.

## Results
TODO: add metric after rerunning notebook.

## Key Insights
- The repo already has a promising modular structure.
- The project should be renamed on GitHub from `pythonProject1` to something like `stock-portfolio-managing-agent`.
- The strongest next improvement is documentation and a reproducible example run.

## Limitations
- This is not investment advice.
- Backtests can be misleading without transaction costs, walk-forward validation, and leakage checks.
- Some notebooks and data files need cleanup.
- The repository name currently undersells the project.

## Future Improvements
- Rename the repository to `stock-portfolio-managing-agent`.
- Add a minimal reproducible backtest command.
- Add architecture notes in `docs/`.
- Add tests for metrics, data loading, and portfolio simulation.
- Separate sample data from generated/cache data.

## How to Run
```bash
git clone https://github.com/BobbY-24/pythonProject1.git
cd pythonProject1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_backtest.py
```
