# Stochastic Demand Optimization

A production-ready pipeline for forecasting stochastic demand using LightGBM quantile regression, followed by multi-period inventory optimization using Mixed-Integer Linear Programming (MILP).

## Installation

This project uses `conda` for environment management.

```bash
conda env create -f environment.yml
conda activate stochastic-demand-opt
```

## Usage

1. **Place data:** Ensure your M5 competition datasets (or similar structures) are available in the `data/` folder.
2. **Run Forecast Pipeline:** Execute the forecasting step which loads data, trains LightGBM, and saves predictions to `preds/`:
```bash
python scripts/forecast.py
```
3. **Run Inventory Optimization:** After generating predictions, run the MILP optimization backtest:
```bash
python scripts/optimize.py
```

## Project Structure

* `src/stochastic_demand_opt`: Core modules (config, data, features, model, inventory, evaluation).
* `scripts/`: Entry point scripts (`forecast.py` and `optimize.py`).
* `data/`: Raw input data.
* `preds/`: Generated predictions and backtest summaries.
