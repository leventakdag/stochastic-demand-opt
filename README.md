# stochastic-demand-opt

This repository contains a production-ready pipeline for forecasting stochastic demand using LightGBM quantile regression, followed by multi-period inventory optimization using Mixed-Integer Linear Programming (MILP).

## Project Summary

- Forecast target: 1-week-ahead quantile forecasts (10th, 50th, 90th percentiles) of weekly unit sales.
- Scope: Retail demand at the store-department level using the M5 Forecasting dataset.
- Optimization: Multi-period Newsvendor inventory backtest using MILP to minimize expected holding and shortage costs under budget and storage constraints.
- Evaluation design: Expanding window cross-validation for hyperparameter tuning using Pinball Loss, and an out-of-sample backtest.
- Model families used: LightGBM (Quantile Regression).
- Optimization tools: SciPy (`highs` method for MILP).

## Repository Structure

```text
.
|-- pyproject.toml
|-- environment.yml
|-- README.md
|-- scripts/
|   |-- forecast.py
|   `-- optimize.py
|-- src/
|   `-- stochastic_demand_opt/
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- model.py
|       |-- evaluation.py
|       `-- inventory.py
|-- data/
|   `-- README.md
`-- preds/
    `-- README.md
```

## Package Design

- `data.py`: M5 dataset loading and validation.
- `features.py`: Temporal melting, weekly aggregation, rolling features, calendar/event extraction, and price engineering.
- `model.py`: Object-oriented LightGBM wrappers for cross-validated quantile regression.
- `evaluation.py`: Pinball loss, interval coverage, MAE, and forecast summarization metrics.
- `inventory.py`: Formulates and solves the constrained multi-period Newsvendor MILP.
- `scripts/`: Thin entry points for running the forecasting and optimization pipelines.

## Workflow Entry Points

### Run the forecasting pipeline

```bash
python scripts/forecast.py
```

### Run the inventory optimization

```bash
python scripts/optimize.py
```

## Mathematical Model

The multi-period inventory backtest optimizes the reorder quantities $q_i$ for each item $i$ to minimize expected shortage and overage (holding) costs, subject to budget and storage constraints.

For each period, the Mixed-Integer Linear Program (MILP) is defined as:

**Objective:**

$$
\min_{q, \text{over}, \text{short}} \sum_{i=1}^{n} \sum_{s=1}^{S} P(s) \Big( h_i \cdot \text{over}_{i,s} + p_i \cdot \text{short}_{i,s} \Big)
$$

**Subject to:**

**1. Scenario Balance:**

$$
\text{over}_{i,s} \ge I_i + q_i - d_{i,s} \quad \forall i, \forall s
$$

$$
\text{short}_{i,s} \ge d_{i,s} - (I_i + q_i) \quad \forall i, \forall s
$$

**2. Budget Constraint:**

$$
\sum_{i=1}^{n} c_i \cdot q_i \le B
$$

**3. Storage Constraint:**

$$
\sum_{i=1}^{n} v_i \cdot (I_i + q_i) \le C
$$

**4. Non-negativity & Integrality:**

$$
q_i \in \mathbb{Z}^+, \quad \text{over}_{i,s} \ge 0, \quad \text{short}_{i,s} \ge 0
$$

Where:
- $S$: Scenarios derived from quantile forecasts (e.g. 10th, 50th, 90th percentiles).
- $P(s)$: Probability weight of scenario $s$.
- $I_i$: Beginning inventory for item $i$.
- $d_{i,s}$: Forecasted demand for item $i$ under scenario $s$.
- $h_i, p_i$: Unit holding and shortage costs.
- $c_i, v_i$: Unit purchase cost and volume footprint.
- $B, C$: Budget and storage capacity caps.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate stochastic-demand-opt
```

## Data Notes

Raw input datasets are not included in the repository. The expected files are documented in `data/README.md`.

To rerun the project locally, place the following files inside the configured data directory:

- `calendar.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`

Sources of data: 
- `https://www.kaggle.com/c/m5-forecasting-accuracy`
