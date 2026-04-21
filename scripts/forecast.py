import os
import sys
import pandas as pd
import numpy as np

# Add the src directory to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from stochastic_demand_opt.config import (
    STORE_ID, DEPT_ID, TEST_WEEKS, VALID_WEEKS_FINAL, CV_VALID_WEEKS, N_CV_FOLDS, TARGET_COL,
    TOP_VOLUME_SHARE, PREDS_DIR
)
from stochastic_demand_opt.data import DataLoader
from stochastic_demand_opt.features import FeatureEngineer, FEATURE_COLS
from stochastic_demand_opt.model import QuantileForecaster
from stochastic_demand_opt.evaluation import summarize_predictions

def main():
    print("Loading data...")
    data_loader = DataLoader()
    try:
        calendar = data_loader.load_calendar()
        sales_train = data_loader.load_sales()
        sell_prices = data_loader.load_sell_prices()
    except Exception as e:
        print(f"Error loading data: {e}. Please ensure data exists at the configured DATA_DIR.")
        return

    print("Engineering features...")
    fe = FeatureEngineer(STORE_ID, DEPT_ID)
    df_long = fe.filter_and_melt(sales_train)
    df_long = fe.add_calendar(df_long, calendar)
    df_long = fe.add_prices(df_long, sell_prices)
    weekly = fe.aggregate_weekly(df_long)
    weekly = fe.build_features(weekly)
    
    model_df = weekly.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    model_df = model_df.sort_values(["date", "item_id"]).reset_index(drop=True)

    print(f"Model data shape: {model_df.shape}")

    all_dates = sorted(model_df["date"].unique())
    if len(all_dates) < TEST_WEEKS + VALID_WEEKS_FINAL:
        print("Not enough dates for the split.")
        return

    test_start = all_dates[-TEST_WEEKS]
    valid_start = all_dates[-(TEST_WEEKS + VALID_WEEKS_FINAL)]

    cat_cols = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]

    forecaster = QuantileForecaster()
    train_df_full, valid_df_final, test_df = forecaster.prepare_data(
        model_df, valid_start, test_start, cat_cols
    )

    print("Running CV search...")
    try:
        best_params = forecaster.cv_search(
            train_df_full, FEATURE_COLS, cat_cols, cv_valid_weeks=CV_VALID_WEEKS, n_cv_folds=N_CV_FOLDS
        )
        print(f"Best params: {best_params}")
    except ValueError as e:
        print(f"CV search skipped or failed: {e}")
        # Default fallback
        forecaster.best_params = {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8}

    print("Training final models...")
    X_train_full = train_df_full[FEATURE_COLS]
    y_train_full = train_df_full[TARGET_COL]
    forecaster.train(X_train_full, y_train_full, cat_cols)

    print("Making predictions...")
    valid_pred = forecaster.predict(valid_df_final, FEATURE_COLS)
    test_pred = forecaster.predict(test_df, FEATURE_COLS)

    summary = pd.DataFrame([
        summarize_predictions(valid_pred, "valid_final"),
        summarize_predictions(test_pred, "test_final"),
    ])
    print("\nPrediction summary:")
    print(summary)

    # Determine top volume items for later optimization
    train_item_rank = train_df_full.groupby("item_id", observed=False)[TARGET_COL].mean().sort_values(ascending=False).reset_index(name="avg_train_target")
    n_keep = int(np.ceil(TOP_VOLUME_SHARE * len(train_item_rank)))
    top_volume_items = set(train_item_rank.head(n_keep)["item_id"])

    # Append price and top volume flags so optimizer can easily load them
    valid_pred["avg_sell_price"] = valid_df_final["avg_sell_price"].values
    test_pred["avg_sell_price"] = test_df["avg_sell_price"].values

    valid_pred["is_top_volume"] = valid_pred["item_id"].isin(top_volume_items).astype(int)
    test_pred["is_top_volume"] = test_pred["item_id"].isin(top_volume_items).astype(int)

    # Save to preds
    os.makedirs(PREDS_DIR, exist_ok=True)
    valid_pred.to_csv(os.path.join(PREDS_DIR, "valid_predictions.csv"), index=False)
    test_pred.to_csv(os.path.join(PREDS_DIR, "test_predictions.csv"), index=False)
    summary.to_csv(os.path.join(PREDS_DIR, "prediction_summary.csv"), index=False)
    print(f"Predictions saved to {PREDS_DIR}/")

if __name__ == "__main__":
    main()
