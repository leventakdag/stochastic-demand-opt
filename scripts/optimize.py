import os
import sys
import pandas as pd
import numpy as np

# Add the src directory to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from stochastic_demand_opt.config import (
    PREDS_DIR, USE_TOP_VOLUME_ONLY, USE_TOP_SHORTAGE_CAP, INITIAL_INVENTORY,
    UNIT_COST_FRAC, HOLDING_COST_FRAC, UNDERAGE_MULT, GOODWILL_MULT, UNIT_VOLUME_DEFAULT
)
from stochastic_demand_opt.inventory import InventoryOptimizer

def main():
    print("Loading predictions from /preds...")
    valid_pred_path = os.path.join(PREDS_DIR, "valid_predictions.csv")
    test_pred_path = os.path.join(PREDS_DIR, "test_predictions.csv")

    if not os.path.exists(valid_pred_path) or not os.path.exists(test_pred_path):
        print("Predictions not found. Please run scripts/forecast.py first.")
        return

    valid_pred = pd.read_csv(valid_pred_path)
    test_pred = pd.read_csv(test_pred_path)

    # Parse dates if necessary
    for df in [valid_pred, test_pred]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    print("Running inventory optimization backtest...")
    
    opt_df = test_pred.copy()
    opt_df = opt_df.sort_values(["item_id", "date"]).reset_index(drop=True)
    
    # Economics configuration
    opt_df["unit_cost"] = UNIT_COST_FRAC * opt_df["avg_sell_price"]
    opt_df["unit_volume"] = UNIT_VOLUME_DEFAULT
    opt_df["overage_cost"] = HOLDING_COST_FRAC * opt_df["unit_cost"]
    opt_df["underage_cost"] = (UNDERAGE_MULT + GOODWILL_MULT) * opt_df["unit_cost"]

    if USE_TOP_VOLUME_ONLY:
        opt_df = opt_df[opt_df["is_top_volume"] == 1].copy()

    opt_df = opt_df.sort_values(["date", "item_id"]).reset_index(drop=True)

    # Build reference panel using validation data (as proxy for pre-test calibration)
    optimizer = InventoryOptimizer()
    
    # We reconstruct cap_price_source from valid_pred since we saved avg_sell_price there
    top_volume_items = set(valid_pred.loc[valid_pred["is_top_volume"] == 1, "item_id"])
    cap_ref_df = optimizer.build_cap_reference_panel(
        valid_pred, valid_pred, top_volume_items, USE_TOP_VOLUME_ONLY
    )
    
    fixed_budget_cap, fixed_storage_cap, _ = optimizer.compute_fixed_caps_from_reference(cap_ref_df)
    print(f"Fixed Caps - Budget: {fixed_budget_cap:.2f}, Storage: {fixed_storage_cap:.2f}")

    weekly_rows = []
    inventory_state_opt = {item: float(INITIAL_INVENTORY) for item in opt_df["item_id"].unique()}

    for decision_date, week_df in opt_df.groupby("date", observed=False):
        week_df = week_df.copy().reset_index(drop=True)
        week_df["inventory_begin"] = week_df["item_id"].map(inventory_state_opt).astype(float)
        
        # Optimize using MILP
        res, sol_df = optimizer.solve_newsvendor_reorder_milp(
            week_df, fixed_budget_cap, fixed_storage_cap, USE_TOP_SHORTAGE_CAP, top_shortage_cap=None
        )
        
        if res and res.success:
            eval_opt = optimizer.evaluate_reorder_plan(sol_df.copy(), "q_opt", "opt", round_qty=False)
            
            # Update inventory for the next period
            for _, row in eval_opt.iterrows():
                inventory_state_opt[row["item_id"]] = float(row["leftover_realized_opt"])
            
            weekly_rows.append({
                "date": decision_date,
                "realized_cost": eval_opt["realized_cost_opt"].sum(),
                "spend": eval_opt["spend_opt"].sum()
            })
        else:
            print(f"MILP failed for {decision_date}")
            weekly_rows.append({
                "date": decision_date,
                "realized_cost": np.nan,
                "spend": np.nan
            })

    backtest_results = pd.DataFrame(weekly_rows)
    print("\nBacktest Summary:")
    print(backtest_results.describe())
    backtest_results.to_csv(os.path.join(PREDS_DIR, "backtest_summary.csv"), index=False)
    print(f"Backtest summary saved to {PREDS_DIR}/backtest_summary.csv")

if __name__ == "__main__":
    main()
