import numpy as np
import pandas as pd
from scipy.optimize import linprog
from stochastic_demand_opt.config import (
    UNIT_COST_FRAC, HOLDING_COST_FRAC, UNDERAGE_MULT, GOODWILL_MULT, 
    UNIT_VOLUME_DEFAULT, MIN_ORDER_QTY_TOP, MIN_ORDER_QTY_ALL, EXOGENOUS_CAP_MODE, EXOGENOUS_CAP_MULT
)

class InventoryOptimizer:
    def __init__(self, scenario_probs=np.array([0.20, 0.60, 0.20]), scenario_cols=["y_pred_q_0.1", "y_pred_q_0.5", "y_pred_q_0.9"]):
        self.scenario_probs = scenario_probs
        self.scenario_cols = scenario_cols

    def solve_newsvendor_reorder_milp(self, week_df, budget_B, storage_C, use_top_shortage_cap=True, top_shortage_cap=None):
        df = week_df.reset_index(drop=True).copy()
        n = len(df)
        S = len(self.scenario_probs)

        n_q = n
        n_over = n * S
        n_short = n * S
        n_vars = n_q + n_over + n_short

        q_start = 0
        over_start = n_q
        short_start = n_q + n_over

        def q_idx(i): return q_start + i
        def over_idx(i, s): return over_start + i * S + s
        def short_idx(i, s): return short_start + i * S + s

        c = np.zeros(n_vars, dtype=float)
        for i in range(n):
            h_i = float(df.loc[i, "overage_cost"])
            p_i = float(df.loc[i, "underage_cost"])
            for s in range(S):
                prob_s = self.scenario_probs[s]
                c[over_idx(i, s)] = prob_s * h_i
                c[short_idx(i, s)] = prob_s * p_i

        A_ub, b_ub = [], []

        row = np.zeros(n_vars, dtype=float)
        for i in range(n): row[q_idx(i)] = float(df.loc[i, "unit_cost"])
        A_ub.append(row)
        b_ub.append(float(budget_B))

        row = np.zeros(n_vars, dtype=float)
        rhs_storage = float(storage_C)
        for i in range(n):
            row[q_idx(i)] = float(df.loc[i, "unit_volume"])
            rhs_storage -= float(df.loc[i, "inventory_begin"]) * float(df.loc[i, "unit_volume"])
        A_ub.append(row)
        b_ub.append(float(rhs_storage))

        if use_top_shortage_cap and (top_shortage_cap is not None):
            row = np.zeros(n_vars, dtype=float)
            for i in range(n):
                if bool(df.loc[i, "is_top_volume"]):
                    for s in range(S): row[short_idx(i, s)] = self.scenario_probs[s]
            A_ub.append(row)
            b_ub.append(float(top_shortage_cap))

        for i in range(n):
            inv_i = float(df.loc[i, "inventory_begin"])
            for s, scen_col in enumerate(self.scenario_cols):
                d_is = float(df.loc[i, scen_col])

                row = np.zeros(n_vars, dtype=float)
                row[q_idx(i)] = 1.0; row[over_idx(i, s)] = -1.0
                A_ub.append(row); b_ub.append(float(d_is - inv_i))

                row = np.zeros(n_vars, dtype=float)
                row[q_idx(i)] = -1.0; row[short_idx(i, s)] = -1.0
                A_ub.append(row); b_ub.append(float(inv_i - d_is))

        A_ub = np.vstack(A_ub)
        b_ub = np.asarray(b_ub, dtype=float)

        bounds = [(0.0, None)] * n_vars
        integrality = np.zeros(n_vars, dtype=int)

        for i in range(n):
            integrality[q_idx(i)] = 1
            min_qty = MIN_ORDER_QTY_TOP if df.loc[i, "is_top_volume"] else MIN_ORDER_QTY_ALL
            bounds[q_idx(i)] = (float(min_qty), None)

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality, method="highs")

        if not res.success: return res, None

        x = res.x
        q_opt = x[q_start:q_start + n_q]

        exp_short, exp_left, exp_cost = np.zeros(n, dtype=float), np.zeros(n, dtype=float), np.zeros(n, dtype=float)

        for i in range(n):
            inv_i = float(df.loc[i, "inventory_begin"])
            h_i = float(df.loc[i, "overage_cost"])
            p_i = float(df.loc[i, "underage_cost"])
            q_i = float(q_opt[i])

            for s, scen_col in enumerate(self.scenario_cols):
                d_is = float(df.loc[i, scen_col])
                prob_s = self.scenario_probs[s]

                available = inv_i + q_i
                leftover = max(available - d_is, 0.0)
                shortage = max(d_is - available, 0.0)

                exp_left[i] += prob_s * leftover
                exp_short[i] += prob_s * shortage
                exp_cost[i] += prob_s * (h_i * leftover + p_i * shortage)

        sol_df = df.copy()
        sol_df["q_opt"] = q_opt
        sol_df["expected_shortage_opt"] = exp_short
        sol_df["expected_leftover_opt"] = exp_left
        sol_df["expected_total_cost_opt"] = exp_cost

        return res, sol_df

    def evaluate_reorder_plan(self, df, q_col, label, round_qty=True):
        out = df.copy()
        if round_qty: out[q_col] = np.round(out[q_col])

        out[f"available_{label}"] = out["inventory_begin"] + out[q_col]
        out[f"spend_{label}"] = out[q_col] * out["unit_cost"]
        out[f"storage_{label}"] = out[f"available_{label}"] * out["unit_volume"]

        out[f"sales_realized_{label}"] = np.minimum(out[f"available_{label}"], out["y_true"])
        out[f"shortage_realized_{label}"] = np.maximum(out["y_true"] - out[f"available_{label}"], 0.0)
        out[f"leftover_realized_{label}"] = np.maximum(out[f"available_{label}"] - out["y_true"], 0.0)

        out[f"realized_cost_{label}"] = (out["underage_cost"] * out[f"shortage_realized_{label}"] + out["overage_cost"] * out[f"leftover_realized_{label}"])

        exp_short, exp_left, exp_cost = np.zeros(len(out), dtype=float), np.zeros(len(out), dtype=float), np.zeros(len(out), dtype=float)
        inv = out["inventory_begin"].values.astype(float)

        for s, scen_col in enumerate(self.scenario_cols):
            d = out[scen_col].values.astype(float)
            q = out[q_col].values.astype(float)
            prob_s = self.scenario_probs[s]

            available = inv + q
            short_s = np.maximum(d - available, 0.0)
            left_s = np.maximum(available - d, 0.0)

            exp_short += prob_s * short_s
            exp_left += prob_s * left_s
            exp_cost += prob_s * (out["underage_cost"].values * short_s + out["overage_cost"].values * left_s)

        out[f"expected_shortage_{label}"] = exp_short
        out[f"expected_leftover_{label}"] = exp_left
        out[f"expected_total_cost_{label}"] = exp_cost

        return out

    def compute_fixed_caps_from_reference(self, ref_df):
        weekly_budget_ref = ref_df.groupby("date", observed=False).apply(lambda g: float((g["q50_ref"] * g["unit_cost"]).sum())).rename("ref_budget").reset_index()
        weekly_storage_ref = ref_df.groupby("date", observed=False).apply(lambda g: float((g["q50_ref"] * g["unit_volume"]).sum())).rename("ref_storage").reset_index()
        weekly_cap_ref = weekly_budget_ref.merge(weekly_storage_ref, on="date", how="inner")

        if EXOGENOUS_CAP_MODE == "mean_q50":
            fixed_budget_cap = EXOGENOUS_CAP_MULT * float(weekly_cap_ref["ref_budget"].mean())
            fixed_storage_cap = EXOGENOUS_CAP_MULT * float(weekly_cap_ref["ref_storage"].mean())
        elif EXOGENOUS_CAP_MODE == "median_q50":
            fixed_budget_cap = EXOGENOUS_CAP_MULT * float(weekly_cap_ref["ref_budget"].median())
            fixed_storage_cap = EXOGENOUS_CAP_MULT * float(weekly_cap_ref["ref_storage"].median())
        else:
            raise ValueError("EXOGENOUS_CAP_MODE must be one of: mean_q50, median_q50, custom")

        return fixed_budget_cap, fixed_storage_cap, weekly_cap_ref

    def build_cap_reference_panel(self, pred_df, price_source_df, top_volume_items, use_top_volume_only):
        ref_df = pred_df.copy()
        ref_price_map = price_source_df[["date", "item_id", "avg_sell_price"]].drop_duplicates().copy()
        ref_df = ref_df.merge(ref_price_map, on=["date", "item_id"], how="left")
        ref_df = ref_df.sort_values(["item_id", "date"]).reset_index(drop=True)
        ref_df["avg_sell_price"] = ref_df.groupby("item_id")["avg_sell_price"].ffill()

        ref_df["unit_cost"] = UNIT_COST_FRAC * ref_df["avg_sell_price"]
        ref_df["unit_volume"] = UNIT_VOLUME_DEFAULT
        ref_df["is_top_volume"] = ref_df["item_id"].isin(top_volume_items).astype(int)

        if use_top_volume_only:
            ref_df = ref_df[ref_df["is_top_volume"] == 1].copy()

        ref_df = ref_df.dropna(subset=["avg_sell_price", "unit_cost"]).copy()
        ref_df["q50_ref"] = ref_df["y_pred_q_0.5"].clip(lower=0.0)

        return ref_df
