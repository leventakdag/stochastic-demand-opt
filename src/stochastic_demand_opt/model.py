import itertools
import lightgbm as lgb
import numpy as np
import pandas as pd
from stochastic_demand_opt.config import QUANTILES, LGBM_PARAM_GRID, TARGET_COL
from stochastic_demand_opt.evaluation import pinball_loss

class QuantileForecaster:
    def __init__(self, target_col=TARGET_COL, quantiles=QUANTILES):
        self.target_col = target_col
        self.quantiles = quantiles
        self.models = {}
        self.best_params = None

    def prepare_data(self, model_df, valid_start, test_start, cat_cols):
        train_df_full = model_df[model_df["date"] < valid_start].copy()
        valid_df_final = model_df[
            (model_df["date"] >= valid_start) & (model_df["date"] < test_start)
        ].copy()
        test_df = model_df[model_df["date"] >= test_start].copy()

        for df_ in [train_df_full, valid_df_final, test_df]:
            for c in cat_cols:
                df_[c] = df_[c].astype("category")

        for c in cat_cols:
            cats = train_df_full[c].cat.categories
            valid_df_final[c] = valid_df_final[c].cat.set_categories(cats)
            test_df[c] = test_df[c].cat.set_categories(cats)
            
        return train_df_full, valid_df_final, test_df

    def cv_search(self, train_df_full, feature_cols, cat_cols, cv_valid_weeks=26, n_cv_folds=3):
        pretest_dates = sorted(train_df_full["date"].unique())
        cv_folds = []
        for i in range(n_cv_folds, 0, -1):
            fold_valid_start = pretest_dates[-i * cv_valid_weeks]
            if i == 1:
                fold_valid_end = pd.Timestamp.max # Up to end
            else:
                fold_valid_end = pretest_dates[-(i - 1) * cv_valid_weeks]

            fold_train = train_df_full[train_df_full["date"] < fold_valid_start].copy()
            fold_valid = train_df_full[
                (train_df_full["date"] >= fold_valid_start) &
                (train_df_full["date"] < fold_valid_end)
            ].copy()
            cv_folds.append((fold_train, fold_valid))

        grid_keys = list(LGBM_PARAM_GRID.keys())
        param_combos = [dict(zip(grid_keys, vals)) for vals in itertools.product(*LGBM_PARAM_GRID.values())]
        cv_results = []

        for params_candidate in param_combos:
            fold_scores = []
            for fold_train, fold_valid in cv_folds:
                fold_train = fold_train.copy()
                fold_valid = fold_valid.copy()
                for c in cat_cols:
                    fold_train[c] = fold_train[c].astype("category")
                    fold_valid[c] = fold_valid[c].astype("category")
                    cats = fold_train[c].cat.categories
                    fold_valid[c] = fold_valid[c].cat.set_categories(cats)

                X_fold_train = fold_train[feature_cols]
                y_fold_train = fold_train[self.target_col]
                X_fold_valid = fold_valid[feature_cols]
                y_fold_valid = fold_valid[self.target_col].values

                fold_pinballs = []
                for q in self.quantiles:
                    model = lgb.LGBMRegressor(
                        objective="quantile", alpha=q, verbosity=-1, n_jobs=-1, random_state=42, **params_candidate
                    )
                    model.fit(X_fold_train, y_fold_train, categorical_feature=cat_cols)
                    pred_q = np.maximum(model.predict(X_fold_valid), 0.0)
                    fold_pinballs.append(pinball_loss(y_fold_valid, pred_q, q))
                
                fold_scores.append(np.mean(fold_pinballs))
            
            cv_results.append({
                **params_candidate,
                "cv_avg_pinball_mean": float(np.mean(fold_scores)),
                "cv_avg_pinball_std": float(np.std(fold_scores)),
            })

        cv_results_df = pd.DataFrame(cv_results).sort_values(["cv_avg_pinball_mean", "cv_avg_pinball_std"])
        best_row = cv_results_df.iloc[0]
        self.best_params = {
            "learning_rate": float(best_row["learning_rate"]),
            "n_estimators": int(best_row["n_estimators"]),
            "max_depth": int(best_row["max_depth"]),
            "num_leaves": int(best_row["num_leaves"]),
            "min_child_samples": int(best_row["min_child_samples"]),
            "subsample": float(best_row["subsample"]),
            "colsample_bytree": float(best_row["colsample_bytree"]),
        }
        return self.best_params

    def train(self, X_train, y_train, cat_cols):
        if not self.best_params:
            raise ValueError("Must run cv_search before train, or set best_params manually.")
        
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q, verbosity=-1, n_jobs=-1, random_state=42, **self.best_params
            )
            model.fit(X_train, y_train, categorical_feature=cat_cols)
            self.models[q] = model

    def predict(self, df, feature_cols):
        preds = df[["date", "item_id", "weekly_sales", self.target_col, "sales_lag_1"]].copy()
        preds = preds.rename(columns={self.target_col: "y_true", "sales_lag_1": "y_pred_naive"})
        X = df[feature_cols]
        for q, model in self.models.items():
            preds[f"y_pred_q_{q}"] = np.maximum(model.predict(X), 0.0)
        return preds
