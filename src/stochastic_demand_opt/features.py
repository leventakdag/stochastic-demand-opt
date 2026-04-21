import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, store_id, dept_id):
        self.store_id = store_id
        self.dept_id = dept_id

    def filter_and_melt(self, sales_train):
        sales_filt = sales_train[
            (sales_train["store_id"] == self.store_id) &
            (sales_train["dept_id"] == self.dept_id)
        ].copy()

        day_cols = [c for c in sales_filt.columns if c.startswith("d_")]
        df_long = sales_filt.melt(
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            value_vars=day_cols,
            var_name="d",
            value_name="sales"
        )
        return df_long

    def add_calendar(self, df_long, calendar):
        calendar_use = calendar[
            [
                "date", "wm_yr_wk", "d", "wday", "month", "year", "weekday",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                "snap_CA", "snap_TX", "snap_WI"
            ]
        ].copy()
        calendar_use["date"] = pd.to_datetime(calendar_use["date"])
        return df_long.merge(calendar_use, on="d", how="left")

    def add_prices(self, df_long, sell_prices):
        return df_long.merge(
            sell_prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left"
        )

    def aggregate_weekly(self, df_long):
        weekly = (
            df_long
            .groupby(
                ["item_id", "store_id", "dept_id", "cat_id", "state_id", "wm_yr_wk"],
                as_index=False
            )
            .agg(
                weekly_sales=("sales", "sum"),
                week_end_date=("date", "max"),
                avg_sell_price=("sell_price", "mean"),
                min_sell_price=("sell_price", "min"),
                max_sell_price=("sell_price", "max"),
                snap_CA_any=("snap_CA", "max"),
                snap_TX_any=("snap_TX", "max"),
                snap_WI_any=("snap_WI", "max"),
                month=("month", lambda x: x.iloc[-1]),
                year=("year", lambda x: x.iloc[-1]),
                weekday_last=("weekday", lambda x: x.iloc[-1]),
                event_name_1=("event_name_1", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan),
                event_type_1=("event_type_1", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan),
                event_name_2=("event_name_2", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan),
                event_type_2=("event_type_2", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan),
            )
            .sort_values(["item_id", "week_end_date"])
            .reset_index(drop=True)
        )
        return weekly.rename(columns={"week_end_date": "date"})

    def build_features(self, weekly):
        weekly["date"] = pd.to_datetime(weekly["date"])
        
        # Snap features
        if self.store_id.startswith("CA"):
            weekly["snap_any"] = weekly["snap_CA_any"]
        elif self.store_id.startswith("TX"):
            weekly["snap_any"] = weekly["snap_TX_any"]
        elif self.store_id.startswith("WI"):
            weekly["snap_any"] = weekly["snap_WI_any"]
        else:
            weekly["snap_any"] = 0

        weekly = weekly.sort_values(["item_id", "date"]).reset_index(drop=True)
        
        # Calendar features
        iso = weekly["date"].dt.isocalendar()
        weekly["week_of_year"] = iso.week.astype(int)
        weekly["iso_year"] = iso.year.astype(int)
        weekly["quarter"] = weekly["date"].dt.quarter
        weekly["month"] = weekly["date"].dt.month
        weekly["year"] = weekly["date"].dt.year
        
        weekly["week_sin"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52)
        weekly["week_cos"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52)
        weekly["week_sin_2"] = np.sin(4 * np.pi * weekly["week_of_year"] / 52)
        weekly["week_cos_2"] = np.cos(4 * np.pi * weekly["week_of_year"] / 52)
        weekly["month_sin"] = np.sin(2 * np.pi * weekly["month"] / 12)
        weekly["month_cos"] = np.cos(2 * np.pi * weekly["month"] / 12)
        
        weekly["series_week_idx"] = weekly.groupby("item_id").cumcount()
        weekly["week_mod_4"] = (weekly["series_week_idx"] % 4).astype(int)
        
        weekly["day_of_month"] = weekly["date"].dt.day
        weekly["is_month_start_window"] = (weekly["day_of_month"] <= 7).astype(int)
        weekly["is_month_end_window"] = (weekly["day_of_month"] >= 24).astype(int)
        
        weekly["is_year_start"] = (weekly["month"] == 1).astype(int)
        weekly["is_year_end"] = (weekly["month"] == 12).astype(int)
        weekly["is_q4"] = (weekly["quarter"] == 4).astype(int)
        weekly["is_holiday_season"] = weekly["month"].isin([11, 12]).astype(int)
        weekly["is_summer"] = weekly["month"].isin([6, 7, 8]).astype(int)

        # Event Features
        weekly["has_event_1"] = weekly["event_name_1"].notna().astype(int)
        weekly["has_event_2"] = weekly["event_name_2"].notna().astype(int)
        weekly["has_any_event"] = ((weekly["has_event_1"] == 1) | (weekly["has_event_2"] == 1)).astype(int)
        
        weekly["is_cultural_event"] = ((weekly["event_type_1"] == "Cultural") | (weekly["event_type_2"] == "Cultural")).astype(int)
        weekly["is_national_event"] = ((weekly["event_type_1"] == "National") | (weekly["event_type_2"] == "National")).astype(int)
        weekly["is_sporting_event"] = ((weekly["event_type_1"] == "Sporting") | (weekly["event_type_2"] == "Sporting")).astype(int)
        weekly["is_religious_event"] = ((weekly["event_type_1"] == "Religious") | (weekly["event_type_2"] == "Religious")).astype(int)
        
        event_dates = weekly.loc[weekly["has_any_event"] == 1, "date"].sort_values().unique()
        def min_abs_days_to_event(dt, event_dates):
            if len(event_dates) == 0: return np.nan
            return np.min(np.abs((pd.to_datetime(event_dates) - dt).days))
        
        unique_dates = weekly["date"].drop_duplicates().sort_values()
        date_to_event_dist = {dt: min_abs_days_to_event(dt, event_dates) for dt in unique_dates}
        weekly["days_to_nearest_event"] = weekly["date"].map(date_to_event_dist)
        weekly["weeks_to_nearest_event"] = weekly["days_to_nearest_event"] / 7.0
        weekly["near_event_1w"] = (weekly["days_to_nearest_event"] <= 7).astype(int)
        weekly["near_event_2w"] = (weekly["days_to_nearest_event"] <= 14).astype(int)
        
        # Price features
        weekly["avg_sell_price"] = weekly.groupby("item_id")["avg_sell_price"].ffill()
        weekly["min_sell_price"] = weekly.groupby("item_id")["min_sell_price"].ffill()
        weekly["max_sell_price"] = weekly.groupby("item_id")["max_sell_price"].ffill()
        
        weekly["price_lag_1"] = weekly.groupby("item_id")["avg_sell_price"].shift(1)
        weekly["price_lag_2"] = weekly.groupby("item_id")["avg_sell_price"].shift(2)
        weekly["price_change_1"] = weekly["avg_sell_price"] - weekly["price_lag_1"]
        weekly["price_pct_change_1"] = np.where(
            weekly["price_lag_1"].abs() > 1e-9,
            weekly["price_change_1"] / weekly["price_lag_1"],
            0.0
        )
        
        weekly["item_price_mean_to_t"] = weekly.groupby("item_id")["avg_sell_price"].transform(lambda s: s.expanding().mean())
        weekly["price_vs_item_mean_to_t"] = weekly["avg_sell_price"] - weekly["item_price_mean_to_t"]
        weekly["price_ratio_to_item_mean_to_t"] = np.where(
            weekly["item_price_mean_to_t"].abs() > 1e-9,
            weekly["avg_sell_price"] / weekly["item_price_mean_to_t"],
            1.0
        )
        
        # Lag features
        for lag in [1, 2, 3, 4, 6, 8, 12, 13, 26, 52]:
            weekly[f"sales_lag_{lag}"] = weekly.groupby("item_id")["weekly_sales"].shift(lag)
            
        weekly["sales_diff_1"] = weekly["sales_lag_1"] - weekly["sales_lag_2"]
        weekly["sales_diff_4"] = weekly["sales_lag_1"] - weekly["sales_lag_4"]
        weekly["sales_diff_13"] = weekly["sales_lag_1"] - weekly["sales_lag_13"]
        weekly["sales_diff_52"] = weekly["sales_lag_1"] - weekly["sales_lag_52"]
        
        # Rolling features
        weekly = self._add_group_rolling(weekly, "item_id", "weekly_sales", [4, 8, 13, 26, 52])
        weekly = self._add_group_rolling(weekly, "item_id", "avg_sell_price", [4, 8, 13, 26])
        
        # Extra
        weekly["last_vs_roll4"] = weekly["sales_lag_1"] - weekly["weekly_sales_roll_mean_4"]
        weekly["last_vs_roll13"] = weekly["sales_lag_1"] - weekly["weekly_sales_roll_mean_13"]
        weekly["last_vs_roll26"] = weekly["sales_lag_1"] - weekly["weekly_sales_roll_mean_26"]
        weekly["last_vs_roll52"] = weekly["sales_lag_1"] - weekly["weekly_sales_roll_mean_52"]
        
        zero_ind = (weekly.groupby("item_id")["weekly_sales"].shift(1) == 0).astype(float)
        for w in [4, 8, 13, 26, 52]:
            weekly[f"zero_rate_roll_{w}"] = zero_ind.groupby(weekly["item_id"]).rolling(w).mean().reset_index(level=0, drop=True)
            
        weekly["sales_to_roll4"] = np.where(weekly["weekly_sales_roll_mean_4"].abs() > 1e-9, weekly["sales_lag_1"] / weekly["weekly_sales_roll_mean_4"], 1.0)
        weekly["sales_to_roll13"] = np.where(weekly["weekly_sales_roll_mean_13"].abs() > 1e-9, weekly["sales_lag_1"] / weekly["weekly_sales_roll_mean_13"], 1.0)
        weekly["sales_to_roll52"] = np.where(weekly["weekly_sales_roll_mean_52"].abs() > 1e-9, weekly["sales_lag_1"] / weekly["weekly_sales_roll_mean_52"], 1.0)
        
        weekly["event_name_1"] = weekly["event_name_1"].fillna("None")
        weekly["event_type_1"] = weekly["event_type_1"].fillna("None")
        weekly["event_name_2"] = weekly["event_name_2"].fillna("None")
        weekly["event_type_2"] = weekly["event_type_2"].fillna("None")
        
        # Target
        weekly["target_wk_plus_1"] = weekly.groupby("item_id")["weekly_sales"].shift(-1)
        weekly["y_wk_1"] = weekly["target_wk_plus_1"]
        
        return weekly

    def _add_group_rolling(self, df, group_col, value_col, windows):
        out = df.copy()
        for w in windows:
            shifted = out.groupby(group_col)[value_col].shift(1)
            out[f"{value_col}_roll_mean_{w}"] = shifted.groupby(out[group_col]).rolling(w).mean().reset_index(level=0, drop=True)
            out[f"{value_col}_roll_std_{w}"] = shifted.groupby(out[group_col]).rolling(w).std().reset_index(level=0, drop=True)
            out[f"{value_col}_roll_min_{w}"] = shifted.groupby(out[group_col]).rolling(w).min().reset_index(level=0, drop=True)
            out[f"{value_col}_roll_max_{w}"] = shifted.groupby(out[group_col]).rolling(w).max().reset_index(level=0, drop=True)
        return out

FEATURE_COLS = [
    "item_id", "store_id", "dept_id", "cat_id", "state_id",
    "month", "year", "quarter", "week_of_year", "iso_year",
    "week_sin", "week_cos", "week_sin_2", "week_cos_2",
    "month_sin", "month_cos", "series_week_idx", "week_mod_4",
    "day_of_month", "is_month_start_window", "is_month_end_window",
    "is_year_start", "is_year_end", "is_q4", "is_holiday_season", "is_summer",
    "snap_any", "has_event_1", "has_event_2", "has_any_event",
    "is_cultural_event", "is_national_event", "is_sporting_event", "is_religious_event",
    "weeks_to_nearest_event", "near_event_1w", "near_event_2w",
    "avg_sell_price", "min_sell_price", "max_sell_price",
    "price_lag_1", "price_lag_2", "price_change_1", "price_pct_change_1",
    "price_vs_item_mean_to_t", "price_ratio_to_item_mean_to_t",
    "avg_sell_price_roll_mean_4", "avg_sell_price_roll_std_4",
    "avg_sell_price_roll_mean_8", "avg_sell_price_roll_std_8",
    "avg_sell_price_roll_mean_13", "avg_sell_price_roll_std_13",
    "avg_sell_price_roll_mean_26", "avg_sell_price_roll_std_26",
    "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_4",
    "sales_lag_6", "sales_lag_8", "sales_lag_12", "sales_lag_13",
    "sales_lag_26", "sales_lag_52",
    "sales_diff_1", "sales_diff_4", "sales_diff_13", "sales_diff_52",
    "weekly_sales_roll_mean_4", "weekly_sales_roll_std_4", "weekly_sales_roll_min_4", "weekly_sales_roll_max_4",
    "weekly_sales_roll_mean_8", "weekly_sales_roll_std_8", "weekly_sales_roll_min_8", "weekly_sales_roll_max_8",
    "weekly_sales_roll_mean_13", "weekly_sales_roll_std_13", "weekly_sales_roll_min_13", "weekly_sales_roll_max_13",
    "weekly_sales_roll_mean_26", "weekly_sales_roll_std_26", "weekly_sales_roll_min_26", "weekly_sales_roll_max_26",
    "weekly_sales_roll_mean_52", "weekly_sales_roll_std_52", "weekly_sales_roll_min_52", "weekly_sales_roll_max_52",
    "last_vs_roll4", "last_vs_roll13", "last_vs_roll26", "last_vs_roll52",
    "zero_rate_roll_4", "zero_rate_roll_8", "zero_rate_roll_13", "zero_rate_roll_26", "zero_rate_roll_52",
    "sales_to_roll4", "sales_to_roll13", "sales_to_roll52",
]
