import pandas as pd
import os
from stochastic_demand_opt.config import DATA_DIR

class DataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def load_calendar(self):
        df = pd.read_csv(os.path.join(self.data_dir, "calendar.csv"))
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_sales(self):
        return pd.read_csv(os.path.join(self.data_dir, "sales_train_validation.csv"))

    def load_sell_prices(self):
        return pd.read_csv(os.path.join(self.data_dir, "sell_prices.csv"))
