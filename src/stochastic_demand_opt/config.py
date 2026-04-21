import os

# Base paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
PREDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "preds")

# Feature configuration
STORE_ID = "CA_1"
DEPT_ID = "FOODS_1"
CAT_ID = "FOODS"

# Inventory Economics
UNIT_COST_FRAC = 0.70
HOLDING_COST_FRAC = 0.05
UNDERAGE_MULT = 0.428
GOODWILL_MULT = 0.10
UNIT_VOLUME_DEFAULT = 1.0

# Inventory Policy Configuration
INITIAL_INVENTORY = 0.0
USE_TOP_VOLUME_ONLY = True
TOP_VOLUME_SHARE = 0.20
USE_TOP_SHORTAGE_CAP = False
TOP_SHORTAGE_CAP_MULT = 1.00

# Constraints Calibration
CAP_CALIBRATION_SOURCE = "train_valid"
EXOGENOUS_CAP_MODE = "mean_q50"
EXOGENOUS_CAP_MULT = 1.00

# Target columns
TARGET_COL = "y_wk_1"

# Forecasting quantiles
QUANTILES = [0.1, 0.5, 0.9]

# Training / Test Split Settings
TEST_WEEKS = 26
VALID_WEEKS_FINAL = 26
CV_VALID_WEEKS = 26
N_CV_FOLDS = 3

# LGBM Grid
LGBM_PARAM_GRID = {
    "learning_rate": [0.03, 0.05],
    "n_estimators": [300, 500],
    "max_depth": [6],
    "num_leaves": [31, 63],
    "min_child_samples": [20, 50],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}
