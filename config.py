"""
All settings for the project, in one flat file.

Timeline convention (used everywhere)
-------------------------------------
To predict bar t the information set is everything up to and including t-1:

    features used      : row t-1
    label predicted    : y_{t-1} = 1[ r_t > 0 ]
    position taken     : at the close of t-1
    pnl                : signal_t * r_t
    cost               : COST_BPS * |signal_t - signal_{t-1}|

So the last training row whose label is known at decision time is t-1-NUM_LEAD.
That is the only purge this design needs.
"""

# ---------------- data ----------------
TICKER = "BTC-USD"
START_DATE = "2015-01-01"
END_DATE = "2025-06-18"
SIGNAL_START = "2024-01-01"       # evaluation period begins here
NUM_LEAD = 1                      # 1-bar prediction horizon

# ---------------- features ----------------
# ADF/KPSS transform decisions are fitted once on data before this date and
# then frozen, so no future information can change how a feature is built.
STATIONARITY_FIT_END = "2020-01-01"
ADF_PVALUE = 0.05                 # ADF decides; KPSS is reported alongside
DROP_NONSTATIONARY = True         # drop features still non-stationary after transform

# ---------------- regime model (the gate) ----------------
N_STATES = 3                      # bear / middle / bull, ordered by fitted mean
HMM_FEATURES = ["returns"]        # what the HMM sees. ["returns", "vol"] also valid.
VOL_WINDOW = 20                   # look-back for the realised-vol observable

# ---------------- experts ----------------
WINDOW_SIZE = 4 * 252             # rolling training window (~4 years)
REFIT_EVERY = 5                   # refit gate + experts every k bars
MIN_SAMPLES_PER_REGIME = 50       # below this an expert is not trained
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 10
RANDOM_STATE = 100

# ---------------- signal ----------------
# P(up) = sum_s P(S_t = s) * p_s   (probability-weighted, never argmax)
SIGNAL_THRESHOLD = 0.03           # dead band: |P(up) - 0.5| must exceed this

# ---------------- costs ----------------
COST_BPS = 10.0
COST_SWEEP_BPS = [0.0, 5.0, 10.0, 20.0, 50.0]

# ---------------- reporting ----------------
PERIODS_PER_YEAR = 365            # crypto trades daily; 252 for equities
REPORT_DIR = "reports"
