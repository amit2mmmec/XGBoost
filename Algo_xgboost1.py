import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

print("Loading dataset...")
df = pd.read_csv('NIFTY_VIX_Minute_By_Minute_Engineered.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 1. Prevent Non-Stationary Leakage (Drop raw prices and raw EMAs)
drop_cols = [
    'open', 'high', 'low', 'close', 'Target', 
    'EMA20_5min', 'EMA50_5min', 'EMA20_15min', 
    'EMA50_15min', 'EMA20_60min', 'EMA50_60min'
]
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df['Target']

# 2. Setup XGBoost and Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)

# 3. Collect Out-Of-Sample Predictions
print("Generating Out-of-Sample Predictions over 10 Years...")
all_test_dates = []
all_predictions = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    all_test_dates.extend(X_test.index)
    all_predictions.extend(preds)

pred_df = pd.DataFrame({'Signal': all_predictions}, index=all_test_dates)
# Merge the original High, Low, Close, and ATR to simulate the actual trade exits
merged_df = pred_df.join(df[['close', 'high', 'low', 'ATR14_5min']])

# 4. The Execution Simulator (Preventing Overtrading / Clustering)
print("Simulating Real Market Execution (Entering and Exiting Trades)...")
dates = merged_df.index
signals = merged_df['Signal'].values
close_prices = merged_df['close'].values
high_prices = merged_df['high'].values
low_prices = merged_df['low'].values
atrs = merged_df['ATR14_5min'].values

executed_trades = []
in_trade = False
trade_tp = 0
trade_sl = 0
timeout_bar = 0

for i in range(len(merged_df)):
    current_time = dates[i]
    
    # Check if we need to exit an active trade
    if in_trade:
        # Did we hit TP or SL? Or did 30 minutes pass?
        if high_prices[i] >= trade_tp or low_prices[i] <= trade_sl or i >= timeout_bar:
            in_trade = False # Trade closed! Capital is free again.
        continue # Cannot take new signals while in a trade
        
    # If we are NOT in a trade, check for a Buy signal
    if signals[i] == 1.0 and not np.isnan(atrs[i]):
        # Execute Buy!
        in_trade = True
        trade_tp = close_prices[i] + (1.5 * atrs[i])
        trade_sl = close_prices[i] - (1.0 * atrs[i])
        timeout_bar = i + 30 # Maximum 30 minutes in the trade
        
        executed_trades.append(current_time)

# 5. Calculate Frequency Statistics
trades_series = pd.Series(1, index=executed_trades)

# Resample by Day, Week, and Month
daily_trades = trades_series.resample('D').sum()
weekly_trades = trades_series.resample('W').sum()
monthly_trades = trades_series.resample('ME').sum()

# Filter out weekends / holidays where trades = 0
active_days = daily_trades[daily_trades > 0]
active_weeks = weekly_trades[weekly_trades > 0]
active_months = monthly_trades[monthly_trades > 0]

print("\n==================================")
print("     REAL TRADE FREQUENCY         ")
print("==================================")
print(f"Total Trades Over Out-Of-Sample Period: {len(executed_trades)}")
print(f"Average Trades per Active Day:   {active_days.mean():.2f}")
print(f"Max Trades in a Single Day:      {active_days.max():.0f}")
print(f"Average Trades per Active Week:  {active_weeks.mean():.2f}")
print(f"Average Trades per Active Month: {active_months.mean():.2f}")
print("==================================")
print("\nDaily Breakdown (How many days had X trades?):")
print(active_days.value_counts().sort_index().to_string())
