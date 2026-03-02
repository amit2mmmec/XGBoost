import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

print("Loading Big Trend Dataset and running XGBoost Engine...")
df = pd.read_csv('NIFTY_VIX_Big_Trends_1_to_3.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

drop_cols = ['open', 'high', 'low', 'close', 'Target', 'EMA20_5min', 'EMA50_5min', 'EMA20_15min', 'EMA50_15min', 'EMA20_60min', 'EMA50_60min']
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df['Target']

tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

all_test_dates = []
all_probabilities = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    all_test_dates.extend(X_test.index)
    all_probabilities.extend(probs)

pred_df = pd.DataFrame({'Buy_Probability': all_probabilities}, index=all_test_dates)
merged_df = pred_df.join(df[['close', 'high', 'low', 'ATR14_5min']])

# ==========================================
# COMBINED PNL & FREQUENCY SIMULATOR
# ==========================================
print("\nSimulating Execution with 65% Confidence & MAX 5 TRADES PER DAY...")
dates = merged_df.index
probs = merged_df['Buy_Probability'].values
close_prices = merged_df['close'].values
high_prices = merged_df['high'].values
low_prices = merged_df['low'].values
atrs = merged_df['ATR14_5min'].values

SLIPPAGE_AND_BROKERAGE_POINTS = 1.5  
CONFIDENCE_THRESHOLD = 0.65  
MAX_TRADES_PER_DAY = 5  # <--- CAP IS BACK

trade_log = []
executed_trades_timestamps = []

in_trade = False
trade_tp = 0
trade_sl = 0
timeout_bar = 0
entry_price = 0

trades_today = 0
current_day = dates[0].date()

for i in range(len(merged_df)):
    current_time = dates[i]
    today = current_time.date()
    
    # Reset daily trade counter at the start of a new day
    if today != current_day:
        trades_today = 0
        current_day = today

    if in_trade:
        if high_prices[i] >= trade_tp:
            pnl = (trade_tp - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            trade_log.append({'Date': current_time, 'Result': 'Win', 'PnL_Points': pnl})
            in_trade = False
        elif low_prices[i] <= trade_sl:
            pnl = (trade_sl - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            trade_log.append({'Date': current_time, 'Result': 'Loss', 'PnL_Points': pnl})
            in_trade = False
        elif i >= timeout_bar: # 180 min timeout for massive moves
            pnl = (close_prices[i] - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            result = 'Win' if pnl > 0 else 'Loss'
            trade_log.append({'Date': current_time, 'Result': f'Timeout_{result}', 'PnL_Points': pnl})
            in_trade = False
        continue 
        
    # Check Probability AND ensure we haven't hit the daily cap
    if probs[i] >= CONFIDENCE_THRESHOLD and not np.isnan(atrs[i]) and trades_today < MAX_TRADES_PER_DAY:
        in_trade = True
        entry_price = close_prices[i]
        trade_tp = entry_price + (3.0 * atrs[i]) # 3.0x REWARD
        trade_sl = entry_price - (1.0 * atrs[i]) # 1.0x RISK
        timeout_bar = i + 180 # Wait 3 hours to let it trend
        trades_today += 1
        
        executed_trades_timestamps.append(current_time)

# ==========================================
# CALCULATE METRICS
# ==========================================
if len(trade_log) == 0:
    print("No trades met the 65% threshold!")
else:
    # --- PNL METRICS ---
    results_df = pd.DataFrame(trade_log)
    results_df.set_index('Date', inplace=True)
    results_df['Cumulative_PnL'] = results_df['PnL_Points'].cumsum()

    results_df['Peak'] = results_df['Cumulative_PnL'].cummax()
    results_df['Drawdown'] = results_df['Cumulative_PnL'] - results_df['Peak']
    max_drawdown = results_df['Drawdown'].min()

    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['PnL_Points'] > 0])
    win_rate = (winning_trades / total_trades) * 100
    avg_points_per_trade = results_df['PnL_Points'].mean()
    total_points = results_df['Cumulative_PnL'].iloc[-1]

    # --- FREQUENCY METRICS ---
    trades_series = pd.Series(1, index=executed_trades_timestamps)
    daily_trades = trades_series.resample('D').sum()
    weekly_trades = trades_series.resample('W').sum()
    monthly_trades = trades_series.resample('ME').sum()

    active_days = daily_trades[daily_trades > 0]
    active_weeks = weekly_trades[weekly_trades > 0]
    active_months = monthly_trades[monthly_trades > 0]

    # --- PRINT OUT ---
    print("\n==========================================")
    print("      FINAL PERFORMANCE (1:3 TRENDS)      ")
    print("==========================================")
    print(f"Total Executed Trades: {total_trades}")
    print(f"Overall Win Rate:      {win_rate:.2f}%")
    print(f"Net Nifty Points Won:  +{total_points:.2f} Points")
    print(f"Average Points/Trade:  {avg_points_per_trade:.2f} Points")
    print(f"Max Drawdown (Points): {max_drawdown:.2f} Points")
    print("==========================================")

    print("\n==================================")
    print("     1:3 TRENDS TRADE FREQUENCY   ")
    print("==================================")
    print(f"Average Trades per Active Day:   {active_days.mean():.2f}")
    print(f"Max Trades in a Single Day:      {active_days.max():.0f}")
    print(f"Average Trades per Active Week:  {active_weeks.mean():.2f}")
    print(f"Average Trades per Active Month: {active_months.mean():.2f}")
    print("==================================")
    print("\nDaily Breakdown (How many days had exactly X trades?):")
    print(active_days.value_counts().sort_index().to_string())
