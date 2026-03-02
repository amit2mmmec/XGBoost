import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

print("Loading Big Trend Dataset...")
df = pd.read_csv('NIFTY_VIX_Big_Trends_1_to_3.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# ==========================================
# THE FIX: NORMALIZED ATR (NATR)
# Convert raw point volatility into percentage volatility
# ==========================================
print("Curing Regime Blindness (Calculating Normalized ATR)...")
df['NATR14_5min'] = (df['ATR14_5min'] / df['close']) * 100
df['NATR14_15min'] = (df['ATR14_15min'] / df['close']) * 100
df['NATR14_60min'] = (df['ATR14_60min'] / df['close']) * 100

# We MUST drop the raw ATR columns from the features so the AI only sees percentages!
drop_cols = [
    'open', 'high', 'low', 'close', 'Target', 
    'EMA20_5min', 'EMA50_5min', 'EMA20_15min', 
    'EMA50_15min', 'EMA20_60min', 'EMA50_60min',
    'ATR14_5min', 'ATR14_15min', 'ATR14_60min' # <--- Raw points dropped
]
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df['Target']

print("Training XGBoost Engine...")
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
# FULL EXECUTION SIMULATOR 
# ==========================================
print("\nSimulating Execution (65% Confidence, Max 5 Trades/Day, 1:3 RR)...")
dates = merged_df.index
probs = merged_df['Buy_Probability'].values
close_prices = merged_df['close'].values
high_prices = merged_df['high'].values
low_prices = merged_df['low'].values
atrs = merged_df['ATR14_5min'].values

SLIPPAGE_AND_BROKERAGE_POINTS = 1.5  
CONFIDENCE_THRESHOLD = 0.65  
MAX_TRADES_PER_DAY = 5  

trade_log = []

in_trade = False
trade_tp = 0
trade_sl = 0
timeout_bar = 0
entry_price = 0
entry_time = None

trades_today = 0
current_day = dates[0].date()

for i in range(len(merged_df)):
    current_time = dates[i]
    today = current_time.date()
    
    if today != current_day:
        trades_today = 0
        current_day = today

    if in_trade:
        if high_prices[i] >= trade_tp:
            pnl = (trade_tp - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            trade_log.append({
                'Entry_Time': entry_time, 'Exit_Time': current_time, 
                'Entry_Price': entry_price, 'Exit_Price': trade_tp, 
                'Take_Profit': trade_tp, 'Stop_Loss': trade_sl,
                'Result': 'Win', 'PnL_Points': pnl
            })
            in_trade = False
            
        elif low_prices[i] <= trade_sl:
            pnl = (trade_sl - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            trade_log.append({
                'Entry_Time': entry_time, 'Exit_Time': current_time, 
                'Entry_Price': entry_price, 'Exit_Price': trade_sl, 
                'Take_Profit': trade_tp, 'Stop_Loss': trade_sl,
                'Result': 'Loss', 'PnL_Points': pnl
            })
            in_trade = False
            
        elif i >= timeout_bar: 
            pnl = (close_prices[i] - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            result = 'Timeout_Win' if pnl > 0 else 'Timeout_Loss'
            trade_log.append({
                'Entry_Time': entry_time, 'Exit_Time': current_time, 
                'Entry_Price': entry_price, 'Exit_Price': close_prices[i], 
                'Take_Profit': trade_tp, 'Stop_Loss': trade_sl,
                'Result': result, 'PnL_Points': pnl
            })
            in_trade = False
        continue 
        
    if probs[i] >= CONFIDENCE_THRESHOLD and not np.isnan(atrs[i]) and trades_today < MAX_TRADES_PER_DAY:
        in_trade = True
        entry_time = current_time
        entry_price = close_prices[i]
        
        # Note: We still use the RAW points (atrs[i]) here to set exact targets!
        trade_tp = entry_price + (3.0 * atrs[i]) 
        trade_sl = entry_price - (1.0 * atrs[i]) 
        timeout_bar = i + 180 
        trades_today += 1

# ==========================================
# CALCULATE METRICS & EXPORT
# ==========================================
if len(trade_log) == 0:
    print("No trades met the 65% threshold!")
else:
    results_df = pd.DataFrame(trade_log)
    results_df['Cumulative_PnL'] = results_df['PnL_Points'].cumsum()

    results_df['Peak'] = results_df['Cumulative_PnL'].cummax()
    results_df['Drawdown'] = results_df['Cumulative_PnL'] - results_df['Peak']
    max_drawdown = results_df['Drawdown'].min()

    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['PnL_Points'] > 0])
    win_rate = (winning_trades / total_trades) * 100
    avg_points_per_trade = results_df['PnL_Points'].mean()
    total_points = results_df['Cumulative_PnL'].iloc[-1]

    print("\n==========================================")
    print("      FINAL PERFORMANCE (1:3 TRENDS)      ")
    print("==========================================")
    print(f"Total Executed Trades: {total_trades}")
    print(f"Overall Win Rate:      {win_rate:.2f}%")
    print(f"Net Nifty Points Won:  +{total_points:.2f} Points")
    print(f"Average Points/Trade:  {avg_points_per_trade:.2f} Points")
    print(f"Max Drawdown (Points): {max_drawdown:.2f} Points")
    print("==========================================")

    float_cols = ['Entry_Price', 'Exit_Price', 'Take_Profit', 'Stop_Loss', 'PnL_Points', 'Cumulative_PnL']
    results_df[float_cols] = results_df[float_cols].round(2)
    results_df.drop(columns=['Peak', 'Drawdown'], inplace=True)
    
    csv_filename = 'Complete_Trade_Log.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\n✅ Successfully generated '{csv_filename}'. Open this in Excel to review your 2024 and 2025 trades!")
