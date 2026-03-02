import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

print("Loading dataset and running XGBoost Engine...")
df = pd.read_csv('NIFTY_VIX_Minute_By_Minute_Engineered.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

drop_cols = ['open', 'high', 'low', 'close', 'Target', 'EMA20_5min', 'EMA50_5min', 'EMA20_15min', 'EMA50_15min', 'EMA20_60min', 'EMA50_60min']
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df['Target']

tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

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
merged_df = pred_df.join(df[['close', 'high', 'low', 'ATR14_5min']])

# ==========================================
# 5. PNL SIMULATOR (NO TIME LIMIT - PURE TP/SL)
# ==========================================
print("\nSimulating Real-World PnL (Holding until pure TP or SL is hit)...")
dates = merged_df.index
signals = merged_df['Signal'].values
close_prices = merged_df['close'].values
high_prices = merged_df['high'].values
low_prices = merged_df['low'].values
atrs = merged_df['ATR14_5min'].values

SLIPPAGE_AND_BROKERAGE_POINTS = 1.5  
MAX_TRADES_PER_DAY = 5               

trade_log = []
in_trade = False
trade_tp = 0
trade_sl = 0
entry_price = 0

trades_today = 0
current_day = dates[0].date()

for i in range(len(merged_df)):
    current_time = dates[i]
    today = current_time.date()
    
    if today != current_day:
        trades_today = 0
        current_day = today

    if in_trade:
        # Check Take Profit
        if high_prices[i] >= trade_tp:
            pnl = (trade_tp - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            trade_log.append({'Date': current_time, 'Result': 'Win', 'PnL_Points': pnl})
            in_trade = False
            
        # Check Stop Loss
        elif low_prices[i] <= trade_sl:
            pnl = (trade_sl - entry_price) - SLIPPAGE_AND_BROKERAGE_POINTS
            trade_log.append({'Date': current_time, 'Result': 'Loss', 'PnL_Points': pnl})
            in_trade = False
            
        # Removed the timeout condition completely!
        continue 
        
    if signals[i] == 1.0 and not np.isnan(atrs[i]) and trades_today < MAX_TRADES_PER_DAY:
        in_trade = True
        entry_price = close_prices[i]
        trade_tp = entry_price + (1.5 * atrs[i])
        trade_sl = entry_price - (1.0 * atrs[i])
        trades_today += 1

# ==========================================
# 6. CALCULATE FINAL METRICS
# ==========================================
results_df = pd.DataFrame(trade_log)
results_df.set_index('Date', inplace=True)
results_df['Cumulative_PnL'] = results_df['PnL_Points'].cumsum()

results_df['Peak'] = results_df['Cumulative_PnL'].cummax()
results_df['Drawdown'] = results_df['Cumulative_PnL'] - results_df['Peak']
max_drawdown = results_df['Drawdown'].min()

total_trades = len(results_df)
winning_trades = len(results_df[results_df['PnL_Points'] > 0])
win_rate = (winning_trades / total_trades) * 100

total_points = results_df['Cumulative_PnL'].iloc[-1]
avg_points_per_trade = results_df['PnL_Points'].mean()

print("==========================================")
print("      FINAL ALGORITHM PERFORMANCE         ")
print("        (NO TIMEOUT APPLIED)              ")
print("==========================================")
print(f"Total Executed Trades: {total_trades}")
print(f"Overall Win Rate:      {win_rate:.2f}%")
print(f"Net Nifty Points Won:  +{total_points:.2f} Points")
print(f"Average Points/Trade:  {avg_points_per_trade:.2f} Points")
print(f"Max Drawdown (Points): {max_drawdown:.2f} Points")
print("==========================================")
print("\n*Note: 1 Nifty Point = ₹50 (at current lot size).*")
print(f"Hypothetical Net Profit (1 Lot): ₹{total_points * 50:,.2f}")
