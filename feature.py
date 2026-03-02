import pandas as pd
import numpy as np
import time

print("Starting minute-by-minute data processing...")
start_time = time.time()

# 1. Load the full datasets
nifty_df = pd.read_csv('NIFTY_50_minute_with_EMA_conditions.csv')
vix_df = pd.read_csv('INDIA_VIX_minute.csv')

# 2. Format Dates and set as Index
nifty_df['date'] = pd.to_datetime(nifty_df['date'])
vix_df['date'] = pd.to_datetime(vix_df['date'])

nifty_df.set_index('date', inplace=True)
vix_df.set_index('date', inplace=True)

# 3. THE FIX: Forward Fill all lagging indicators
# This ensures that a 60-min EMA calculated at 10:00 stays active for 10:01, 10:02, etc.
cols_to_ffill = [col for col in nifty_df.columns if any(x in col for x in ['EMA', 'ATR', 'ROC', 'slope'])]
nifty_df[cols_to_ffill] = nifty_df[cols_to_ffill].ffill()

# 4. Merge VIX into Nifty data
vix_df = vix_df[['close']].rename(columns={'close': 'VIX_Close'})
master_df = nifty_df.join(vix_df, how='left')
master_df['VIX_Close'] = master_df['VIX_Close'].ffill()

if 'volume' in master_df.columns:
    master_df.drop(columns=['volume'], inplace=True)

# 5. Feature Engineering: Dynamic Distance from Mean
# Calculated every single minute against the active higher-timeframe EMA
master_df['Dist_to_EMA20_5m'] = ((master_df['close'] - master_df['EMA20_5min']) / master_df['EMA20_5min']) * 100
master_df['Dist_to_EMA50_5m'] = ((master_df['close'] - master_df['EMA50_5min']) / master_df['EMA50_5min']) * 100

master_df['Dist_to_EMA20_15m'] = ((master_df['close'] - master_df['EMA20_15min']) / master_df['EMA20_15min']) * 100
master_df['Dist_to_EMA50_15m'] = ((master_df['close'] - master_df['EMA50_15min']) / master_df['EMA50_15min']) * 100

master_df['Dist_to_EMA20_60m'] = ((master_df['close'] - master_df['EMA20_60min']) / master_df['EMA20_60min']) * 100
master_df['Dist_to_EMA50_60m'] = ((master_df['close'] - master_df['EMA50_60min']) / master_df['EMA50_60min']) * 100

# 6. Feature Engineering: VIX Momentum (5-minute rolling change)
master_df['VIX_ROC5'] = master_df['VIX_Close'].pct_change(periods=5) * 100

# 7. Feature Engineering: Micro-Market Structure (15-min rolling)
master_df['Rolling_High_15m'] = master_df['high'].rolling(window=15).max()
master_df['Rolling_Low_15m'] = master_df['low'].rolling(window=15).min()
master_df['Position_in_Range_15m'] = (master_df['close'] - master_df['Rolling_Low_15m']) / (master_df['Rolling_High_15m'] - master_df['Rolling_Low_15m'])

# 8. Feature Engineering: Time Cyclicality
master_df['Hour'] = master_df.index.hour
master_df['Minute'] = master_df.index.minute
master_df['Minutes_from_Open'] = ((master_df['Hour'] - 9) * 60) + (master_df['Minute'] - 15)

# 9. Optimized Target Generation: Triple Barrier Method
print("Engineering targets using vectorized NumPy operations...")
close_arr = master_df['close'].values
high_arr = master_df['high'].values
low_arr = master_df['low'].values
atr_arr = master_df['ATR14_5min'].values

targets = np.full(len(master_df), np.nan)
look_forward = 30 # Check the next 30 minutes

for i in range(len(master_df) - look_forward):
    c = close_arr[i]
    a = atr_arr[i]
    
    if np.isnan(a) or a == 0:
        continue
        
    # Strict 1:1.5 Risk/Reward Ratio based on live ATR
    tp = c + (1.5 * a)
    sl = c - (1.0 * a)
    
    f_high = high_arr[i+1 : i+1+look_forward]
    f_low = low_arr[i+1 : i+1+look_forward]
    
    tp_hits = np.where(f_high >= tp)[0]
    sl_hits = np.where(f_low <= sl)[0]
    
    first_tp = tp_hits[0] if len(tp_hits) > 0 else 999
    first_sl = sl_hits[0] if len(sl_hits) > 0 else 999
    
    if first_tp < first_sl and first_tp != 999:
        targets[i] = 1.0  
    else:
        targets[i] = 0.0  

master_df['Target'] = targets

# 10. Clean up
master_df.drop(columns=['Hour', 'Minute'], inplace=True)

# Drop only the initial rows where the very first 60min EMA hasn't formed yet
final_df = master_df.dropna(subset=['EMA50_60min', 'Target'])

# Export to CSV
file_name = 'NIFTY_VIX_Minute_By_Minute_Engineered.csv'
final_df.to_csv(file_name)

print(f"Data processing completed in {time.time() - start_time:.2f} seconds.")
print(f"File successfully saved as {file_name}. It now contains every minute evaluated dynamically!")
