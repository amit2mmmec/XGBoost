import pandas as pd
import numpy as np
import time

print("Re-engineering targets for Massive 1:3 Risk/Reward Trends...")
start_time = time.time()

# Load the current engineered file
df = pd.read_csv('NIFTY_VIX_Minute_By_Minute_Engineered.csv')

close_arr = df['close'].values
high_arr = df['high'].values
low_arr = df['low'].values
atr_arr = df['ATR14_5min'].values

targets = np.full(len(df), np.nan)

# LOOK FORWARD 3 HOURS (180 minutes) to give big trends room to develop
look_forward = 180 

for i in range(len(df) - look_forward):
    c = close_arr[i]
    a = atr_arr[i]
    
    if np.isnan(a) or a == 0:
        continue
        
    # THE BIG CHANGE: 3.0x Take Profit, 1.0x Stop Loss (1:3 RR)
    tp = c + (3.0 * a)
    sl = c - (1.0 * a)
    
    f_high = high_arr[i+1 : i+1+look_forward]
    f_low = low_arr[i+1 : i+1+look_forward]
    
    tp_hits = np.where(f_high >= tp)[0]
    sl_hits = np.where(f_low <= sl)[0]
    
    first_tp = tp_hits[0] if len(tp_hits) > 0 else 999
    first_sl = sl_hits[0] if len(sl_hits) > 0 else 999
    
    # If it hits our massive 3x TP before the 1x SL
    if first_tp < first_sl and first_tp != 999:
        targets[i] = 1.0  
    else:
        targets[i] = 0.0  

df['Target'] = targets

# Drop nulls from the very end of the dataset
final_df = df.dropna(subset=['Target'])

# Save the new high-target file
final_csv_name = 'NIFTY_VIX_Big_Trends_1_to_3.csv'
final_df.to_csv(final_csv_name, index=False)
print(f"Done in {time.time() - start_time:.2f}s. Saved as {final_csv_name}")
