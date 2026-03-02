import pandas as pd
import numpy as np

print("Loading Price Action Data...")
df = pd.read_csv('NIFTY_Price_Action_Base.csv')
df['date'] = pd.to_datetime(df['date'])
df['Day'] = df['date'].dt.date
df['Time_Obj'] = df['date'].dt.time

# --- Calculate ATR directly to avoid KeyErrors ---
print("Calculating ATR...")
df['tr'] = np.maximum(df['high'] - df['low'], 
                     np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                abs(df['low'] - df['close'].shift(1))))
df['ATR14'] = df['tr'].rolling(window=14).mean()

SLIPPAGE = 1.5
trade_log = []
in_trade = False
current_day = None

start_trading = pd.to_datetime('09:30:00').time()
end_trading = pd.to_datetime('14:00:00').time()
eod_square_off = pd.to_datetime('15:15:00').time()

print("Starting Trailing Stop Simulation...")

for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    
    if row['Day'] != current_day:
        current_day = row['Day']
        trades_today = 0
        in_trade = False
        
    if in_trade:
        # Use a fallback ATR if current is NaN
        current_atr = row['ATR14'] if not np.isnan(row['ATR14']) else 15
        
        if side == 'Buy':
            # Update the peak price of the trade to trail from the top
            high_since_entry = max(high_since_entry, row['high'])
            # Trail SL: Peak Price minus 2.0x ATR
            new_sl = high_since_entry - (2.0 * current_atr)
            if new_sl > sl:
                sl = new_sl
                
            if row['high'] >= tp:
                trade_log.append({'Date': row['date'], 'Result': 'Win', 'Points': (tp - entry) - SLIPPAGE})
                in_trade = False
            elif row['low'] <= sl:
                pnl = (sl - entry) - SLIPPAGE
                trade_log.append({'Date': row['date'], 'Result': 'Exit_Trail', 'Points': pnl})
                in_trade = False
                
        elif side == 'Sell':
            low_since_entry = min(low_since_entry, row['low'])
            # Trail SL: Lowest Price plus 2.0x ATR
            new_sl = low_since_entry + (2.0 * current_atr)
            if new_sl < sl:
                sl = new_sl
                
            if row['low'] <= tp:
                trade_log.append({'Date': row['date'], 'Result': 'Win', 'Points': (entry - tp) - SLIPPAGE})
                in_trade = False
            elif row['high'] >= sl:
                pnl = (entry - sl) - SLIPPAGE
                trade_log.append({'Date': row['date'], 'Result': 'Exit_Trail', 'Points': pnl})
                in_trade = False
        
        # Square off at end of day
        if in_trade and row['Time_Obj'] >= eod_square_off:
            pnl = (row['close'] - entry) if side == 'Buy' else (entry - row['close'])
            trade_log.append({'Date': row['date'], 'Result': 'EOD', 'Points': pnl - SLIPPAGE})
            in_trade = False
        continue

    # --- ENTRY LOGIC ---
    if not in_trade and trades_today < 1:
        if start_trading <= row['Time_Obj'] <= end_trading:
            current_atr = row['ATR14'] if not np.isnan(row['ATR14']) else 15
            
            # PDH Breakout (Long)
            if prev_row['close'] <= row['PDH'] and row['close'] > row['PDH']:
                in_trade = True
                side = 'Buy'
                entry = row['close']
                high_since_entry = row['high']
                sl = entry - (1.5 * current_atr) # Initial risk
                tp = entry + (5.0 * current_atr) # Aiming for huge moves
                trades_today += 1
                
            # PDL Breakout (Short)
            elif prev_row['close'] >= row['PDL'] and row['close'] < row['PDL']:
                in_trade = True
                side = 'Sell'
                entry = row['close']
                low_since_entry = row['low']
                sl = entry + (1.5 * current_atr)
                tp = entry - (5.0 * current_atr)
                trades_today += 1

# Results Summary
if len(trade_log) > 0:
    results = pd.DataFrame(trade_log)
    print("\n" + "="*45)
    print("   PDH/PDL BREAKOUT + CHANDELIER TRAILING    ")
    print("="*45)
    print(f"Total Trades:       {len(results)}")
    print(f"Profit % (Net > 0): {len(results[results['Points'] > 0]) / len(results) * 100:.2f}%")
    print(f"Net Points Won:     {results['Points'].sum():.2f}")
    print(f"Average Points/Trd: {results['Points'].mean():.2f}")
    print(f"Max Win (Points):   {results['Points'].max():.2f}")
    print(f"Max Loss (Points):  {results['Points'].min():.2f}")
    print("="*45)
else:
    print("No trades triggered. Verify PDH/PDL columns in CSV.")
