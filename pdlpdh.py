import pandas as pd
import numpy as np

print("Generating Price Action Dataset (PDH/PDL)...")

# Load your base minute data
df = pd.read_csv('NIFTY_VIX_Minute_By_Minute_Engineered.csv')
df['date'] = pd.to_datetime(df['date'])
df['Day'] = df['date'].dt.date

# 1. Calculate Daily High/Low for every day
daily_stats = df.groupby('Day').agg({'high': 'max', 'low': 'min'}).shift(1)
daily_stats.columns = ['PDH', 'PDL']

# 2. Map PDH/PDL back to the minute data
df = df.merge(daily_stats, left_on='Day', right_index=True)

# 3. Identify Opening Range (First 15 mins)
df['Time'] = df['date'].dt.time
orb_data = df[df['Time'] < pd.to_datetime('09:30:00').time()]
daily_orb = orb_data.groupby('Day').agg({'high': 'max', 'low': 'min'})
daily_orb.columns = ['ORB_High', 'ORB_Low']

df = df.merge(daily_orb, left_on='Day', right_index=True)

df.to_csv('NIFTY_Price_Action_Base.csv', index=False)
print("Done! Saved as NIFTY_Price_Action_Base.csv")
