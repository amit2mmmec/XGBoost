import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score

print("Loading engineered dataset...")
# Load the dataset we generated in the previous step
df = pd.read_csv('NIFTY_VIX_Minute_By_Minute_Engineered.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 1. Define Features (X) and Target (y)
# Drop raw prices and non-predictive columns. Keep only the math.
drop_cols = ['open', 'high', 'low', 'close', 'Target']
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df['Target']

# 2. Configure TimeSeriesSplit (e.g., 5 expanding windows)
tscv = TimeSeriesSplit(n_splits=5)

# 3. Configure XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=300,        # Number of trees
    max_depth=4,             # Prevent overfitting
    learning_rate=0.05,      # Slow, robust learning
    subsample=0.8,           # Use 80% of data per tree to add randomness
    colsample_bytree=0.8,    # Use 80% of features per tree
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

print("\nStarting Walk-Forward Cross Validation...")

fold = 1
precisions = []
accuracies = []

for train_index, test_index in tscv.split(X):
    # Split the data chronologically
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Get the dates for our own logging
    train_start, train_end = X_train.index[0].date(), X_train.index[-1].date()
    test_start, test_end = X_test.index[0].date(), X_test.index[-1].date()
    
    print(f"\n--- Fold {fold} ---")
    print(f"Training on: {train_start} to {train_end} ({len(X_train)} minutes)")
    print(f"Testing on:  {test_start} to {test_end} ({len(X_test)} minutes)")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict the unseen future
    predictions = model.predict(X_test)
    
    # Evaluate
    # Precision is crucial: Out of all the 'Buys' the model signaled, how many hit the 1.5x ATR target?
    prec = precision_score(y_test, predictions, zero_division=0)
    acc = accuracy_score(y_test, predictions)
    
    precisions.append(prec)
    accuracies.append(acc)
    
    print(f"Fold {fold} Precision (Win Rate of Buy Signals): {prec * 100:.2f}%")
    fold += 1

print("\n==================================")
print(f"Average Out-of-Sample Precision: {np.mean(precisions) * 100:.2f}%")
print("==================================")

# 4. Feature Importance: Find out what actually drives the edge
feature_importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Mathematical Features:")
print(feature_importances.head(10).to_string(index=False))
