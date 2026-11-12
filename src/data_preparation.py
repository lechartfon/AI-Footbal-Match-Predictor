"""
Phase 1: Data Preparation
Load dataset, select features, clean, normalize, and split into Train/Val/Test
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("=" * 50)
print("PHASE 1: DATA PREPARATION")
print("=" * 50)

# 1. Load dataset
print("\n1. Loading Matches.csv dataset...")
df = pd.read_csv('../dataset/Matches.csv')
print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Select ONLY the 3 features per team (6 total)
# NOTE: goals (FTHome/FTAway) are deliberately excluded: they fully
# determine FTResult, so including them would leak the answer.
print("\n2. Selecting specific features...")
features_columns = [
    # Home Team
    'HomeTarget',
    'HomeCorners',
    'HomeFouls',
    # Away Team
    'AwayTarget',
    'AwayCorners',
    'AwayFouls',
    # Target
    'FTResult'
]

# Filter required columns
df_filtered = df[features_columns].copy()
print(f"   Features selected: {len(features_columns) - 1}")

# 3. Add calculated features: Saves
print("\n3. Calculating additional features...")
# Home goalkeeper saves = Away shots on target
df_filtered['HomeSaves'] = df_filtered['AwayTarget']
# Away goalkeeper saves = Home shots on target
df_filtered['AwaySaves'] = df_filtered['HomeTarget']

print("   Calculated features:")
print("   - Saves: Opponent's shots on target")

# 4. Drop rows with NaN values
print("\n4. Cleaning data (dropping NaN)...")
rows_before = len(df_filtered)
df_filtered = df_filtered.dropna()
rows_after = len(df_filtered)
print(f"   Rows dropped: {rows_before - rows_after}")
print(f"   Remaining rows: {rows_after}")

# 5. Separate features (X) and target (y)
print("\n5. Separating features and target...")
# Final features (8 total, 4 per team)
feature_names = [
    'HomeTarget', 'HomeCorners', 'HomeFouls', 'HomeSaves',
    'AwayTarget', 'AwayCorners', 'AwayFouls', 'AwaySaves'
]
X = df_filtered[feature_names].values
y_raw = df_filtered['FTResult'].values

print(f"   Shape of X: {X.shape}")
print(f"   Total features: {X.shape[1]}")

# 6. Encode target: H=0, D=1, A=2 (one-hot encoding)
print("\n6. Encoding target (H/D/A)...")
label_mapping = {'H': 0, 'D': 1, 'A': 2}
y_encoded = np.array([label_mapping[label] for label in y_raw])

# One-hot encoding
y = np.zeros((len(y_encoded), 3))
for i, label in enumerate(y_encoded):
    y[i, label] = 1

print(f"   H (Home Win) = [1, 0, 0]")
print(f"   D (Draw)     = [0, 1, 0]")
print(f"   A (Away Win) = [0, 0, 1]")
print(f"   Shape of y: {y.shape}")

# Count class distribution
unique, counts = np.unique(y_encoded, return_counts=True)
class_names = ['Home Win', 'Draw', 'Away Win']
print("\n   Class distribution:")
for i, count in enumerate(counts):
    print(f"   {class_names[i]}: {count} ({count/len(y_encoded)*100:.1f}%)")

# 7. Split into Train (70%), Validation (15%), Test (15%)
print("\n7. Splitting dataset...")
# First separate Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y_encoded
)

# Then separate Train (70%) and Validation (15% of total = ~17.6% of temp)
y_temp_encoded = np.argmax(y_temp, axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp_encoded
)

print(f"   Train:      {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   Test:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# 8. Normalize features (StandardScaler)
print("\n8. Normalizing features (mean=0, std=1)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("   Normalization complete")
print(f"   Train mean after scaling: {X_train_scaled.mean():.6f}")
print(f"   Train standard deviation: {X_train_scaled.std():.6f}")

# 9. Create folders and save processed data
print("\n9. Saving processed data...")
os.makedirs('../data', exist_ok=True)

np.save('../data/X_train.npy', X_train_scaled)
np.save('../data/y_train.npy', y_train)
np.save('../data/X_val.npy', X_val_scaled)
np.save('../data/y_val.npy', y_val)
np.save('../data/X_test.npy', X_test_scaled)
np.save('../data/y_test.npy', y_test)

# Save the scaler for later use
import pickle
with open('../data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
np.save('../data/feature_names.npy', np.array(feature_names))

print("   Data saved in ../data/:")
print("   - X_train.npy, y_train.npy")
print("   - X_val.npy, y_val.npy")
print("   - X_test.npy, y_test.npy")
print("   - scaler.pkl")
print("   - feature_names.npy")

print("\n" + "=" * 50)
print("PHASE 1 COMPLETE")
print("=" * 50)
