# Data Analytics using Python - Hands-on Lab Evaluation 2
# BCA503[T+P] | Module II
# Dataset: UCI Adult Income Dataset

import os
import time
import numpy as np
import pandas as pd

DATA_FILE = "adult.csv"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

# ------------------------------------------------------------
# 0. Load dataset
# ------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    print("adult.csv not found. Downloading the UCI Adult dataset...")
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS, skipinitialspace=True)
    df.to_csv(DATA_FILE, index=False)
else:
    df = pd.read_csv(DATA_FILE, skipinitialspace=True)

# Normalize whitespace in text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print("Shape:", df.shape)
display(df.head())

# ------------------------------------------------------------
# 1. NumPy arrays: field access, basic slicing, advanced indexing
# ------------------------------------------------------------
age_array = df["age"].to_numpy()
hours_array = df["hours-per-week"].to_numpy()

print("Age array first 10 values:", age_array[:10])
print("Hours-per-week array first 10 values:", hours_array[:10])

print("\nBasic slicing (age[5:15]):")
print(age_array[5:15])

print("\nAdvanced/boolean indexing (age >= 50, first 10):")
print(age_array[age_array >= 50][:10])

print("\nFancy indexing using explicit positions:")
positions = [0, 5, 10, 15, 20]
print(age_array[positions])

print("\nDifference:")
print("Basic slicing selects a continuous range and normally returns a NumPy view.")
print("Advanced indexing selects specific positions/conditions and returns a new NumPy array.")

# ------------------------------------------------------------
# 2. NumPy broadcasting + timing comparison
# ------------------------------------------------------------
n = len(df)

# Broadcast two 1-D arrays into a 2-D derived array without a loop.
age_col = age_array.reshape(-1, 1)
hours_col = hours_array.reshape(-1, 1)

start = time.perf_counter()
for _ in range(100):
    broadcast_result = age_col + hours_col
broadcast_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(100):
    loop_result = np.array([[age_array[i] + hours_array[i]] for i in range(n)])
loop_time = time.perf_counter() - start

print("\nBroadcasting result shape:", broadcast_result.shape)
print("First 5 derived values:")
print(broadcast_result[:5].ravel())

print(f"\nBroadcasting time for 100 runs: {broadcast_time:.6f} seconds")
print(f"Python-loop time for 100 runs: {loop_time:.6f} seconds")
print(f"Broadcasting is approximately {loop_time / broadcast_time:.2f}x faster than the loop.")

print("\nWhy broadcasting is efficient:")
print("NumPy performs the operation using optimized compiled array operations,")
print("avoiding Python-level iteration over every record.")

# ------------------------------------------------------------
# 3. Pandas: three simultaneous conditions across three columns
# ------------------------------------------------------------
filtered = df[
    (df["age"] >= 30) &
    (df["education-num"] >= 10) &
    (df["hours-per-week"] >= 40)
].copy()

print("\nFiltered subset: age >= 30 AND education-num >= 10 AND hours-per-week >= 40")
print("Number of matching records:", len(filtered))
display(filtered.head(10))

print("\nWhy three conditions are used:")
print("Using only age, education, or working hours would include many records")
print("that do not satisfy the complete business requirement. The three-condition")
print("filter returns only records meeting all three requirements simultaneously.")

filtered.to_csv("filtered_three_conditions.csv", index=False)

# ------------------------------------------------------------
# 4. CSV/XLSX round-trip proof
# ------------------------------------------------------------
# Basic groupby + aggregate operation
grouped = (
    df.groupby("education", dropna=False)
      .agg(
          record_count=("age", "size"),
          average_age=("age", "mean"),
          average_hours=("hours-per-week", "mean")
      )
      .reset_index()
      .sort_values("record_count", ascending=False)
)

grouped.to_csv("education_summary.csv", index=False)
grouped.to_excel("education_summary.xlsx", index=False)

csv_reload = pd.read_csv("education_summary.csv")
xlsx_reload = pd.read_excel("education_summary.xlsx")

# Align dtypes for a fair value comparison where Excel/CSV infer numeric types differently.
numeric_cols = ["record_count", "average_age", "average_hours"]
for col in numeric_cols:
    csv_reload[col] = pd.to_numeric(csv_reload[col])
    xlsx_reload[col] = pd.to_numeric(xlsx_reload[col])

values_match = np.allclose(
    csv_reload[numeric_cols].to_numpy(dtype=float),
    xlsx_reload[numeric_cols].to_numpy(dtype=float),
    equal_nan=True
)

columns_match = list(csv_reload.columns) == list(xlsx_reload.columns)

print("\nCSV/XLSX round-trip verification")
print("Column names match:", columns_match)
print("Numeric values match:", values_match)
print("\nCSV dtypes:")
print(csv_reload.dtypes)
print("\nXLSX dtypes:")
print(xlsx_reload.dtypes)

display(csv_reload.head())

# ------------------------------------------------------------
# 5. value_counts() and suspiciously dominant values
# ------------------------------------------------------------
income_counts = df["income"].value_counts(dropna=False)
workclass_counts = df["workclass"].value_counts(dropna=False)
native_country_counts = df["native-country"].value_counts(dropna=False)

print("\nIncome value counts:")
print(income_counts)

print("\nWorkclass value counts:")
print(workclass_counts.head(10))

print("\nNative-country value counts:")
print(native_country_counts.head(10))

income_share = income_counts.iloc[0] / len(df)
print(f"\nLargest income-class share: {income_share:.2%}")

if income_share >= 0.70:
    print("Risk: the target is strongly imbalanced. A model can look accurate by")
    print("mostly predicting the dominant class while performing poorly on the minority class.")
else:
    print("The target is not above the 70% severe-imbalance threshold used in this lab check.")

# Also identify columns with a dominant category >= 90%.
dominance = {}
for col in df.columns:
    counts = df[col].value_counts(dropna=False)
    dominance[col] = counts.iloc[0] / len(df)

dominance_series = pd.Series(dominance).sort_values(ascending=False)
print("\nTop dominant-value shares by column:")
print(dominance_series.head(10))

print("\nCompleted. Files created:")
for f in [
    "filtered_three_conditions.csv",
    "education_summary.csv",
    "education_summary.xlsx"
]:
    print("-", f)

# ------------------------------------------------------------
# 6. Short conclusion (150-200 words)
# ------------------------------------------------------------
conclusion = """
This hands-on lab used the Adult Income Dataset to demonstrate NumPy and Pandas
operations for data manipulation. First, two numeric columns, age and
hours-per-week, were converted into NumPy arrays. Basic slicing, boolean
indexing, and fancy indexing were then used to extract values in different
ways. NumPy broadcasting was also applied to create a derived array without an
explicit Python loop, and its execution time was compared with a loop-based
approach. The broadcasting operation was faster because NumPy performs array
operations using optimized compiled routines.

Next, Pandas was used to create a meaningful subset using three simultaneous
conditions: age of at least 30, education-num of at least 10, and at least 40
working hours per week. A groupby operation was then used to summarize records
by education. The result was written to both CSV and Excel files and reloaded
to verify that column names and numeric values matched. Finally, value_counts()
was used to inspect the income and categorical distributions. The analysis
shows why class imbalance and dominant categories should be checked before
using the data for later analysis or modeling.
""".strip()

print("\nCONCLUSION:\n")
print(conclusion)
