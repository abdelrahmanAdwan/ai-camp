# pandas for data manipulation, numpy for numeric helpers
import pandas as pd
import numpy as np

# Load the Titanic dataset directly from the GitHub raw URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Original shape (rows, columns):", df.shape)
display(df.head())

# Column data types and non-null counts
print("Info:")
df.info()

# Missing values per column
print("\nMissing values per column:")
print(df.isnull().sum())

# Number of fully-duplicated rows
print("\nFully duplicated rows:", df.duplicated().sum())

# Work on a copy so the raw data stays intact
clean = df.copy()

# Age -> median imputation
median_age = clean["Age"].median()
clean["Age"] = clean["Age"].fillna(median_age)

# Embarked -> mode imputation
mode_embarked = clean["Embarked"].mode()[0]
clean["Embarked"] = clean["Embarked"].fillna(mode_embarked)

# Cabin -> delete (too many missing values)
clean = clean.drop(columns=["Cabin"])

print("Median age imputed:", median_age)
print("Mode Embarked imputed:", mode_embarked)
print("\nMissing values after handling:")
print(clean.isnull().sum())

# --- Demonstrate duplicate removal ---
# Intentionally append a copy of the first row to create a duplicate
clean = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)
print("Rows after adding a demo duplicate:", clean.shape[0])
print("Duplicate rows detected:", clean.duplicated().sum())

# Remove exact duplicate rows and reset the index
clean = clean.drop_duplicates().reset_index(drop=True)
print("Rows after drop_duplicates():", clean.shape[0])
print("Duplicate rows remaining:", clean.duplicated().sum())

# Total family size aboard
clean["FamilySize"] = clean["SibSp"] + clean["Parch"] + 1

# Traveling alone flag (1 = alone, 0 = with family)
clean["IsAlone"] = (clean["FamilySize"] == 1).astype(int)

print("FamilySize distribution:")
print(clean["FamilySize"].value_counts().sort_index())
print("\nIsAlone counts (1=alone, 0=with family):")
print(clean["IsAlone"].value_counts())
display(clean[["Name", "SibSp", "Parch", "FamilySize", "IsAlone"]].head())

# Compute IQR bounds for Fare
Q1 = clean["Fare"].quantile(0.25)
Q3 = clean["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers before capping
num_outliers = ((clean["Fare"] < lower_bound) | (clean["Fare"] > upper_bound)).sum()

# Cap the Fare values to the computed bounds
clean["Fare"] = clean["Fare"].clip(lower=lower_bound, upper=upper_bound)

print(f"Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
print(f"Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}")
print(f"Fare outliers capped: {num_outliers}")

# Final cleaned DataFrame
print("Final cleaned dataset:")
display(clean.head())
print("Final shape (rows, columns):", clean.shape)
print("Remaining missing values:", int(clean.isnull().sum().sum()))

# Cleaning summary (before -> after) for reproducibility
cleaning_summary = {
    "rows_before": int(df.shape[0]),
    "rows_after": int(clean.shape[0]),
    "columns_before": int(df.shape[1]),
    "columns_after": int(clean.shape[1]),
    "missing_before": int(df.isnull().sum().sum()),
    "missing_after": int(clean.isnull().sum().sum()),
    "median_age_imputed": float(median_age),
    "mode_embarked_imputed": mode_embarked,
    "fare_outliers_capped": int(num_outliers),
    "new_features": ["FamilySize", "IsAlone"],
}

print("\nCleaning summary:")
for key, value in cleaning_summary.items():
    print(f"  {key}: {value}")

# Export the cleaned dataset for reuse
clean.to_csv("titanic_cleaned.csv", index=False)
print("\nSaved cleaned dataset to 'titanic_cleaned.csv'")

