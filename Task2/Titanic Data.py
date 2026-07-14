# Import pandas for data manipulation and numpy for numeric helpers
import pandas as pd
import numpy as np

# Load the Titanic dataset directly from the GitHub raw URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Show the first few rows to confirm it loaded correctly
display(df.head())

# Shape of the dataset (rows, columns)
print("Rows and columns:", df.shape)

# Numeric summary statistics (count, mean, std, min, quartiles, max)
display(df.describe())

# Count of missing values per column, so we know what to clean
print("\nMissing values per column:")
print(df.isnull().sum())

# Work on a copy so the original loaded data stays intact
clean = df.copy()

# Age -> fill missing with the median age
median_age = clean["Age"].median()
clean["Age"] = clean["Age"].fillna(median_age)

# Embarked -> fill missing with the mode (most frequent value)
mode_embarked = clean["Embarked"].mode()[0]
clean["Embarked"] = clean["Embarked"].fillna(mode_embarked)

# Cabin -> drop it entirely (mostly missing)
clean = clean.drop(columns=["Cabin"])

# Name -> standardize to title case for a consistent format
clean["Name"] = clean["Name"].str.title()

# Confirm there are no missing values left in the cleaned columns
print("Median age used:", median_age)
print("Mode Embarked used:", mode_embarked)
print("\nMissing values after cleaning:")
print(clean.isnull().sum())

# Display the cleaned DataFrame
display(clean.head())

# Group by passenger class: mean Fare and number of passengers per class
grouped = clean.groupby("Pclass").agg(
    mean_fare=("Fare", "mean"),
    passenger_count=("PassengerId", "count"),
).reset_index()

print("Grouped by Pclass:")
display(grouped)

# Secondary DataFrame: a mock human-readable description for each class
class_desc = pd.DataFrame({
    "Pclass": [1, 2, 3],
    "ClassDescription": ["Upper", "Middle", "Lower"],
})

# Merge the grouped stats with the descriptions on the Pclass key
merged = pd.merge(grouped, class_desc, on="Pclass")

print("Merged result:")
display(merged)

# Pivot table: rows = Embarked, columns = Sex, values = mean Fare
pivot = pd.pivot_table(
    clean,
    index="Embarked",
    columns="Sex",
    values="Fare",
    aggfunc="mean",
)

# Fill any missing pivot cells with 0
pivot = pivot.fillna(0)

print("Pivot table (mean Fare by Embarked x Sex):")
display(pivot)

# Total family size aboard for each passenger
clean["FamilySize"] = clean["SibSp"] + clean["Parch"] + 1

# Helper function to categorize each passenger by family size
def travel_group(size):
    if size == 1:
        return "Solo"
    elif size <= 3:      # 2 or 3
        return "Small"
    else:                # 4 or more
        return "Large"

# Apply the function to create the new TravelGroup feature
clean["TravelGroup"] = clean["FamilySize"].apply(travel_group)

print("TravelGroup counts:")
print(clean["TravelGroup"].value_counts())
display(clean[["Name", "SibSp", "Parch", "FamilySize", "TravelGroup"]].head())

# Compute the IQR bounds for Fare
Q1 = clean["Fare"].quantile(0.25)
Q3 = clean["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count how many fares fall outside the bounds (before capping)
num_outliers = ((clean["Fare"] < lower_bound) | (clean["Fare"] > upper_bound)).sum()

# Cap (clip) the Fare values to the bounds
clean["Fare"] = clean["Fare"].clip(lower=lower_bound, upper=upper_bound)

print(f"Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
print(f"Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}")
print(f"Number of Fare outliers capped: {num_outliers}")

# Display the final, fully processed DataFrame
print("Final processed DataFrame:")
display(clean.head())

# Store key summary statistics in a dictionary
summary_stats = {
    "num_rows": int(clean.shape[0]),
    "num_columns": int(clean.shape[1]),
    "mean_fare": round(float(clean["Fare"].mean()), 2),
    "max_family_size": int(clean["FamilySize"].max()),
    "most_common_travel_group": clean["TravelGroup"].mode()[0],
    "mean_age": round(float(clean["Age"].mean()), 2),
    "fare_outliers_capped": int(num_outliers),
}

print("\nSummary statistics dictionary:")
for key, value in summary_stats.items():
    print(f"  {key}: {value}")
