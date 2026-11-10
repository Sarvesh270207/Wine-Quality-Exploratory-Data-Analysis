# WINE QUALITY DATASET - EXPLORATORY DATA ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



file_path = Path(__file__).parent /r"C:\Users\Sarvesh pal\OneDrive\Desktop\INTERNSHIP\Wine Quality Dataset.csv"

df = pd.read_csv(file_path)

sns.set_theme(style="whitegrid")


# TASK 1 - LOAD AND STUDY THE DATA

print("\n" + "=" * 60)
print("TASK 1: LOAD AND STUDY THE DATA")
print("=" * 60)

# Display the first five records
print("\nFirst five rows:")
print(df.head())

# Display the last five records
print("\nLast five rows:")
print(df.tail())

# Display the number of rows and columns
print("\nDataset shape (rows, columns):")
print(df.shape)

# Display all column names
print("\nDataset features:")
print(df.columns.tolist())

# Display information about the dataset
print("\nDataset information:")
df.info()

# Display summary statistics
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for duplicate records
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Display the main acid-related features
acid_columns = ["fixed acidity", "volatile acidity", "citric acid"]

print("\nAcidity features:")
print(df[acid_columns].head())


# TASK 2 - DISTRIBUTIONS AND CENTRAL TENDENCIES


print("\n" + "=" * 60)
print("TASK 2: DISTRIBUTIONS AND CENTRAL TENDENCIES")
print("=" * 60)

# Select all numerical columns
numerical_columns = df.select_dtypes(include=np.number).columns

# Calculate mean, median, mode and skewness
central_tendency = pd.DataFrame({
    "Mean": df[numerical_columns].mean(),
    "Median": df[numerical_columns].median(),
    "Mode": df[numerical_columns].mode().iloc[0],
    "Skewness": df[numerical_columns].skew()
})

print("\nCentral tendency and skewness:")
print(central_tendency.round(3))

# Interpret skewness
print("\nSkewness interpretation:")

for column in numerical_columns:
    skew_value = df[column].skew()

    if skew_value > 0:
        result = "Positively skewed (right-skewed)"
    elif skew_value < 0:
        result = "Negatively skewed (left-skewed)"
    else:
        result = "Approximately symmetric"

    print(f"{column}: {result} (Skewness = {skew_value:.3f})")

# Plot the distributions of all numerical features
df[numerical_columns].hist(
    figsize=(16, 12),
    bins=25,
    edgecolor="black",
    color="skyblue"
)

plt.suptitle("Distribution of Wine Quality Dataset Features", fontsize=16)
plt.tight_layout()
plt.show()

# Plot histograms with KDE curves for the main acidity features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, column in zip(axes, acid_columns):
    sns.histplot(
        data=df,
        x=column,
        kde=True,
        bins=25,
        ax=ax
    )
    ax.set_title(f"Distribution of {column}")

plt.tight_layout()
plt.show()

# Plot boxplots to observe the spread of acidity features
plt.figure(figsize=(10, 5))

sns.boxplot(data=df[acid_columns])

plt.title("Boxplots of Acidity Features")
plt.xlabel("Acidity Features")
plt.ylabel("Values")
plt.tight_layout()
plt.show()


# TASK 3 - CREATE A PANDAS SERIES FOR ACID TYPES BY QUALITY

print("\n" + "=" * 60)
print("TASK 3: PANDAS SERIES FOR ACID TYPES BY QUALITY")
print("=" * 60)

# Select the wine quality to analyse.
# Change 6 to another quality value if required.
quality_to_analyze = 6

# Filter the dataset for the selected quality
selected_wines = df[df["quality"] == quality_to_analyze]

if selected_wines.empty:
    print(f"\nNo records found for quality {quality_to_analyze}.")
else:
    # Calculate the average value of each acidity feature
    # and store the results in a Pandas Series.
    acid_series = selected_wines[acid_columns].mean()

    print(f"\nNumber of wine records with quality {quality_to_analyze}:")
    print(len(selected_wines))

    print(f"\nAverage acidity values for wine quality {quality_to_analyze}:")
    print(acid_series)

    print("\nPandas Series details:")
    print(type(acid_series))

    # Visualise the acidity values for the selected quality
    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=acid_series.index,
        y=acid_series.values
    )

    plt.title(f"Average Acidity Features for Wine Quality {quality_to_analyze}")
    plt.xlabel("Acidity Features")
    plt.ylabel("Average Value")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# Compare average acidity values across all quality levels
quality_acidity = df.groupby("quality")[acid_columns].mean()

print("\nAverage acidity values for each wine quality:")
print(quality_acidity.round(3))

print("\n" + "=" * 60)
print("ASSIGNMENT COMPLETED")
print("=" * 60)
