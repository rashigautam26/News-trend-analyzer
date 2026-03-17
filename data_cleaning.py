# 
import pandas as pd

print("Reading raw data...")

df = pd.read_csv("news_raw.csv")

print("Before Cleaning:", len(df))

# Remove null rows
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("After Cleaning:", len(df))

df.to_csv("news_cleaned.csv", index=False)

print("Cleaned data saved to news_cleaned.csv")

# Analysis
print("\n--- Analysis ---")
print("Total News Articles:", len(df))
print("Unique News Sources:", df["source"].nunique())

print("\n--- Top 5 News ---")
print(df.head())