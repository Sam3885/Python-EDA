import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("33_Constituency_Wise_Detailed_Result.csv")
print("First 5 Rows:")
print(df.head())
print("\nLast 5 Rows:")
print(df.tail())
print("\nShape of Dataset:", df.shape)
print("\nData Types and Non-Null Counts:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))

print("\nColumn Names:")
print(df.columns.tolist())
print("\nMissing Values:")
print(df.isnull().sum())

# Visualizing missing value plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap="YlGnBu", cbar=False)
plt.title("Missing Values Heatmap")
plt.show()
# Fill 'Gender' and 'Category' with mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Category'].fillna(df['Category'].mode()[0], inplace=True)

# Fill 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Confirm no missing values remain
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Gender Distribution")
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], kde=True, bins=20)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
