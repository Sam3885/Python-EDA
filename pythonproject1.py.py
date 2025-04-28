import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("33_Constituency_Wise_Detailed_Result.csv")
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# Histograms
df[numerical_cols].hist(figsize=(15, 10), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Boxplots
for col in numerical_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

for cat in categorical_cols:
    for num in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[cat], y=df[num])
        plt.title(f"{num} by {cat}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
