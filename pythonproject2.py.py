import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("33_Constituency_Wise_Detailed_Result.csv")

print("Shape of dataset:", df.shape)
print("Data types:\n", df.dtypes)
print("Numerical columns:\n", df.select_dtypes(include=np.number).columns.tolist())

Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_filtered = df[(df['Age'] >= lower) & (df['Age'] <= upper)]

df['Age'] = np.where(df['Age'] < lower, lower,
              np.where(df['Age'] > upper, upper, df['Age']))

df['Log_Age'] = np.log1p(df['Age'])

plt.figure(figsize=(6, 1.5))
sns.boxplot(x=df_filtered['Age'])
plt.title("Boxplot Before Handling Outliers in Age")
plt.show()

plt.figure(figsize=(6, 1.5))
sns.boxplot(x=df['Age'])
plt.title("Boxplot After Handling Outliers in Age")
plt.show()

plt.figure(figsize=(6, 1.5))
sns.boxplot(x=df['Log_Age'])
plt.title("Boxplot of Log-Transformed Age")
plt.show()

print("Skewness Before (filtered):", df_filtered['Age'].skew())
print("Skewness After (capped):", df['Age'].skew())
print("Skewness After Log Transform:", df['Log_Age'].skew())
