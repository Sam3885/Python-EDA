import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load dataset
df = pd.read_csv("33_Constituency_Wise_Detailed_Result.csv")

# Fix: Remove leading/trailing spaces from all column names
df.columns = df.columns.str.strip()

# Now define correct column names (no space needed at end)
x = 'Total Votes Polled In The Constituency'
y = 'Total Electors'

# Scatter Plot with Regression Line
plt.figure(figsize=(8, 5))
sns.regplot(x=df[x], y=df[y], scatter_kws={'color':'blue'}, line_kws={"color":"red"})
plt.title(f"Scatter Plot with Linear Regression ({x} vs {y})")
plt.xlabel(x)
plt.ylabel(y)
plt.grid(True)
plt.show()

# Pearson Correlation Coefficient
correlation, p_value = pearsonr(df[x].dropna(), df[y].dropna())
print(f"Pearson Correlation Coefficient between {x} and {y}: {correlation:.3f}")
print(f"P-value: {p_value:.5f}")
