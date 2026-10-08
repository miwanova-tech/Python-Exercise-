import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(iris.head())

print("\nSummary Statistics:")
print(iris.describe())

print("\nDataset Information:")
print(iris.info())

print("\nMissing Values:")
print(iris.isnull().sum())

iris.drop(columns=["Id"], errors="ignore").hist(figsize=(10, 8))
plt.suptitle("Distribution of Iris Features")
plt.tight_layout()
plt.show()

sns.scatterplot(data=iris, x="SepalLengthCm", y="SepalWidthCm", hue="Species")
plt.title("Sepal Length vs Sepal Width")
plt.show()

sns.boxplot(data=iris, x="Species", y="PetalLengthCm")
plt.title("Petal Length by Species")
plt.show()

sns.pairplot(iris.drop(columns=["Id"], errors="ignore"), hue="Species")
plt.show()

correlation = iris.select_dtypes("number").drop(columns=["Id"], errors="ignore").corr()
print("\nCorrelation Matrix:")
print(correlation)

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Iris Features")
plt.tight_layout()
plt.show()
