# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(url, header=None, names=column_names)

# Step 3: Explore the dataset
print(iris_df.head())  # Display the first few rows of the dataset
print(iris_df.info())  # Display information about the dataset

# Step 4: Perform basic statistical analysis
print(iris_df.describe())

# Step 5: Create visualizations
# Pairplot to visualize relationships between features
sns.pairplot(iris_df, hue='class')
plt.show()

# Boxplot to show the distribution of each feature for each class
plt.figure(figsize=(12, 8))
for i, feature in enumerate(column_names[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='class', y=feature, data=iris_df)
    plt.title(f'Distribution of {feature}')
plt.show()
