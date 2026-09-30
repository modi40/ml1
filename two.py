import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("iris.csv")

print("Original Iris Dataset Categorical Values:")
print(df[['sepal_length', 'species']].head())

species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['species_numeric'] = df['species'].map(species_mapping)

print("\nConverted Categorical Species into Numeric Format:")
print(df[['species', 'species_numeric']].drop_duplicates())

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df['sepal_length'],
    df['sepal_width'],
    c=df['species_numeric'],
    cmap='viridis',
    edgecolor='k'
)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset Scatter Plot')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species (0: Setosa, 1: Versicolor, 2: Virginica)')
plt.savefig('iris_scatter.png')
plt.show()
print("\nScatter plot saved as iris_scatter.png and displayed.")
