# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = sns.load_dataset("iris")

# Display first few records
print("Sample Data:")
print(df.head(10))

# Check for null values
print("\nChecking for Null Values:")
print(df.isnull().sum())

# Dataset shape and types
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Display value counts for target
print("\nSpecies Count:")
print(df['species'].value_counts())

# Describe dataset
print("\nStatistical Summary:")
print(df.describe())

# Unique species list
print("\nUnique Species:", df['species'].unique())

# Correlation Matrix
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot of dataset
sns.pairplot(df, hue="species", palette="Set2")
plt.suptitle("Iris Features vs Species", y=1.02)
plt.show()

# Boxplot for each feature
plt.figure(figsize=(12, 6))
for i, column in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=column, data=df, palette='Set3')
    plt.title(f"{column} by Species")
plt.tight_layout()
plt.show()

# Prepare input and target
X = df.drop("species", axis=1)
y = df["species"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Function to evaluate and print results
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Logistic Regression
lr_model = LogisticRegression(max_iter=200)
evaluate_model(lr_model, "Logistic Regression")

# Support Vector Classifier
svc_model = SVC()
evaluate_model(svc_model, "Support Vector Machine")

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn_model, "K-Nearest Neighbors")

# Save one model's prediction results
final_predictions = pd.DataFrame({
    "SepalLength": X_test['sepal_length'],
    "SepalWidth": X_test['sepal_width'],
    "PetalLength": X_test['petal_length'],
    "PetalWidth": X_test['petal_width'],
    "Actual Species": y_test.values,
    "Predicted Species": lr_model.predict(X_test)
})

# Save to CSV
final_predictions.to_csv("Iris_Final_Predictions.csv", index=False)
print("\nFinal predictions saved as 'Iris_Final_Predictions.csv'")

# Completion message
print("\nIris Flower Classification Task Completed Successfully ✅")
