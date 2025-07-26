# sales_prediction.py

# 📌 Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 📌 Step 2: Load the dataset
data = pd.read_csv("advertising.csv")  # Make sure this file is in the same folder
print("✅ Dataset Loaded Successfully!\n")

# 📌 Step 3: Display first few rows
print("🔎 First 5 rows of the dataset:")
print(data.head(), "\n")

# 📌 Step 4: Check for missing values
print("❓ Checking for missing values:")
print(data.isnull().sum(), "\n")

# 📌 Step 5: Data Visualization - TV vs Sales
print("📊 Showing scatter plot of TV Budget vs Sales")
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='TV', y='Sales', color='purple')
plt.title("TV Ad Budget vs Sales")
plt.xlabel("TV Advertisement Budget ($)")
plt.ylabel("Sales (in millions)")
plt.show()

# 📌 Step 6: Feature and Target selection
X = data[['TV']]  # Feature (TV budget)
y = data['Sales'] # Target (Sales)

# 📌 Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 8: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model Trained Successfully!\n")

# 📌 Step 9: Predict on test set
y_pred = model.predict(X_test)

# 📌 Step 10: Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"📈 R² Score: {r2:.4f}")
print(f"📉 Mean Squared Error: {mse:.4f}\n")

# 📌 Step 11: Visualization - Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Sales')
plt.xlabel('TV Ad Budget')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.grid(True)
plt.show()

print("✅ Sales Prediction Task Completed Successfully!")
