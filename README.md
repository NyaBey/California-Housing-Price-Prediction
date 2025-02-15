# import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
housing_data = pd.read_csv(r"C:\Users\yunia\OneDrive\Dokumente\DS PROJECTS PY\California Housing Price Pred.zip")  
# Ensure the correct file path

# Check for missing values
missing_values = housing_data.isnull().sum()

# Generate summary statistics
summary_stats = housing_data.describe()

# Visualizing missing values
plt.figure(figsize=(8, 4))
sns.heatmap(housing_data.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Values in the Dataset")
plt.show()

# Display the summary statistics
print(housing_data.describe())  # Prints summary stats in the terminal

# Show missing values count
missing_values

# Clean and prepare dataset

from sklearn.preprocessing import LabelEncoder

# Fill missing values in 'total_bedrooms' with median
housing_data['total_bedrooms'].fillna(housing_data['total_bedrooms'].median(), inplace=True)

# Convert categorical 'ocean_proximity' into numerical values
label_encoder = LabelEncoder()
housing_data['ocean_proximity'] = label_encoder.fit_transform(housing_data['ocean_proximity'])

# Confirm changes
housing_data.info(), housing_data.head()

print("Preprocessing done! Here's the first 5 rows:")
print(housing_data.head())  # Show first 5 rows
print(housing_data.isnull().sum())  # Show missing values count

from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = housing_data.drop(columns=["median_house_value"])  # All columns except target
y = housing_data["median_house_value"]  # Target variable

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm split sizes
X_train.shape, X_test.shape, y_train.shape, y_test.shape

print("Script started...")  

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize and train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lin_reg.predict(X_test)

# Train the model
lin_reg.fit(X_train, y_train)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Get model coefficients
lin_reg.coef_, lin_reg.intercept_

print("📊 Model Evaluation Metrics:")
print(f"✔️ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✔️ Mean Squared Error (MSE): {mse:.2f}")
print(f"✔️ R² Score: {r2:.4f}")  # Higher is better, ranges from 0 to 1

from sklearn.ensemble import RandomForestRegressor

# Initialize and train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n🌳 Random Forest Model Metrics:")
print(f"✔️ MAE: {rf_mae:.2f}")
print(f"✔️ MSE: {rf_mse:.2f}")
print(f"✔️ R² Score: {rf_r2:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
feature_importance = rf_model.feature_importances_
features = X_train.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
