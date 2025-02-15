# California-Housing-Price-Prediction

## Project Overview
A machine learning model to predict **housing prices in California** using **Random Forest**.

This project aims to predict **housing prices in California** using **Machine Learning** techniques. The dataset is sourced from **Kaggle** and includes features like **location, number of rooms, population, median income, and proximity to the ocean**.

We built multiple models, including **Linear Regression** and **Random Forest**, and found that **Random Forest** performed significantly better with an **R² score of 0.8075**.

---

## Project Structure


Dataset
The dataset contains **20,640 records** with the following key features:

- **`longitude`, `latitude`** → Geographic location  
- **`housing_median_age`** → Age of the house  
- **`total_rooms`, `total_bedrooms`** → House size  
- **`population`, `households`** → Area demographics  
- **`median_income`** → Economic factor  
- **`ocean_proximity`** → Categorical location feature  
- **`median_house_value`** → **Target variable (House Price)** 
---

## The model Performance:

### **Linear Regression:** 
- **Mean Absolute Error (MAE):** **\$51,820.75**  
- **Mean Squared Error (MSE):** **5,062,019,613.46**  
- **R² Score:** **0.6137** 

### **Random Forest Model Metrics:**
- **Mean Absolute Error (MAE):** **\$32,156.73**  
- **Mean Squared Error (MSE):** **2,523,161,074.32**  
- **R² Score:** **0.8075**  
**The Random Forest outperforms LR** by having **a lower error margin** and **a higher accuracy**.
---

**Usage requires:...

### installation of the dependencies
    ```bash
pip install -r requirements.txt 

 python scripts/preprocess.py
 
python scripts/train_model.py
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/housing_price_model.pkl")

# Sample input
sample_data = pd.DataFrame({
    "longitude": [-122.23],
    "latitude": [37.88],
    "housing_median_age": [41],
    "total_rooms": [880],
    "total_bedrooms": [129],
    "population": [322],
    "households": [126],
    "median_income": [8.32],
    "ocean_proximity": [3]
})

# Predict
prediction = model.predict(sample_data)
print("Predicted House Price:", prediction[0])

## Followed by:
Feature Engineering: Add more meaningful features
Hyperparameter Tuning: Further optimize the Random Forest model
Deploy the Model: Create a Flask or Streamlit app

 License
 
---

