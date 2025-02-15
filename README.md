# California-Housing-Price-Prediction
A machine learning model to predict housing prices using Random Forest
This project aims to predict housing prices in California using Machine Learning techniques. The dataset is sourced from Kaggle and includes features like location, number of rooms, population, median income, and proximity to the ocean.

We built multiple models, including Linear Regression and Random Forest, and found that Random Forest performed significantly better with an R² score of 0.8075.
The structure
California-Housing-Price-Prediction/
│-- data/
│   ├── housing.csv   # Raw dataset
│-- models/
│   ├── housing_price_model.pkl   # Saved trained model
│-- notebooks/
│   ├── EDA.ipynb    # Exploratory Data Analysis
│   ├── Model_Training.ipynb    # Model training & evaluation
│-- scripts/
│   ├── preprocess.py    # Data preprocessing script
│   ├── train_model.py    # Model training script
│-- README.md    # Project documentation

Dataset
The dataset contains 20,640 records with the following key features:

longitude, latitude → Geographic location

housing_median_age → Age of the house

total_rooms, total_bedrooms → House size

population, households → Area demographics

median_income → Economic factor

ocean_proximity → Categorical location feature

median_house_value → Target variable (House Price)
The model Performance
Linear Regression 
Model Evaluation Metrics:
 Mean Absolute Error (MAE): 51820.75
 Mean Squared Error (MSE): 5062019613.46
R² Score: 0.6137

Random Forest Model Metrics:
 MAE: 32156.73
 MSE: 2523161074.32
 R² Score: 0.8075
 Random Forest outperforms LR model by having a lower error margin and a higher accuracy

 Usage requires
 1. installation of the dependencies
 pip install -r requirements.txt 
 2. run data preprocessing
    python scripts/preprocess.py
3. Train the model
4. Make predictions
5. Evaluate and visualize
 6. 
 
