#importing necessary libraries for data manipulation, analysis, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Loading the dataset into a pandas DataFrame
file_path = r'C:\Users\Mohit Singh\Desktop\DSCodSoftInternship\Task2\IMDb-Movies-India.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

#Exploring the dataset to understand its structure and contents
print("First 5 rows of the dataframe:")
print(df.head())  # shows first 5 entries of rows

print("\nInformation about the dataframe:")
print(df.info()) 

print("\nDescriptive statistics of the dataframe:")
print(df.describe())

# Checking the actual column names
print("\nColumn names in the DataFrame:")
print(df.columns)

#Data Preprocessing
##Preprocessing the data to handle missing values, encode categorical variables, and scale numerical features if necessary
###Handling Missing Values
print("\nMissing values in each column:")
print(df.isnull().sum())  # Check for missing values

df.dropna(inplace=True)  # Drop rows with missing values (or we can fill them with appropriate values)
print("\nMissing values after dropping rows:")
print(df.isnull().sum())  # Check for missing values again

###Encoding Categorical Variables
# Encode categorical variables like genre, director, and actors using one-hot encoding
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
numerical_features = ['Duration']  # Example numerical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Printing unique values of categorical columns to understand the data better
print("\nUnique values in 'Genre':")
print(df['Genre'].unique())
print("\nUnique values in 'Director':")
print(df['Director'].unique())
print("\nUnique values in 'Actor 1':")
print(df['Actor 1'].unique())
print("\nUnique values in 'Actor 2':")
print(df['Actor 2'].unique())
print("\nUnique values in 'Actor 3':")
print(df['Actor 3'].unique())

# Feature Engineering
# Create or transform features if needed. For example, we can create a feature that counts the number of actors
df['num_actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].notna().sum(axis=1)

# Ensure numerical columns are of numeric type
df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')

# Ensure target column exists and is of numeric type
if 'Rating' not in df.columns:
    raise ValueError("The 'Rating' column is not present in the dataset.")
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Splitting the data into training and testing sets
X = df.drop('Rating', axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if there are any remaining missing values after dropping/handling
if df.isnull().sum().any():
    raise ValueError("There are still missing values in the data.")
# Building a regression model using a pipeline to streamline preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)

# Evaluating the model using metrics like Mean Squared Error (MSE) and R-squared (R²)
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Interpreting the results and considering ways to improve the model, such as using more sophisticated models like Random Forest, Gradient Boosting, or neural networks



