# ----------------------------------------------
# Customer Purchase Behavior Analysis & Forecasting
# ----------------------------------------------

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv('superstore.csv', encoding='latin1')

# --- Data Cleaning ---
df.dropna(inplace=True)  # remove missing values
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Order Date'])  # drop rows with invalid dates
df = df.sort_values('Order Date')

# --- Exploratory Data Analysis ---
print("\nTop 5 rows:")
print(df.head())

# Monthly sales aggregation
monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum()

# Top selling categories
top_categories = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print("\nTop Categories:\n", top_categories)

# Visualization - Sales Trend
plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# Visualization - Category-wise sales
plt.figure(figsize=(7, 5))
sns.barplot(x=top_categories.index, y=top_categories.values, palette="viridis")
plt.title('Total Sales by Category')
plt.ylabel('Sales')
plt.show()

# --- Forecasting using ARIMA ---
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales[0:train_size], monthly_sales[train_size:]

model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

# Evaluation
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"\nRMSE: {rmse}")

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.legend()
plt.title('ARIMA Sales Forecast')
plt.show()
