import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Modeling libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Time series modeling
import statsmodels.api as sm
import statsmodels.tsa.api as tsa

# For Generalized Additive Model (GAM)
from pygam import LinearGAM, s

# ------------------------------
# Data Preprocessing
# ------------------------------
# Load the dataset
df = pd.read_csv("fredupdated.csv")

# Convert 'observation_date' to datetime and sort the DataFrame
df['observation_date'] = pd.to_datetime(df['observation_date'])
df = df.sort_values(by='observation_date')

# Compute the Inflation Rate as the year-over-year percentage change in CPIAUCSL
df['Inflation_Rate'] = df['CPIAUCSL'].pct_change(periods=12) * 100

# Filter the data from 1948 onward (assuming data reliability from 1948 for unemployment and inflation)
df_filtered = df[df['observation_date'] >= '1948-01-01'].copy()

# ------------------------------
# Visualization of Inflation Data
# ------------------------------
# Plot the Inflation Rate over time
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['observation_date'], df_filtered['Inflation_Rate'], label='Inflation Rate', color='blue', alpha=0.7)
plt.title("Inflation Rate Over Time (1948 - 2025)")
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# Compute a 10-year (approximately 120-month) rolling mean of the Inflation Rate to capture long-term trends
df_filtered['Rolling_Inflation'] = df_filtered['Inflation_Rate'].rolling(window=120, min_periods=1).mean()

# Plot the Rolling Mean of Inflation Rate
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['observation_date'], df_filtered['Rolling_Inflation'], label='10-Year Rolling Mean', color='red', alpha=0.8)
plt.title("10-Year Rolling Mean of Inflation Rate")
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Prepare Data for Modeling
# ------------------------------
# We use UNRATE as the predictor and Inflation_Rate as the response.
# Drop rows with missing values in either column.
model_data = df_filtered[['observation_date', 'UNRATE', 'Inflation_Rate']].dropna()

# Create the feature matrix X and target vector y
X = model_data[['UNRATE']].values
y = model_data['Inflation_Rate'].values

# Split the data into training and testing sets (here, we avoid shuffling to maintain time series order)
split_index = int(len(model_data) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ------------------------------
# 1. Linear Regression Model (OLS)
# ------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression RMSE:", rmse_lr)

# ------------------------------
# 2. Ridge Regression (with Hyperparameter Tuning)
# ------------------------------
ridge = Ridge()
params_ridge = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(ridge, params_ridge, cv=5)
ridge_cv.fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print("Ridge Regression RMSE:", rmse_ridge, "Best alpha:", ridge_cv.best_params_['alpha'])

# ------------------------------
# 3. Lasso Regression (with Hyperparameter Tuning)
# ------------------------------
lasso = Lasso(max_iter=10000)
params_lasso = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
lasso_cv = GridSearchCV(lasso, params_lasso, cv=5)
lasso_cv.fit(X_train, y_train)
y_pred_lasso = lasso_cv.predict(X_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print("Lasso Regression RMSE:", rmse_lasso, "Best alpha:", lasso_cv.best_params_['alpha'])

# ------------------------------
# 4. Polynomial Regression Model (Degree 2)
# ------------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_pred_poly = poly_lr.predict(X_test_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression (Degree 2) RMSE:", rmse_poly)

# ------------------------------
# 5. ARIMA Model on Inflation Rate (Time Series Model)
# ------------------------------
# For ARIMA, we work with the inflation time series directly.
ts_data = df_filtered.set_index('observation_date')['Inflation_Rate'].dropna()

# Fit an ARIMA model; here we use order (1, 0, 1) as an example. In practice, you would tune this.
model_arima = tsa.ARIMA(ts_data, order=(1, 0, 1))
arima_result = model_arima.fit()
print(arima_result.summary())

# ------------------------------
# 6. Random Forest Regression Model
# ------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest RMSE:", rmse_rf)

# ------------------------------
# 7. k-Nearest Neighbors (k-NN) Regression Model
# ------------------------------
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
print("k-NN Regression RMSE:", rmse_knn)

# ------------------------------
# 8. Generalized Additive Model (GAM)
# ------------------------------
# Using a smoothing spline for the UNRATE predictor.
gam = LinearGAM(s(0)).fit(X_train, y_train)
y_pred_gam = gam.predict(X_test)
rmse_gam = np.sqrt(mean_squared_error(y_test, y_pred_gam))
print("GAM RMSE:", rmse_gam)

# ------------------------------
# Plotting Predictions vs. Actual Inflation Rate
# ------------------------------
plt.figure(figsize=(12, 6))
# Use the test period's dates from the model_data (the last 20% of the data)
test_dates = model_data['observation_date'].iloc[split_index:]
plt.plot(test_dates, y_test, label='Actual Inflation', color='black')
plt.plot(test_dates, y_pred_lr, label='Linear Regression Predictions', color='blue')
plt.plot(test_dates, y_pred_gam, label='GAM Predictions', color='green')
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.title("Inflation Rate: Actual vs. Predictions")
plt.legend()
plt.grid(True)
plt.show()
