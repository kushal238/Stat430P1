# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Statistical and time series libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Machine learning libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# For GAM modeling (ensure pygam is installed; if not, run: pip install pygam)
from pygam import LinearGAM, s

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------
# Load the dataset (update the file path as needed)
file_path = "/content/fredupdated.csv"
df = pd.read_csv(file_path)

# Convert observation_date to datetime and sort the DataFrame
df['observation_date'] = pd.to_datetime(df['observation_date'])
df = df.sort_values(by='observation_date')

# Compute Inflation Rate as the year-over-year percentage change in CPIAUCSL
df['Inflation_Rate'] = df['CPIAUCSL'].pct_change(periods=12) * 100

# Define additional predictor columns available in the dataset
predictor_cols = ['UNRATE', 'GDP', 'PCE', 'FEDFUNDS', 'GS10', 
                  'M2SL', 'GDPC1', 'CIVPART', 'PPIACO', 'CES0500000003']

# Filter data from 1948 onward and drop rows missing either Inflation_Rate or any predictor
df = df[df['observation_date'] >= '1948-01-01'].reset_index(drop=True)
df_model = df.dropna(subset=['Inflation_Rate'] + predictor_cols).copy()

# ---------------------------
# 2. Visualizations
# ---------------------------
# Time Series Plot: Monthly Inflation Rate
plt.figure(figsize=(12, 6))
plt.plot(df['observation_date'], df['Inflation_Rate'], label='Monthly Inflation Rate', color='blue', alpha=0.7)
plt.title("Inflation Rate Over Time (1948 - 2025)")
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# 10-Year (120 months) Rolling Mean of Inflation Rate
df['Rolling_Inflation'] = df['Inflation_Rate'].rolling(window=120, min_periods=1).mean()
plt.figure(figsize=(12, 6))
plt.plot(df['observation_date'], df['Rolling_Inflation'], label='10-Year Rolling Mean', color='red', alpha=0.8)
plt.title("10-Year Rolling Mean of Inflation Rate")
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot: Unemployment vs. Inflation (Phillips Curve)
plt.figure(figsize=(8, 6))
plt.scatter(df_model['UNRATE'], df_model['Inflation_Rate'], alpha=0.5)
plt.title("Phillips Curve: Inflation vs. Unemployment")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Inflation Rate (%)")
plt.grid(True)
plt.show()

# ---------------------------
# 3. Regression & Prediction Models Using Multiple Predictors
# ---------------------------
# Prepare the multivariate predictors (X) and response variable (y)
X = df_model[predictor_cols].values   # All additional predictors including UNRATE
y = df_model['Inflation_Rate'].values  # Response: Inflation Rate

# --- Model 1: Multivariate Linear Regression (OLS) ---
X_sm = sm.add_constant(X)  # Add constant term for intercept
ols_model = sm.OLS(y, X_sm).fit()
print("Multivariate OLS Regression Summary:")
print(ols_model.summary())
print("\n")

# --- Model 2: Regularized Regression (Ridge and Lasso) ---
# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_scaled, y)
y_pred_ridge = ridge_reg.predict(X_scaled)
print("Ridge Regression Model:")
print("Coefficients:", ridge_reg.coef_)
print("Intercept:", ridge_reg.intercept_)
print("R2 Score:", r2_score(y, y_pred_ridge))
print("MSE:", mean_squared_error(y, y_pred_ridge))
print("\n")

# Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_scaled, y)
y_pred_lasso = lasso_reg.predict(X_scaled)
print("Lasso Regression Model:")
print("Coefficients:", lasso_reg.coef_)
print("Intercept:", lasso_reg.intercept_)
print("R2 Score:", r2_score(y, y_pred_lasso))
print("MSE:", mean_squared_error(y, y_pred_lasso))
print("\n")

# --- Model 3: Polynomial Regression (Degree 2) ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
print("Polynomial Regression (Degree 2) Model:")
print("Coefficients:", poly_reg.coef_)
print("Intercept:", poly_reg.intercept_)
print("R2 Score:", r2_score(y, y_pred_poly))
print("MSE:", mean_squared_error(y, y_pred_poly))
print("\n")

# Plotting predictions of the polynomial model against actual inflation
plt.figure(figsize=(8,6))
plt.scatter(range(len(y)), y, alpha=0.5, label="Actual Inflation")
plt.plot(range(len(y)), y_pred_poly, color='red', label="Polynomial Predictions")
plt.title("Polynomial Regression (Degree 2): Actual vs. Predicted Inflation")
plt.xlabel("Observation Index")
plt.ylabel("Inflation Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# --- Model 4: ARIMA Model for Inflation (Univariate) ---
# ARIMA is applied on the full inflation time series, independent of the predictors.
inflation_ts = df.set_index('observation_date')['Inflation_Rate'].dropna()
# Use an ARIMA(1,1,1) model for demonstration (optimal orders can be selected using AIC/BIC)
arima_model = ARIMA(inflation_ts, order=(1, 1, 1))
arima_result = arima_model.fit()
print("ARIMA(1,1,1) Model Summary:")
print(arima_result.summary())
print("\n")

# Plot ARIMA fitted values vs. actual inflation rate
plt.figure(figsize=(12, 6))
plt.plot(inflation_ts.index, inflation_ts, label="Actual Inflation")
plt.plot(inflation_ts.index, arima_result.fittedvalues, color='red', label="ARIMA Fitted")
plt.title("ARIMA(1,1,1): Actual vs. Fitted Inflation")
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# --- Model 5: Random Forest Regression ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
y_pred_rf = rf_model.predict(X)
print("Random Forest Regression:")
print("R2 Score:", r2_score(y, y_pred_rf))
print("MSE:", mean_squared_error(y, y_pred_rf))
print("\n")

# --- Model 6: k-Nearest Neighbors Regression ---
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X, y)
y_pred_knn = knn_model.predict(X)
print("k-Nearest Neighbors Regression (k=5):")
print("R2 Score:", r2_score(y, y_pred_knn))
print("MSE:", mean_squared_error(y, y_pred_knn))
print("\n")

# --- Model 7: Generalized Additive Model (GAM) ---
# Build a GAM with a smoothing spline term for each predictor variable.
# Build GAM terms by starting with the first term and adding the others
gam_terms = s(0)
for i in range(1, X.shape[1]):
    gam_terms += s(i)

gam = LinearGAM(gam_terms).fit(X, y)
y_pred_gam = gam.predict(X)
print("Generalized Additive Model (GAM):")
print("R2 Score:", r2_score(y, y_pred_gam))
print("MSE:", mean_squared_error(y, y_pred_gam))
gam = LinearGAM(gam_terms).fit(X, y)
y_pred_gam = gam.predict(X)
print("Generalized Additive Model (GAM):")
print("R2 Score:", r2_score(y, y_pred_gam))
print("MSE:", mean_squared_error(y, y_pred_gam))
print("\n")

# Plot GAM predictions vs. actual inflation
plt.figure(figsize=(8,6))
plt.scatter(range(len(y)), y, alpha=0.5, label="Actual Inflation")
plt.plot(range(len(y)), y_pred_gam, color='green', label="GAM Predictions")
plt.title("GAM: Actual vs. Predicted Inflation")
plt.xlabel("Observation Index")
plt.ylabel("Inflation Rate (%)")
plt.legend()
plt.grid(True)
plt.show()
