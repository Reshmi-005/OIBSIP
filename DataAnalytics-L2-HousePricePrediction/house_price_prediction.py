"""
Task 1: Predicting House Prices with Linear Regression & Regularization
Track: Data Analytics (Level 2) - OIBSIP
Author: Data Analytics Intern
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "house_prices_dataset.csv")
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 70)
    print("TASK 1 · PREDICTING HOUSE PRICES WITH LINEAR REGRESSION")
    print("=" * 70)

    # 1. Ingestion & Exploratory Data Analysis
    print("\n[1] DATA INGESTION & INTEGRITY AUDIT")
    df = pd.read_csv(data_path)
    print(f"- Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nMissing Values Audit:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "No missing values.")

    print("\nDescriptive Summary of Numerical Features:")
    numeric_cols = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Floors', 'House_Age', 'Garage_Capacity', 'Distance_to_City_Center_km', 'Price']
    print(df[numeric_cols].describe().round(2).to_string())

    # Target variable distribution
    plt.figure(figsize=(9, 4.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    sns.histplot(df['Price'], kde=True, color='#2980b9', edgecolor='black', alpha=0.6)
    plt.axvline(df['Price'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: ${df['Price'].mean():,.0f}")
    plt.axvline(df['Price'].median(), color='green', linestyle=':', linewidth=1.5, label=f"Median: ${df['Price'].median():,.0f}")
    plt.title("Target Variable Distribution: House Sale Price ($)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Sale Price ($)", fontweight='bold')
    plt.ylabel("Number of Houses", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    chart1_path = os.path.join(images_dir, "01_price_distribution.png")
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"\nSaved price distribution plot to: {chart1_path}")

    # 2. Missing Value Imputation & Feature Preprocessing
    print("\n[2] MISSING VALUE HANDLING & ONE-HOT ENCODING")
    # Impute missing values
    df['Garage_Capacity'] = df['Garage_Capacity'].fillna(df['Garage_Capacity'].median())
    df['Overall_Condition'] = df['Overall_Condition'].fillna(df['Overall_Condition'].mode()[0])

    # Feature selection & One-Hot Encoding
    features = [
        'Square_Feet', 'Bedrooms', 'Bathrooms', 'Floors', 'House_Age',
        'Garage_Capacity', 'Has_Pool', 'Distance_to_City_Center_km',
        'Neighborhood', 'Overall_Condition'
    ]
    X_raw = df[features]
    y = df['Price']

    # One-hot encode categoricals with drop_first=True to avoid multicollinearity
    X = pd.get_dummies(X_raw, columns=['Neighborhood', 'Overall_Condition'], drop_first=True, dtype=float)
    print(f"- Processed Feature Matrix Shape: {X.shape[0]:,} rows × {X.shape[1]} features")

    # 3. Correlation Heatmap
    print("\n[3] GENERATING CORRELATION MATRIX HEATMAP...")
    plt.figure(figsize=(11, 7.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    
    corr_df = X.copy()
    corr_df['Price'] = y
    corr_matrix = corr_df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7.5})
    plt.title("Correlation Matrix of Housing Predictors & Price", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    chart2_path = os.path.join(images_dir, "02_correlation_heatmap.png")
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to: {chart2_path}")

    # 4. Train/Test Split (80/20)
    print("\n[4] TRAIN / TEST PARTITIONING (80% TRAIN, 20% TEST)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"- Training set: {len(X_train):,} houses | Testing set: {len(X_test):,} houses")

    # 5. Train Linear Regression (OLS) Model
    print("\n[5] TRAINING ORDINARY LEAST SQUARES (OLS) LINEAR REGRESSION")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # 6. Evaluation Metrics: MSE, RMSE, R²
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print("\nLinear Regression (OLS) Performance Metrics:")
    print(f"  * Mean Squared Error (MSE) : ${mse_lr:,.2f}")
    print(f"  * Root Mean Squared Error  : ${rmse_lr:,.2f}")
    print(f"  * R² Score (Variance Exp.) : {r2_lr:.4f} ({r2_lr*100:.2f}%)")

    # 7. Diagnostic Plots: Actual vs Predicted & Residuals
    print("\n[7] GENERATING DIAGNOSTIC RESIDUAL & ACCURACY PLOTS...")
    
    # Scatter: Actual vs Predicted
    plt.figure(figsize=(7.5, 6))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    plt.scatter(y_test, y_pred_lr, color='#2980b9', alpha=0.6, edgecolors='black', linewidth=0.5, s=40)
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Prediction ($y=x$)')
    plt.title("Actual vs. Predicted House Prices", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Actual Price ($)", fontweight='bold')
    plt.ylabel("Predicted Price ($)", fontweight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    chart3_path = os.path.join(images_dir, "03_actual_vs_predicted_prices.png")
    plt.savefig(chart3_path, dpi=150)
    plt.close()
    print(f"Saved actual vs. predicted plot to: {chart3_path}")

    # Residual Plot
    residuals = y_test - y_pred_lr
    plt.figure(figsize=(9, 4.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    plt.scatter(y_pred_lr, residuals, color='#8e44ad', alpha=0.6, edgecolors='black', linewidth=0.5, s=35)
    plt.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.8, label='Zero Residual Line')
    plt.title("Residuals vs. Fitted Values (Homoscedasticity Check)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Fitted (Predicted) Price ($)", fontweight='bold')
    plt.ylabel("Residuals ($: Actual - Predicted)", fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    chart4_path = os.path.join(images_dir, "04_residuals_plot.png")
    plt.savefig(chart4_path, dpi=150)
    plt.close()
    print(f"Saved residual plot to: {chart4_path}")

    # 8. Coefficient Impact Analysis
    print("\n[8] FEATURE COEFFICIENT ANALYSIS")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr.coef_
    }).sort_values(by='Coefficient', ascending=True)

    print(coef_df.round(2).to_string(index=False))

    plt.figure(figsize=(10, 5.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    colors = ['#e74c3c' if c < 0 else '#27ae60' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black', linewidth=0.5)
    plt.title("Linear Regression Coefficients (Dollar Impact on House Price)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Coefficient Value ($ per unit increase)", fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    chart5_path = os.path.join(images_dir, "05_feature_coefficients.png")
    plt.savefig(chart5_path, dpi=150)
    plt.close()
    print(f"Saved coefficient plot to: {chart5_path}")

    # 9. Regularization Comparison: Ridge & Lasso (Bonus)
    print("\n[9] BONUS: BENCHMARKING AGAINST RIDGE & LASSO REGULARIZATION")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)

    lasso = Lasso(alpha=100.0, max_iter=2000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    r2_lasso = r2_score(y_test, y_pred_lasso)

    comparison_results = pd.DataFrame([
        {"Model": "Linear Regression (OLS)", "RMSE ($)": round(rmse_lr, 2), "R² Score": round(r2_lr, 4), "Notes": "Standard baseline without penalty"},
        {"Model": "Ridge Regression (L2)", "RMSE ($)": round(rmse_ridge, 2), "R² Score": round(r2_ridge, 4), "Notes": "Shrinks collinear coefficients"},
        {"Model": "Lasso Regression (L1)", "RMSE ($)": round(rmse_lasso, 2), "R² Score": round(r2_lasso, 4), "Notes": "Sparse solution with feature selection"}
    ])
    print(comparison_results.to_string(index=False))

    # Save Comparison Chart
    plt.figure(figsize=(8, 4.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    x_pos = np.arange(len(comparison_results))
    bars = plt.bar(x_pos, comparison_results['R² Score'], color=['#3498db', '#2ecc71', '#9b59b6'], edgecolor='black', linewidth=0.6, width=0.5)
    plt.xticks(x_pos, comparison_results['Model'], fontweight='bold')
    plt.ylim(0.85, 1.0)
    plt.title("Model Generalization Comparison: OLS vs. Ridge vs. Lasso", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("R² Score (Variance Explained)", fontweight='bold')
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 0.003, f"{h:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    chart6_path = os.path.join(images_dir, "06_ols_ridge_lasso_comparison.png")
    plt.savefig(chart6_path, dpi=150)
    plt.close()
    print(f"Saved model comparison chart to: {chart6_path}")

    print("\n" + "=" * 70)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
