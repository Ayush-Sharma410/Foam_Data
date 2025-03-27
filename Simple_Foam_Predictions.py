"""
Simple Foam Data Prediction Model

This script provides a simplified approach to predicting foam properties using only
the original features without additional feature engineering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import warnings
import os
import datetime
import json
from sklearn.base import clone
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
results_dir = 'basic_foam_predictions'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create separate folders for images and CSV files
images_dir = os.path.join(results_dir, 'visualizations')
csv_dir = os.path.join(results_dir, 'data')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Create a timestamped report file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"{results_dir}/foam_prediction_report_{timestamp}.txt"
report_file = open(report_filename, 'w')

# Function to write to both console and report file
def log(message):
    print(message)
    report_file.write(message + "\n")

# Set better plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the data
log("Loading data...")
df = pd.read_csv("Foam Data.csv")

# 2. Basic data exploration
log("\nData Overview:")
log(f"Shape: {df.shape}")
log("\nData Types:")
log(str(df.dtypes))
log("\nSummary Statistics:")
log(str(df.describe()))
log("\nFirst few rows:")
log(str(df.head()))

# Save the initial exploration as CSV
df.describe().to_csv(f"{csv_dir}/data_summary_stats.csv")

# 3. Simple Feature Processing (just one-hot encoding categorical variables)
log("\nProcessing features (one-hot encoding only)...")

# One-hot encode categorical variables
categorical_cols = ['Techniques', 'Composition']
encoded_cols = pd.get_dummies(df[categorical_cols], drop_first=False)

# Keep Passes as a numeric feature
numeric_features = ['Passes']

# Combine features
X = pd.concat([df[numeric_features], encoded_cols], axis=1)
y = df['R1']

# Save the processed dataset
X.to_csv(f"{csv_dir}/processed_features.csv", index=False)
pd.DataFrame({'R1': y}).to_csv(f"{csv_dir}/target.csv", index=False)

# 4. Basic correlation analysis
log("\nAnalyzing feature correlations...")
# Calculate correlations with target for numeric features
if len(numeric_features) > 0:
    corr_with_target = df[numeric_features + ['R1']].corr()['R1'].abs().sort_values(ascending=False)
    log("\nFeature correlation with target variable (R1):")
    log(str(corr_with_target))
    
    # Save correlations to CSV
    corr_with_target.to_csv(f"{csv_dir}/feature_correlations.csv")

# 5. Data Preparation
log("\nPreparing data for modeling...")
all_features = list(X.columns)

# Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log(f"Training set size: {X_train.shape}")
log(f"Test set size: {X_test.shape}")

# 6. Model Selection and Training
log("\nTraining multiple models with hyperparameter tuning...")

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models to evaluate
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}

# Hyperparameter grids for each model
param_grids = {
    'Ridge': {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'max_iter': [1000, 5000],
        'tol': [1e-4, 1e-5]
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [1000, 5000],
        'tol': [1e-4, 1e-5]
    },
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.9],
        'max_iter': [1000, 5000],
        'tol': [1e-4, 1e-5]
    },
    'SVR': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf'],
        'epsilon': [0.01, 0.1]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'colsample_bytree': [0.8, 1.0],
        'subsample': [0.8, 1.0]
    }
}

# Train and evaluate each model with hyperparameter tuning
best_models = {}
model_results = []

for model_name, model in models.items():
    log(f"\nTraining {model_name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation score
    cv_r2 = cross_val_score(best_model, X, y, cv=cv, scoring='r2').mean()
    
    # Store results
    model_results.append({
        'Model': model_name,
        'Test R2': r2,
        'CV R2': cv_r2,
        'RMSE': rmse,
        'MAE': mae,
        'Best Parameters': grid_search.best_params_
    })
    
    log(f"  Best parameters: {grid_search.best_params_}")
    log(f"  Test R2: {r2:.4f}")
    log(f"  CV R2: {cv_r2:.4f}")
    log(f"  RMSE: {rmse:.4f}")
    log(f"  MAE: {mae:.4f}")
    
    # Save predictions for this model
    model_predictions = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Error': y_test.values - y_pred
    })
    model_predictions.to_csv(f"{csv_dir}/{model_name}_predictions.csv", index=False)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual R1', fontsize=14)
    plt.ylabel('Predicted R1', fontsize=14)
    plt.title(f'{model_name}: Actual vs Predicted Values', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{images_dir}/{model_name}_predictions.png")

# 7. Model Comparison
results_df = pd.DataFrame(model_results)
log("\nModel Comparison:")
comparison_table = results_df[['Model', 'Test R2', 'CV R2', 'RMSE', 'MAE']].sort_values('CV R2', ascending=False)
log(str(comparison_table))

# Save model comparison to CSV
results_df.to_csv(f"{csv_dir}/model_comparison.csv", index=False)

# Plot model comparison
plt.figure(figsize=(14, 8))
models_to_plot = comparison_table['Model'].values
cv_r2_values = [results_df[results_df['Model'] == model]['CV R2'].values[0] for model in models_to_plot]
test_r2_values = [results_df[results_df['Model'] == model]['Test R2'].values[0] for model in models_to_plot]

x = np.arange(len(models_to_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x - width/2, cv_r2_values, width, label='Cross-Validation R²', color='royalblue')
ax.bar(x + width/2, test_r2_values, width, label='Test R²', color='lightcoral')

ax.set_ylabel('R² Score', fontsize=14)
ax.set_title('Model Performance Comparison', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models_to_plot, rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{images_dir}/model_comparison_chart.png")
log(f"Model comparison chart saved to {images_dir}/model_comparison_chart.png")

# 8. Create an Ensemble Model
log("\nCreating ensemble model from top 3 models...")
# Identify the top 3 models
top_models = results_df.sort_values('CV R2', ascending=False).head(3)['Model'].values
log(f"Top models selected for ensemble: {top_models}")

# Create voting regressor with the best models
voting_estimators = [(name, best_models[name]) for name in top_models]
voting_regressor = VotingRegressor(estimators=voting_estimators)
voting_regressor.fit(X_train, y_train)

# Evaluate ensemble model
y_pred_ensemble = voting_regressor.predict(X_test)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)

log("\nEnsemble Model Performance:")
log(f"  Test R2: {r2_ensemble:.4f}")
log(f"  RMSE: {rmse_ensemble:.4f}")
log(f"  MAE: {mae_ensemble:.4f}")

# Save ensemble predictions
ensemble_predictions = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_ensemble,
    'Error': y_test.values - y_pred_ensemble
})
ensemble_predictions.to_csv(f"{csv_dir}/ensemble_predictions.csv", index=False)

# Plot actual vs predicted for ensemble
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_ensemble, alpha=0.8, s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual R1', fontsize=14)
plt.ylabel('Predicted R1', fontsize=14)
plt.title('Ensemble Model: Actual vs Predicted Values', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{images_dir}/ensemble_predictions.png")
log(f"Ensemble model predictions saved to {images_dir}/ensemble_predictions.png")

# 9. Feature Importance Analysis
log("\nAnalyzing feature importance...")
if 'RandomForest' in best_models or 'GradientBoosting' in best_models or 'XGBoost' in best_models:
    importance_model = None
    for model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
        if model_name in best_models:
            importance_model = best_models[model_name]
            model_with_importance = model_name
            break
    
    if importance_model is not None:
        if hasattr(importance_model, 'feature_importances_'):
            importances = importance_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            log(f"\nFeature Importance from {model_with_importance}:")
            log(str(feature_importance))
            
            # Save feature importance to CSV
            feature_importance.to_csv(f"{csv_dir}/feature_importance.csv", index=False)
            
            # Plot feature importance
            plt.figure(figsize=(14, 10))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
            plt.title(f'Feature Importance from {model_with_importance}', fontsize=16)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{images_dir}/feature_importance.png")
            log(f"Feature importance plot saved to {images_dir}/feature_importance.png")

# 10. Sensitivity Analysis for Ensemble Model
log("\nPerforming sensitivity analysis using feature removal method...")
sensitivity_results = []

# Get baseline R² with all features
baseline_r2 = r2_score(y_test, y_pred_ensemble)
log(f"\nBaseline R² (all features): {baseline_r2:.4f}")

for feature in all_features:
    # Create copy of training and test data without this feature
    X_train_reduced = X_train.drop(columns=[feature])
    X_test_reduced = X_test.drop(columns=[feature])
    
    # Train a new model without this feature
    voting_estimators_reduced = []
    for name in top_models:
        model = clone(best_models[name])  # Clone to get a fresh copy
        model.fit(X_train_reduced, y_train)
        voting_estimators_reduced.append((name, model))
    
    voting_regressor_reduced = VotingRegressor(estimators=voting_estimators_reduced)
    voting_regressor_reduced.fit(X_train_reduced, y_train)
    
    # Get predictions without this feature
    y_pred_reduced = voting_regressor_reduced.predict(X_test_reduced)
    
    # Calculate R² without this feature
    r2_reduced = r2_score(y_test, y_pred_reduced)
    
    # Calculate sensitivity as the change in R²
    sensitivity = baseline_r2 - r2_reduced
    
    # Store results
    sensitivity_results.append({
        'Feature': feature,
        'R2_without_feature': r2_reduced,
        'Sensitivity': sensitivity,
        'Relative_Importance': (sensitivity / baseline_r2) * 100  # as percentage
    })

# Create and display sensitivity DataFrame
sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df = sensitivity_df.sort_values('Sensitivity', ascending=False)

log("\nSensitivity Analysis Results (all features):")
log(str(sensitivity_df))

# Save detailed sensitivity analysis to CSV
sensitivity_df.to_csv(f"{csv_dir}/sensitivity_analysis.csv", index=False)

# Plot sensitivity analysis results
plt.figure(figsize=(14, 10))
sns.barplot(x='Sensitivity', y='Feature', data=sensitivity_df, palette='viridis')
plt.title('Feature Sensitivity Analysis\nBased on R² Change when Feature Removed', fontsize=16)
plt.xlabel('Sensitivity (Change in R² when feature removed)', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig(f"{images_dir}/sensitivity_analysis.png")
log(f"Sensitivity analysis plot saved to {images_dir}/sensitivity_analysis.png")

# Save the best model
import joblib
joblib.dump(voting_regressor, f"{csv_dir}/best_foam_model.joblib")
log(f"\nBest model saved as {csv_dir}/best_foam_model.joblib")

# Create a summary of model performance and insights
with open(f"{csv_dir}/summary_insights.txt", 'w') as f:
    f.write("# FOAM DATA PREDICTION MODEL - SUMMARY INSIGHTS\n\n")
    f.write(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## DATA OVERVIEW\n")
    f.write(f"Total observations: {df.shape[0]}\n")
    f.write(f"Target variable range: {df['R1'].min():.2f} to {df['R1'].max():.2f}\n\n")
    
    f.write("## MODEL PERFORMANCE\n")
    f.write(f"Best individual model: {top_models[0]} (R² = {results_df[results_df['Model'] == top_models[0]]['CV R2'].values[0]:.4f})\n")
    f.write(f"Ensemble model R²: {r2_ensemble:.4f}\n")
    f.write(f"Ensemble model RMSE: {rmse_ensemble:.4f}\n\n")
    
    f.write("## FEATURE IMPORTANCE\n")
    for i, row in feature_importance.iterrows():
        f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")
    
    f.write("## TOP MOST SENSITIVE FEATURES\n")
    for i, row in sensitivity_df.head(3).iterrows():
        f.write(f"{i+1}. {row['Feature']}: {row['Sensitivity']:.4f}\n")
    f.write("\n")
    
    f.write("## KEY INSIGHTS\n")
    f.write("- Using only the basic features (without engineering new features)\n")
    
    # Most important feature
    top_feature = feature_importance.iloc[0]['Feature']
    f.write(f"- The most influential feature is '{top_feature}', suggesting this has the greatest impact on foam properties.\n")
    
    # Final conclusion
    f.write("\n## CONCLUSION\n")
    if r2_ensemble > 0.9:
        f.write("Even with only basic features, the model shows excellent predictive performance.\n")
    elif r2_ensemble > 0.7:
        f.write("Using only basic features, the model shows good predictive performance.\n")
    else:
        f.write("Using only basic features, the model shows moderate predictive performance. Consider adding feature engineering for improvement.\n")

log(f"\nSummary insights saved to {csv_dir}/summary_insights.txt")

log("\n--- Analysis Complete ---")

# Close the report file
report_file.close()

log(f"\nAll results have been saved in the '{csv_dir}' directory.")
log(f"Main report file: {report_filename}")
log(f"Visualizations are in: {images_dir}") 