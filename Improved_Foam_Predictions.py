"""
Improved Foam Data Prediction Model

This script provides an enhanced approach to predicting foam properties using advanced
machine learning techniques including feature engineering, model selection, and hyperparameter tuning.
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
results_dir = 'foam_prediction_results'
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

# 3. Enhanced Feature Engineering
log("\nPerforming advanced feature engineering...")

# Create interaction terms between categorical variables
df['Tech_Comp'] = df['Techniques'] + '_' + df['Composition']

# Create numeric encoding for categorical variables
technique_map = {'Buired': 0, 'Sandwich': 1, 'Groove': 2}
composition_map = {'A': 0, 'B': 1, 'C': 2}
df['Technique_Num'] = df['Techniques'].map(technique_map)
df['Composition_Num'] = df['Composition'].map(composition_map)

# Create non-linear transformations of numerical features
df['Passes_Squared'] = df['Passes'] ** 2
df['Passes_Sqrt'] = np.sqrt(df['Passes'])
df['Passes_Log'] = np.log(df['Passes'])
df['inv_Passes'] = 1 / df['Passes']

# Create interaction terms
df['Passes_Tech'] = df['Passes'] * df['Technique_Num']
df['Passes_Comp'] = df['Passes'] * df['Composition_Num']
df['Tech_Comp_Num'] = df['Technique_Num'] * df['Composition_Num']

# One-hot encode categorical variables
categorical_cols = ['Techniques', 'Composition', 'Tech_Comp']
encoded_cols = pd.get_dummies(df[categorical_cols], drop_first=False)
df = pd.concat([df, encoded_cols], axis=1)

# Save the enhanced dataset
df.to_csv(f"{csv_dir}/enhanced_dataset.csv", index=False)

# 4. Feature Selection based on correlation analysis
log("\nAnalyzing feature correlations...")
numeric_features = ['Passes', 'Passes_Squared', 'Passes_Sqrt', 'Passes_Log', 'inv_Passes',
                   'Technique_Num', 'Composition_Num', 'Passes_Tech', 'Passes_Comp', 'Tech_Comp_Num']
one_hot_features = [col for col in encoded_cols.columns]
all_features = numeric_features + one_hot_features

# Calculate correlations with target
corr_with_target = df[numeric_features + ['R1']].corr()['R1'].abs().sort_values(ascending=False)
log("\nFeature correlation with target variable (R1):")
log(str(corr_with_target))

# Save correlations to CSV
corr_with_target.to_csv(f"{csv_dir}/feature_correlations.csv")

# Create correlation heatmap
plt.figure(figsize=(14, 10))
corr_matrix = df[numeric_features + ['R1']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig(f"{images_dir}/correlation_heatmap.png")
log(f"Correlation heatmap saved to {images_dir}/correlation_heatmap.png")

# 5. Data Preparation
log("\nPreparing data for modeling...")
X = df[all_features]
y = df['R1']

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
        'max_iter': [1000, 2000, 5000],
        'tol': [1e-4, 1e-5, 1e-6]
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [1000, 2000, 5000],
        'tol': [1e-4, 1e-5, 1e-6]
    },
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [1000, 2000, 5000],
        'tol': [1e-4, 1e-5, 1e-6]
    },
    'SVR': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['linear', 'rbf', 'poly'],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'subsample': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
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
models_to_plot = results_df.sort_values('CV R2', ascending=False)['Model'].values
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
            log(str(feature_importance.head(15)))
            
            # Save feature importance to CSV
            feature_importance.to_csv(f"{csv_dir}/feature_importance.csv", index=False)
            
            # Plot feature importance
            plt.figure(figsize=(14, 10))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette='viridis')
            plt.title(f'Top 15 Features (Importance from {model_with_importance})', fontsize=16)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{images_dir}/feature_importance.png")
            log(f"Feature importance plot saved to {images_dir}/feature_importance.png")

# 10. Visualization of Predictions
# Plot actual vs predicted values with a more attractive design
plt.figure(figsize=(12, 10))
scatter = plt.scatter(y_test, y_pred_ensemble, 
                     alpha=0.7, s=100, 
                     c=abs(y_test-y_pred_ensemble), cmap='viridis')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual R1', fontsize=14)
plt.ylabel('Predicted R1', fontsize=14)
plt.title('Actual vs Predicted Values (Ensemble Model)', fontsize=16)
plt.colorbar(scatter, label='Absolute Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{images_dir}/actual_vs_predicted_enhanced.png")
log(f"Enhanced actual vs predicted plot saved to {images_dir}/actual_vs_predicted_enhanced.png")

# 11. Sensitivity Analysis for Ensemble Model
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

# Plot sensitivity analysis results for top 15 features
plt.figure(figsize=(14, 10))
top_features = sensitivity_df.head(15)
sns.barplot(x='Sensitivity', y='Feature', data=top_features, palette='viridis')
plt.title('Feature Sensitivity Analysis (Top 15)\nBased on R² Change when Feature Removed', fontsize=16)
plt.xlabel('Sensitivity (Change in R² when feature removed)', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig(f"{images_dir}/sensitivity_analysis.png")
log(f"Sensitivity analysis plot saved to {images_dir}/sensitivity_analysis.png")

# Add sensitivity analysis results to summary insights
with open(f"{csv_dir}/summary_insights.txt", 'a') as f:
    f.write("\n## SENSITIVITY ANALYSIS INSIGHTS\n")
    f.write("Feature importance based on R² degradation when removed:\n\n")
    
    for _, row in sensitivity_df.head(5).iterrows():
        f.write(f"- {row['Feature']}:\n")
        f.write(f"  * R² without feature: {row['R2_without_feature']:.4f}\n")
        f.write(f"  * Sensitivity: {row['Sensitivity']:.4f}\n")
        f.write(f"  * Relative Importance: {row['Relative_Importance']:.2f}%\n\n")
    
    # Add interpretation
    most_important = sensitivity_df.iloc[0]
    least_important = sensitivity_df.iloc[-1]
    f.write("\nKey Findings:\n")
    f.write(f"- Most influential feature: {most_important['Feature']} ")
    f.write(f"(reduces R² by {most_important['Sensitivity']:.4f} when removed)\n")
    f.write(f"- Least influential feature: {least_important['Feature']} ")
    f.write(f"(reduces R² by {least_important['Sensitivity']:.4f} when removed)\n")

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
    
    f.write("## TOP 5 MOST IMPORTANT FEATURES\n")
    for i, row in feature_importance.head(5).iterrows():
        f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")
    
    f.write("## TOP 5 MOST SENSITIVE FEATURES\n")
    for i, row in sensitivity_df.head(5).iterrows():
        f.write(f"{i+1}. {row['Feature']}: {row['Sensitivity']:.4f}\n")
    f.write("\n")
    
    f.write("## KEY INSIGHTS\n")
    
    # Add some automatic insights based on the results
    # Most important feature
    top_feature = feature_importance.iloc[0]['Feature']
    f.write(f"- The most influential feature is '{top_feature}', suggesting this has the greatest impact on foam properties.\n")
    
    # Look at Passes importance
    passes_importance = feature_importance[feature_importance['Feature'] == 'Passes']['Importance'].values
    if len(passes_importance) > 0:
        f.write(f"- The number of passes has an importance of {passes_importance[0]:.4f}, ranking it among the " + 
                ("top" if passes_importance[0] > 0.1 else "less critical") + " factors.\n")
    
    # Technique vs Composition influence
    tech_features = [f for f in feature_importance['Feature'] if 'Technique' in f]
    comp_features = [f for f in feature_importance['Feature'] if 'Composition' in f]
    
    tech_total_imp = feature_importance[feature_importance['Feature'].isin(tech_features)]['Importance'].sum()
    comp_total_imp = feature_importance[feature_importance['Feature'].isin(comp_features)]['Importance'].sum()
    
    if tech_total_imp > comp_total_imp:
        f.write(f"- Technique-related features have more influence ({tech_total_imp:.4f}) than composition-related features ({comp_total_imp:.4f}).\n")
    else:
        f.write(f"- Composition-related features have more influence ({comp_total_imp:.4f}) than technique-related features ({tech_total_imp:.4f}).\n")
    
    # Interaction effects
    interaction_features = [f for f in feature_importance['Feature'] if '_' in f]
    inter_total_imp = feature_importance[feature_importance['Feature'].isin(interaction_features)]['Importance'].sum()
    
    f.write(f"- Interaction effects between features account for {inter_total_imp:.4f} of the total importance, suggesting " +
            ("significant" if inter_total_imp > 0.3 else "moderate" if inter_total_imp > 0.1 else "minimal") + 
            " interactions between factors.\n")
    
    # Most predictable condition
    f.write(f"- The model achieves an average error (MAE) of {mae_ensemble:.4f} across all test samples.\n")
    
    # Final recommendation based on performance
    f.write("\n## CONCLUSION\n")
    if r2_ensemble > 0.9:
        f.write("The model shows excellent predictive performance and can be reliably used for foam property prediction.\n")
    elif r2_ensemble > 0.7:
        f.write("The model shows good predictive performance and should be useful for most foam property prediction tasks.\n")
    else:
        f.write("The model shows moderate predictive performance. Consider collecting more data or exploring additional features for improvement.\n")

log(f"\nSummary insights saved to {csv_dir}/summary_insights.txt")

# 12. Create an HTML report with all results
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Foam Data Prediction Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ margin-top: 30px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .image-container {{ margin: 20px 0; }}
        .image-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .section {{ margin: 30px 0; }}
        .highlight {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Foam Data Prediction Results</h1>
    <p>Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Data Overview</h2>
        <p>Dataset contains <b>{df.shape[0]}</b> observations with <b>{df.shape[1]}</b> columns after feature engineering.</p>
        <p>Target variable (R1) range: <b>{df['R1'].min():.2f}</b> to <b>{df['R1'].max():.2f}</b></p>
    </div>
    
    <div class="section">
        <h2>Feature Engineering</h2>
        <p>We created {len(all_features)} engineered features from the original dataset.</p>
        <p>Key transformations include:</p>
        <ul>
            <li>Non-linear transformations of numerical features (squared, square root, logarithmic)</li>
            <li>Interaction terms between features</li>
            <li>One-hot encoding of categorical variables</li>
        </ul>
        
        <h3>Feature Correlation with Target</h3>
        <div class="image-container">
            <img src="correlation_heatmap.png" alt="Correlation Heatmap">
        </div>
    </div>
    
    <div class="section">
        <h2>Model Performance Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Test R²</th>
                <th>CV R²</th>
                <th>RMSE</th>
                <th>MAE</th>
            </tr>
"""

# Add model results to HTML table
for _, row in comparison_table.iterrows():
    html_report += f"""
            <tr>
                <td>{row['Model']}</td>
                <td>{row['Test R2']:.4f}</td>
                <td>{row['CV R2']:.4f}</td>
                <td>{row['RMSE']:.4f}</td>
                <td>{row['MAE']:.4f}</td>
            </tr>
    """

html_report += f"""
        </table>
        
        <div class="image-container">
            <img src="model_comparison_chart.png" alt="Model Performance Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>Ensemble Model Results</h2>
        <p>We created an ensemble of the top {len(top_models)} models: <b>{', '.join(top_models)}</b></p>
        <p>Ensemble model performance:</p>
        <ul>
            <li>R² Score: <span class="highlight">{r2_ensemble:.4f}</span></li>
            <li>RMSE: {rmse_ensemble:.4f}</li>
            <li>MAE: {mae_ensemble:.4f}</li>
        </ul>
        
        <div class="image-container">
            <img src="ensemble_predictions.png" alt="Ensemble Predictions">
        </div>
        
        <div class="image-container">
            <img src="actual_vs_predicted_enhanced.png" alt="Enhanced Predictions Visualization">
        </div>
    </div>
    
    <div class="section">
        <h2>Feature Importance</h2>
        <div class="image-container">
            <img src="feature_importance.png" alt="Feature Importance">
        </div>
    </div>
    
    <div class="section">
        <h2>Sensitivity Analysis</h2>
        <div class="image-container">
            <img src="sensitivity_analysis.png" alt="Sensitivity Analysis">
        </div>
    </div>
    
    <div class="section">
        <h2>Key Insights</h2>
        <ul>
"""

# Read insights from the summary file
with open(f"{csv_dir}/summary_insights.txt", 'r') as f:
    content = f.read()
    insights_section = content.split("## KEY INSIGHTS")[1].split("##")[0]
    insights = [line.strip() for line in insights_section.split('-') if line.strip()]
    for insight in insights:
        html_report += f"            <li>{insight}</li>\n"

html_report += """
        </ul>
    </div>
    
    <div class="section">
        <h2>Conclusion</h2>
"""

# Add conclusion from the summary file
with open(f"{csv_dir}/summary_insights.txt", 'r') as f:
    content = f.read()
    conclusion_section = content.split("## CONCLUSION")[1]
    html_report += f"        <p>{conclusion_section.strip()}</p>\n"

html_report += """
    </div>
</body>
</html>
"""

# Save HTML report
with open(f"{csv_dir}/foam_prediction_report.html", 'w') as f:
    f.write(html_report)

log(f"\nHTML report saved to {csv_dir}/foam_prediction_report.html")

log("\n--- Analysis Complete ---")

# Close the report file
report_file.close()

log(f"\nAll results have been saved in the '{csv_dir}' directory.")
log(f"Main report file: {report_filename}")
log(f"HTML report: {csv_dir}/foam_prediction_report.html") 