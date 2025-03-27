"""
Foam Model Interpretability (Basic Features)

This script provides advanced interpretability methods for the foam data prediction model
using only the original features without additional feature engineering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import warnings
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Create directories for saving visualizations
results_dir = 'basic_foam_interpretability'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("Loading data and model...")
# Load the saved model
try:
    model = joblib.load('basic_foam_predictions/data/best_foam_model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please run Simple_Foam_Predictions.py first.")
    exit()

# Load and prepare data (similar to the prediction script)
df = pd.read_csv("Foam Data.csv")

# Simple Feature Processing (just one-hot encoding categorical variables)
print("Processing features (using only basic features)...")

# One-hot encode categorical variables
categorical_cols = ['Techniques', 'Composition']
encoded_cols = pd.get_dummies(df[categorical_cols], drop_first=False)

# Keep Passes as a numeric feature
numeric_features = ['Passes']

# Combine features
X = pd.concat([df[numeric_features], encoded_cols], axis=1)
y = df['R1']

# Define feature sets
all_features = list(X.columns)
one_hot_features = list(encoded_cols.columns)

# Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Save original categorical values for plotting
original_df = df.copy()

# 1. Enhanced SHAP Values Analysis
print("\nComputing comprehensive SHAP values analysis...")
try:
    # Try to use the first estimator in the voting regressor (likely a tree-based model)
    estimators = model.estimators_
    model_for_shap = estimators[0]
    model_name = model.estimators_[0][0]  # Get the name of the model
    
    # Check if it's a tree-based model (for TreeExplainer)
    if hasattr(model_for_shap, 'feature_importances_'):
        explainer = shap.TreeExplainer(model_for_shap)
        is_tree = True
        print(f"Using TreeExplainer with {model_name} model")
    else:
        # Fall back to KernelExplainer for non-tree models
        explainer = shap.KernelExplainer(model_for_shap.predict, shap.sample(X, 100))
        is_tree = False
        print(f"Using KernelExplainer with {model_name} model")
    
    # Compute SHAP values for all samples
    if is_tree:
        shap_values = explainer.shap_values(X)
        # For sample visualizations, compute a smaller subset
        shap_values_sample = explainer.shap_values(X.iloc[:20])
        X_sample = X.iloc[:20]
    else:
        # For non-tree models, use a smaller sample for efficiency
        X_sample = shap.sample(X, 50)
        shap_values = explainer.shap_values(X_sample)
        shap_values_sample = explainer.shap_values(X_sample.iloc[:20])
        X_sample = X_sample.iloc[:20]
    
    # 1.1 SHAP Summary Plot (Bar) - Feature Impact Ranking
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"Feature Impact on Predictions (SHAP Values) - {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/SHAP_Feature_Impact_Bar.png')
    print(f"SHAP feature impact bar plot saved as '{results_dir}/SHAP_Feature_Impact_Bar.png'")
    plt.close()
    
    # 1.2 SHAP Summary Plot (Violin) - Distribution of Feature Effects
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, max_display=10, show=False)
    plt.title(f"SHAP Values Distribution for Top Features - {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/SHAP_Top_Features_Distribution.png')
    print(f"SHAP values distribution plot saved as '{results_dir}/SHAP_Top_Features_Distribution.png'")
    plt.close()
    
    # 1.3 SHAP Dependence Plots - Interaction Effects
    # For the most important features
    most_imp_features = np.argsort(np.abs(shap_values).mean(0))[-3:]  # Top 3 features
    for i, feat_idx in enumerate(most_imp_features):
        feature_name = X.columns[feat_idx]
        plt.figure(figsize=(12, 8))
        # Color by the next most important feature
        interaction_idx = most_imp_features[i-1] if i > 0 else most_imp_features[-1]
        interaction_feature = X.columns[interaction_idx]
        
        # Create the dependence plot
        shap.dependence_plot(
            feature_name, 
            shap_values, 
            X, 
            interaction_index=interaction_feature,
            show=False,
            alpha=0.8
        )
        plt.title(f"SHAP Dependence: {feature_name} (interacting with {interaction_feature})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/SHAP_Dependence_{feature_name}_with_{interaction_feature}.png')
        print(f"SHAP dependence plot saved as '{results_dir}/SHAP_Dependence_{feature_name}_with_{interaction_feature}.png'")
        plt.close()
    
    # 1.4 SHAP Force Plots - Sample Explanations
    # Select representative samples from each category for detailed explanation
    # For each technique
    for technique in df['Techniques'].unique():
        # Find a sample with this technique
        tech_mask = df['Techniques'] == technique
        if tech_mask.sum() > 0:
            sample_idx = df[tech_mask].index[0]
            if sample_idx < len(X):
                # Create force plot
                plt.figure(figsize=(14, 3))
                force_plot = shap.force_plot(
                    explainer.expected_value, 
                    shap_values[sample_idx:sample_idx+1, :], 
                    X.iloc[sample_idx:sample_idx+1, :],
                    matplotlib=True,
                    show=False
                )
                plt.title(f"SHAP Force Plot: Sample with Technique = {technique}", fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{results_dir}/SHAP_Force_Technique_{technique.replace(" ", "_")}.png')
                print(f"SHAP force plot saved as '{results_dir}/SHAP_Force_Technique_{technique.replace(' ', '_')}.png'")
                plt.close()
    
    # 1.5 SHAP Decision Plot - Multiple Sample Comparison
    # Compare samples with different compositions
    composition_samples = []
    for composition in df['Composition'].unique():
        mask = df['Composition'] == composition
        if mask.sum() > 0:
            idx = df[mask].index[0]
            if idx < len(X):
                composition_samples.append(idx)
    
    if len(composition_samples) > 1:
        plt.figure(figsize=(12, 10))
        shap.decision_plot(
            explainer.expected_value,
            shap_values[composition_samples, :],
            X.iloc[composition_samples, :],
            feature_names=X.columns.tolist(),
            show=False
        )
        plt.title("SHAP Decision Plot: Comparison Across Compositions", fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/SHAP_Decision_Compositions.png')
        print(f"SHAP decision plot saved as '{results_dir}/SHAP_Decision_Compositions.png'")
        plt.close()
    
    # 1.6 SHAP Waterfall Plot - Detailed Feature Contribution
    # Pick a sample with interesting prediction
    interesting_idx = np.argmax(np.abs(y - np.mean(y)))
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[interesting_idx, :],
            base_values=explainer.expected_value,
            data=X.iloc[interesting_idx, :].values,
            feature_names=X.columns.tolist()
        ),
        max_display=10,
        show=False
    )
    plt.title("SHAP Waterfall Plot: Detailed Feature Contributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/SHAP_Waterfall_Plot.png')
    print(f"SHAP waterfall plot saved as '{results_dir}/SHAP_Waterfall_Plot.png'")
    plt.close()
    
    # 1.7 SHAP Interaction Values (if tree-based model)
    if is_tree and hasattr(explainer, 'shap_interaction_values'):
        try:
            # Compute SHAP interaction values (for a smaller sample)
            interaction_values = explainer.shap_interaction_values(X.iloc[:10])
            
            # Plot top interaction effects
            interactions_sum = np.abs(interaction_values).sum(axis=0)
            
            # Get the top interactions
            feature_names = X.columns.tolist()
            top_interactions = np.dstack(np.unravel_index(np.argsort(interactions_sum.flatten())[-10:], interactions_sum.shape))[0]
            
            # Create a heatmap of interactions
            plt.figure(figsize=(12, 10))
            interaction_matrix = np.zeros((len(feature_names), len(feature_names)))
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    interaction_matrix[i, j] = np.abs(interaction_values[:, i, j]).mean()
            
            mask = np.zeros_like(interaction_matrix)
            mask[np.triu_indices_from(mask)] = True  # Mask the upper triangle
            
            # Plot the heatmap
            sns.heatmap(interaction_matrix, mask=mask, cmap='viridis',
                      xticklabels=feature_names, yticklabels=feature_names,
                      square=True, linewidths=.5, cbar_kws={"shrink": .8})
            
            plt.title("SHAP Interaction Values", fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{results_dir}/SHAP_Interaction_Heatmap.png')
            print(f"SHAP interaction heatmap saved as '{results_dir}/SHAP_Interaction_Heatmap.png'")
            plt.close()
        except Exception as e:
            print(f"Could not compute interaction values: {str(e)}")
    
except Exception as e:
    print(f"Could not compute SHAP values: {str(e)}")
    print("Continuing with other analyses...")

# 2. Partial Dependence Plots for key features
print("\nGenerating partial dependence plots...")
try:
    # Get the final estimator if it's a pipeline, or the model itself if not
    if hasattr(model, 'estimators_'):
        # Use the first estimator from the voting regressor for PDPs
        model_for_pdp = model.estimators_[0]
    else:
        model_for_pdp = model
    
    # Create partial dependence plot for numeric feature (Passes)
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model_for_pdp, X, ['Passes'], 
        kind="both", subsample=50, ax=ax, n_jobs=-1
    )
    plt.suptitle("Partial Dependence Plot for Passes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{results_dir}/Partial_Dependence_Passes.png')
    print(f"Partial dependence plot saved as '{results_dir}/Partial_Dependence_Passes.png'")
    
    # Create partial dependence plots for categorical features by sampling some one-hot encoded features
    categorical_pdp_features = []
    for technique in ['Techniques_Buired', 'Techniques_Groove', 'Techniques_Sandwich']:
        if technique in X.columns:
            categorical_pdp_features.append(technique)
            break
    
    for composition in ['Composition_A', 'Composition_B', 'Composition_C']:
        if composition in X.columns:
            categorical_pdp_features.append(composition)
            break
    
    if categorical_pdp_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            model_for_pdp, X, categorical_pdp_features, 
            kind="both", subsample=50, ax=ax, n_jobs=-1
        )
        plt.suptitle("Partial Dependence Plots for Categorical Features", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'{results_dir}/Partial_Dependence_Categorical.png')
        print(f"Categorical partial dependence plots saved as '{results_dir}/Partial_Dependence_Categorical.png'")
    
except Exception as e:
    print(f"Could not generate partial dependence plots: {str(e)}")
    print("Continuing with other analyses...")

# 3. Feature Interaction Analysis using Actual vs Predicted Values
print("\nAnalyzing feature interactions through visualization...")

# Make predictions with the model
y_pred = model.predict(X)

# Combine the predictions with the features and the target
results_df = X.copy()
results_df['Actual_R1'] = y
results_df['Predicted_R1'] = y_pred
results_df['Error'] = y - y_pred

# Save original categorical values for plotting
results_df['Techniques_Original'] = df['Techniques']
results_df['Composition_Original'] = df['Composition']
results_df['Passes_Original'] = df['Passes']

# Plot error distribution by Techniques, Composition, and Passes
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Error by Technique
sns.boxplot(x='Techniques_Original', y='Error', data=results_df, ax=axs[0])
axs[0].set_title('Prediction Error by Technique')
axs[0].set_ylabel('Error (Actual - Predicted)')

# Error by Composition
sns.boxplot(x='Composition_Original', y='Error', data=results_df, ax=axs[1])
axs[1].set_title('Prediction Error by Composition')
axs[1].set_ylabel('')

# Error by Passes
sns.boxplot(x='Passes_Original', y='Error', data=results_df, ax=axs[2])
axs[2].set_title('Prediction Error by Passes')
axs[2].set_ylabel('')

plt.tight_layout()
plt.savefig(f'{results_dir}/Error_Distribution_by_Features.png')
print(f"Error distribution plot saved as '{results_dir}/Error_Distribution_by_Features.png'")

# 4. Actual vs Predicted by Categories
print("\nCreating plots of actual vs predicted values by categories...")

# By Technique
plt.figure(figsize=(14, 8))
techniques = df['Techniques'].unique()
for i, technique in enumerate(techniques):
    mask = results_df['Techniques_Original'] == technique
    plt.scatter(
        results_df.loc[mask, 'Actual_R1'],
        results_df.loc[mask, 'Predicted_R1'],
        label=technique,
        alpha=0.7,
        s=80
    )

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual R1')
plt.ylabel('Predicted R1')
plt.title('Actual vs Predicted R1 by Technique')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_dir}/Actual_vs_Predicted_by_Technique.png')
print(f"Actual vs Predicted by Technique plot saved as '{results_dir}/Actual_vs_Predicted_by_Technique.png'")

# By Composition
plt.figure(figsize=(14, 8))
compositions = df['Composition'].unique()
for i, composition in enumerate(compositions):
    mask = results_df['Composition_Original'] == composition
    plt.scatter(
        results_df.loc[mask, 'Actual_R1'],
        results_df.loc[mask, 'Predicted_R1'],
        label=composition,
        alpha=0.7,
        s=80
    )

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual R1')
plt.ylabel('Predicted R1')
plt.title('Actual vs Predicted R1 by Composition')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_dir}/Actual_vs_Predicted_by_Composition.png')
print(f"Actual vs Predicted by Composition plot saved as '{results_dir}/Actual_vs_Predicted_by_Composition.png'")

# By Passes
plt.figure(figsize=(14, 8))
passes_values = sorted(df['Passes'].unique())
for i, passes in enumerate(passes_values):
    mask = results_df['Passes_Original'] == passes
    plt.scatter(
        results_df.loc[mask, 'Actual_R1'],
        results_df.loc[mask, 'Predicted_R1'],
        label=f'Passes={passes}',
        alpha=0.7,
        s=80
    )

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual R1')
plt.ylabel('Predicted R1')
plt.title('Actual vs Predicted R1 by Number of Passes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_dir}/Actual_vs_Predicted_by_Passes.png')
print(f"Actual vs Predicted by Passes plot saved as '{results_dir}/Actual_vs_Predicted_by_Passes.png'")

# 5. Combined Feature Importances Visualization
print("\nCreating combined feature importance visualization...")
try:
    # Get feature importances from the best tree-based model
    tree_models = ['RandomForest', 'GradientBoosting', 'XGBoost']
    tree_model = None
    
    for name, est in model.estimators_:
        if name in tree_models and hasattr(est, 'feature_importances_'):
            tree_model = est
            tree_model_name = name
            break
    
    if tree_model is not None:
        importances = tree_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importances with SHAP values if available
        try:
            # Create a plot with both importances and SHAP values
            plt.figure(figsize=(14, 10))
            
            # Plot feature importances
            ax1 = plt.subplot(2, 1, 1)
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax1)
            ax1.set_title(f'Feature Importance from {tree_model_name}', fontsize=14)
            ax1.set_xlabel('Importance', fontsize=12)
            ax1.set_ylabel('Feature', fontsize=12)
            
            # Plot SHAP summary
            ax2 = plt.subplot(2, 1, 2)
            if 'shap_values' in locals():
                shap.summary_plot(shap_values, X, plot_type="bar", show=False, ax=ax2)
                ax2.set_title('SHAP Feature Impact', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(f'{results_dir}/Combined_Feature_Importance.png')
            print(f"Combined feature importance plot saved as '{results_dir}/Combined_Feature_Importance.png'")
            plt.close()
            
        except Exception as e:
            print(f"Error creating combined plot: {str(e)}")
    
except Exception as e:
    print(f"Could not create combined feature importance plot: {str(e)}")

# 6. Simplified Decision Boundary Visualization for basic features
print("\nVisualizing simplified decision boundaries...")
try:
    # Create a grid of Passes values
    passes_range = np.linspace(df['Passes'].min(), df['Passes'].max(), 100)
    
    # Create a figure for Technique comparison
    plt.figure(figsize=(12, 8))
    
    # For each technique
    for technique in df['Techniques'].unique():
        predictions = []
        
        for p in passes_range:
            # Create a sample with the base features
            sample = pd.DataFrame(
                {f: 0 for f in X.columns}, 
                index=[0]
            )
            
            # Set Passes (standardized)
            sample['Passes'] = (p - df['Passes'].mean()) / df['Passes'].std()
            
            # Set technique
            tech_col = f'Techniques_{technique}'
            if tech_col in sample.columns:
                sample[tech_col] = 1
            
            # Predict
            pred = model.predict(sample)[0]
            predictions.append(pred)
        
        # Plot this technique
        plt.plot(passes_range, predictions, label=f'Technique: {technique}', linewidth=3)
    
    plt.xlabel('Number of Passes')
    plt.ylabel('Predicted R1 Value')
    plt.title('Predicted R1 by Passes and Technique')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/Decision_Boundary_Techniques.png')
    print(f"Decision boundary plot saved as '{results_dir}/Decision_Boundary_Techniques.png'")
    
    # Create a figure for Composition comparison
    plt.figure(figsize=(12, 8))
    
    # For each composition
    for composition in df['Composition'].unique():
        predictions = []
        
        for p in passes_range:
            # Create a sample with the base features
            sample = pd.DataFrame(
                {f: 0 for f in X.columns}, 
                index=[0]
            )
            
            # Set Passes (standardized)
            sample['Passes'] = (p - df['Passes'].mean()) / df['Passes'].std()
            
            # Set composition
            comp_col = f'Composition_{composition}'
            if comp_col in sample.columns:
                sample[comp_col] = 1
            
            # Predict
            pred = model.predict(sample)[0]
            predictions.append(pred)
        
        # Plot this composition
        plt.plot(passes_range, predictions, label=f'Composition: {composition}', linewidth=3)
    
    plt.xlabel('Number of Passes')
    plt.ylabel('Predicted R1 Value')
    plt.title('Predicted R1 by Passes and Composition')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/Decision_Boundary_Compositions.png')
    print(f"Decision boundary plot saved as '{results_dir}/Decision_Boundary_Compositions.png'")
    
except Exception as e:
    print(f"Could not create decision boundary visualization: {str(e)}")

print("\n--- Basic Model Interpretability Analysis Complete ---")
print(f"All visualizations saved to the '{results_dir}' directory") 