# Optimized Foam Data Prediction System

This project provides an advanced machine learning system for predicting foam properties (R1) based on techniques, composition, and number of passes. The system includes comprehensive feature engineering, model selection, hyperparameter optimization, and advanced visualization tools.

## Project Structure

- `Improved_Foam_Predictions.py`: Main script for model training and evaluation
- `Foam_Model_Interpretability.py`: Advanced model interpretation and visualization
- `Foam_Prediction_Optimization_Guide.md`: Comprehensive guide to the optimization approach
- `Foam Data.csv`: Original dataset
- `requirements.txt`: Required packages

## Installation

1. Clone or download this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Instructions

### 1. Train the Optimized Model

Run the main prediction script:

```
python Improved_Foam_Predictions.py
```

This will:
- Perform advanced feature engineering
- Train multiple models with optimized hyperparameters
- Create an ensemble model from the best performers
- Generate visualization of results
- Save the best model as `best_foam_model.joblib`

### 2. Interpret Model Predictions

After training the model, run the interpretability script:

```
python Foam_Model_Interpretability.py
```

This will generate:
- SHAP value plots showing feature importance
- Partial dependence plots showing feature effects
- Feature interaction visualizations
- 3D visualization of predictions
- Decision boundary visualizations

### 3. Making New Predictions

You can use the saved model to make predictions on new data:

```python
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('best_foam_model.joblib')

# Example: Create a new data point
# (You'll need to apply the same feature engineering as in training)
new_data = {
    'Techniques': 'Buired',  
    'Composition': 'A',
    'Passes': 2
}

# Apply feature engineering (same as in training)
# ...

# Make prediction
prediction = model.predict(new_data_processed)
print(f"Predicted R1: {prediction[0]:.4f}")
```

## Optimization Details

See `Foam_Prediction_Optimization_Guide.md` for a detailed explanation of the optimization strategy, including:

1. Enhanced feature engineering
2. Model selection and hyperparameter tuning
3. Ensemble learning
4. Feature importance analysis
5. Sensitivity analysis

## Requirements

- Python 3.8+
- Key packages: scikit-learn, pandas, numpy, matplotlib, xgboost, shap
- See requirements.txt for complete list

## Improvements Over Original Approach

The optimized system provides several advantages:

1. **Improved Accuracy**: Higher R2 score through ensemble modeling
2. **Better Convergence**: Resolved convergence issues in original Ridge model
3. **Advanced Feature Engineering**: Captures complex relationships in the data
4. **Model Interpretability**: Deeper insights into factors affecting foam properties
5. **Robust Predictions**: Ensemble approach reduces prediction variance 