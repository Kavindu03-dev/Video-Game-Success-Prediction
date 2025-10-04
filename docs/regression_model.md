# Regression Model for Total Sales Prediction

This document explains how to train and use the regression model to predict total sales (in million units) for video games.

## Overview

The regression model complements the classification model by predicting the **actual total sales value** instead of just whether a game will be a "hit" or not.

### Models Trained
- **Random Forest Regressor** (300 trees, max_depth=15)
- **Gradient Boosting Regressor** (200 trees, max_depth=7)
- **XGBoost Regressor** (optional, if xgboost is installed)

The best model is selected based on **R² score** (coefficient of determination).

## Training the Regression Model

### 1. Ensure Dependencies

Make sure you have these packages installed:

```bash
pip install scikit-learn pandas numpy joblib
```

Optional (for XGBoost):
```bash
pip install xgboost
```

### 2. Run Training

From the project root:

```bash
python src/train_regression.py
```

Or from the `src` directory:

```bash
cd src
python train_regression.py
```

### 3. Output

The script will:
- Load and preprocess data from `data/vg_sales_2024.csv`
- Train multiple regression models
- Evaluate using MAE, RMSE, and R² metrics
- Save the best model to `models/best_regressor.joblib`
- Save metrics to `models/regressor_metrics.json`

Example output:
```
Training random_forest_reg...

Model: random_forest_reg
  MAE (Mean Absolute Error): 0.1234 million units
  RMSE (Root Mean Squared Error): 0.5678 million units
  R² Score: 0.8234

==================================================
BEST REGRESSION MODEL: random_forest_reg
  MAE: 0.1234 million units
  RMSE: 0.5678 million units
  R² Score: 0.8234
==================================================

Saved best regressor to: models/best_regressor.joblib
```

## Using the Model

### In Streamlit App

Once trained, the regression model is automatically loaded by `app.py`:

1. Run the app: `streamlit run src/app.py`
2. Go to the "Predict" page
3. Enter game details (genre, console, publisher, developer, critic score)
4. Click "Predict"
5. You'll see both:
   - **Hit Classification** (Hit/Not Hit with probability)
   - **Total Sales Prediction** (predicted sales in million units)

### Programmatically

```python
import joblib
import pandas as pd

# Load model
regressor = joblib.load('models/best_regressor.joblib')

# Prepare input
X = pd.DataFrame([{
    'genre': 'action',
    'console': 'ps4',
    'publisher': 'sony',
    'developer': 'naughty dog',
    'critic_score': 9.5,
    'release_year': 2020
}])

# Predict
predicted_sales = regressor.predict(X)[0]
print(f"Predicted sales: {predicted_sales:.2f} million units")
```

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average difference between predicted and actual sales
  - Lower is better
  - In million units
  
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals
  - Lower is better
  - Penalizes larger errors more than MAE
  - In million units
  
- **R² Score**: Proportion of variance explained by the model
  - Range: 0 to 1 (higher is better)
  - 1.0 = perfect prediction
  - 0.0 = model is no better than predicting the mean

## Comparison: Classification vs Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Predicts** | Hit (1) or Not Hit (0) | Total sales value |
| **Output** | Probability (0-1) | Sales in million units |
| **Threshold** | 1.0 million units | N/A |
| **Metrics** | Accuracy, F1, Precision, Recall | MAE, RMSE, R² |
| **Use Case** | "Will this game be successful?" | "How many units will it sell?" |
| **Model File** | `best_model.joblib` | `best_regressor.joblib` |

## Preprocessing

Both models use the same feature preprocessing:
- **Categorical features**: console, genre, publisher, developer (one-hot encoded)
- **Numeric features**: critic_score, release_year (median imputation + scaling)
- **Text normalization**: lowercase, strip whitespace
- **Rare category bucketing**: publishers/developers with <20 occurrences → "other"

The key difference is the target variable:
- Classification: `success` (binary: 0 or 1)
- Regression: `total_sales` (continuous: 0.0 to 100.0+ million)

## Troubleshooting

### Error: "Dataset not found"
- Ensure `data/vg_sales_2024.csv` exists
- Or set environment variable: `export VG_DATA_PATH=/path/to/csv`

### Warning: "XGBoost not available"
- Optional: `pip install xgboost`
- The script will still work with Random Forest and Gradient Boosting

### Low R² score
- Try feature engineering (add interaction terms, polynomial features)
- Collect more data
- Check for data leakage (e.g., using future information)

## Next Steps

1. **Frontend Integration**: Update React frontend API to support regression predictions
2. **Ensemble**: Combine classification and regression for hybrid recommendations
3. **Confidence Intervals**: Add prediction intervals for regression estimates
4. **Feature Engineering**: Add regional sales ratios, release timing features
5. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV

## Files

- `src/train_regression.py` - Training script
- `src/preprocess.py` - Shared preprocessing utilities
- `models/best_regressor.joblib` - Saved regression model
- `models/regressor_metrics.json` - Model performance metrics
