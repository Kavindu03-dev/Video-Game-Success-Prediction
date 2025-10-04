# Regression Prediction Quick Start Guide

## 🎯 What You Can Do Now

You now have **TWO prediction models** working together:

1. **Classification Model**: Predicts if a game will be a "Hit" (≥1.0M sales)
   - Output: Hit or Not Hit + Probability
   
2. **Regression Model**: Predicts actual total sales
   - Output: Sales value in million units

## 🚀 Quick Start (3 Steps)

### Step 1: Train the Regression Model

```powershell
# From project root
python src/train_regression.py
```

**Output:** `models/best_regressor.joblib` (Gradient Boosting model)

### Step 2: Test It

```powershell
# Single game prediction test
python test_dual_prediction.py
```

**Shows:** Both classification and regression predictions side-by-side

### Step 3: Use in Streamlit App

```powershell
streamlit run src/app.py
```

Go to **Predict** page → Enter game details → Click "Predict" → See both predictions!

## 📊 Batch Predictions

Predict for multiple games at once:

```powershell
# Create input CSV with columns:
# genre,console,publisher,developer,critic_score,release_year

python predict_batch.py sample_games.csv output_predictions.csv
```

**Example included:** `sample_games.csv` with 7 test games

## 📈 Model Performance

| Model | Task | Best Algorithm | Score |
|-------|------|---------------|-------|
| Classification | Hit/Not Hit | Random Forest | 93% accuracy |
| Regression | Sales Value | Gradient Boosting | R²=0.33, MAE=0.26M |

## 💡 Understanding the Predictions

### Example Output

```
Game: Action game on PS4 by Sony/Naughty Dog, Score 9.5

Classification: Not Hit (41% probability)
Regression: 2.18 million units

Interpretation: Mixed signals - predicted sales above 1M threshold
but classification model is cautious. Consider as moderate success.
```

### Prediction Combinations

| Classification | Regression | Meaning |
|---------------|------------|---------|
| Hit (>50%) | ≥1.0M | ✅ Strong - Both agree it will succeed |
| Not Hit (<50%) | <1.0M | ✅ Weak - Both agree it won't hit threshold |
| Hit (>50%) | <1.0M | ⚠️ Mixed - May be close to threshold |
| Not Hit (<50%) | ≥1.0M | ⚠️ Mixed - Regression more optimistic |

## 🛠️ Files Overview

### Training Scripts
- `src/train.py` - Train classification model
- `src/train_regression.py` - Train regression model

### Models (Generated)
- `models/best_model.joblib` - Classification model
- `models/best_regressor.joblib` - Regression model
- `models/metrics.json` - Classification metrics
- `models/regressor_metrics.json` - Regression metrics

### Utility Scripts
- `test_dual_prediction.py` - Test single game prediction
- `predict_batch.py` - Batch prediction for CSV files
- `sample_games.csv` - Example input for batch prediction

### Documentation
- `docs/regression_model.md` - Detailed regression guide
- `REGRESSION_IMPLEMENTATION.md` - Implementation summary

### Apps
- `src/app.py` - Streamlit app (updated with dual prediction)

## 🎨 Streamlit App Features

### Predict Page Now Shows:

**Before clicking Predict:**
- Input fields for game details

**After clicking Predict:**
1. **Hit Classification** section
   - Prediction: Hit or Not Hit
   - Probability: XX.X%
   - Progress bar showing probability

2. **Total Sales Prediction** section
   - Predicted Total Sales: X.XX M units
   - Caption explaining it's from regression model

## 📝 Code Examples

### Python API

```python
import joblib
import pandas as pd

# Load both models
classifier = joblib.load('models/best_model.joblib')
regressor = joblib.load('models/best_regressor.joblib')

# Prepare game data
game = pd.DataFrame([{
    'genre': 'action',
    'console': 'ps4',
    'publisher': 'sony',
    'developer': 'naughty dog',
    'critic_score': 9.5,
    'release_year': 2020
}])

# Get both predictions
hit_prob = classifier.predict_proba(game)[:, 1][0]
hit_label = "Hit" if hit_prob >= 0.5 else "Not Hit"
predicted_sales = regressor.predict(game)[0]

print(f"Classification: {hit_label} ({hit_prob:.1%})")
print(f"Regression: {predicted_sales:.2f}M units")
```

### Batch Processing

```python
import pandas as pd

# Read games
games_df = pd.read_csv('games_to_predict.csv')

# Normalize text
for col in ['genre', 'console', 'publisher', 'developer']:
    games_df[col] = games_df[col].str.lower().str.strip()

# Predict
hit_probs = classifier.predict_proba(games_df)[:, 1]
sales_preds = regressor.predict(games_df)

# Add to dataframe
games_df['hit_probability'] = hit_probs
games_df['predicted_sales'] = sales_preds

# Save
games_df.to_csv('predictions.csv', index=False)
```

## 🔧 Troubleshooting

### "Model not found"
```powershell
# Train the missing model
python src/train.py              # For classifier
python src/train_regression.py  # For regressor
```

### "Dataset not found"
- Ensure `data/vg_sales_2024.csv` exists
- Or set: `$env:VG_DATA_PATH = "path\to\your.csv"`

### sklearn version warnings
- Not critical, models will still work
- To fix: `pip install --upgrade scikit-learn`

### XGBoost not available
- Optional package
- Install: `pip install xgboost`
- Or ignore - RF and GB still work great

## 📊 When to Use Which Model

### Use Classification When:
- Making go/no-go decisions
- Need binary outcome for filtering
- Probability of success is more important than exact value
- Example: "Should we greenlight this project?"

### Use Regression When:
- Need specific sales estimates
- Planning budgets or resources
- Comparing multiple options quantitatively
- Example: "Which game will sell more units?"

### Use Both When:
- Making important strategic decisions
- Want comprehensive risk assessment
- Need to explain predictions to stakeholders
- Comparing across different scenarios

## 🎯 Best Practices

1. **Always use both predictions together** for important decisions
2. **Pay attention to disagreements** - they highlight uncertainty
3. **Consider confidence** - lower hit probability means higher risk
4. **Test with similar games** - compare to historical data
5. **Update models regularly** - retrain with new sales data

## 📚 Learn More

- Full regression guide: `docs/regression_model.md`
- Implementation details: `REGRESSION_IMPLEMENTATION.md`
- Main project README: `README.md`

## ✨ Quick Commands Reference

```powershell
# Train both models
python src/train.py
python src/train_regression.py

# Test predictions
python test_dual_prediction.py

# Batch predict
python predict_batch.py input.csv output.csv

# Run app
streamlit run src/app.py

# Check model metrics
cat models/regressor_metrics.json
```

## 🎉 You're Ready!

Your dual prediction system is fully functional. Try it out:

1. Open Streamlit app: `streamlit run src/app.py`
2. Go to Predict page
3. Enter a game (or use batch CSV)
4. See both Hit/Not Hit AND sales prediction
5. Make data-driven decisions! 🚀
