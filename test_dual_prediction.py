"""
Quick test script to demonstrate dual prediction:
- Classification: Hit or Not Hit
- Regression: Predicted total sales
"""
import os
import sys
import joblib
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Load models
project_root = Path(__file__).parent
classifier = joblib.load(project_root / 'models' / 'best_model.joblib')
regressor = joblib.load(project_root / 'models' / 'best_regressor.joblib')

# Test game example
test_game = pd.DataFrame([{
    'genre': 'action',
    'console': 'ps4',
    'publisher': 'sony computer entertainment',
    'developer': 'naughty dog',
    'critic_score': 9.5,
    'release_year': 2020
}])

print("="*60)
print("TEST GAME PREDICTION")
print("="*60)
print(f"Genre:      {test_game['genre'].iloc[0]}")
print(f"Console:    {test_game['console'].iloc[0]}")
print(f"Publisher:  {test_game['publisher'].iloc[0]}")
print(f"Developer:  {test_game['developer'].iloc[0]}")
print(f"Critic Score: {test_game['critic_score'].iloc[0]}")
print(f"Year:       {test_game['release_year'].iloc[0]}")
print()

# Classification prediction
if hasattr(classifier, 'predict_proba'):
    hit_proba = classifier.predict_proba(test_game)[:, 1][0]
else:
    hit_proba = float(classifier.predict(test_game)[0])
hit_prediction = "Hit" if hit_proba >= 0.5 else "Not Hit"

print("--- CLASSIFICATION PREDICTION ---")
print(f"Prediction:   {hit_prediction}")
print(f"P(Hit):       {hit_proba:.2%}")
print()

# Regression prediction
sales_prediction = regressor.predict(test_game)[0]

print("--- REGRESSION PREDICTION ---")
print(f"Predicted Total Sales: {sales_prediction:.2f} million units")
print()

# Combined interpretation
print("--- COMBINED INSIGHTS ---")
print(f"Classification: {hit_prediction} ({hit_proba:.1%} probability)")
print(f"Expected Sales: {sales_prediction:.2f}M units")
print()

if hit_prediction == "Hit":
    print("✅ Classification predicts HIT")
    print(f"  Probability: {hit_proba:.1%}")
    print(f"  Expected sales: {sales_prediction:.2f}M units")
    if sales_prediction >= 1.5:
        print("  💪 Strong sales potential - well above typical threshold")
    elif sales_prediction >= 1.0:
        print("  ✓ Moderate sales potential - near threshold")
    else:
        print("  ⚠️ Sales estimate below typical Hit threshold (1.0M)")
else:
    print("📊 Classification predicts NOT HIT")
    print(f"  Probability: {hit_proba:.1%}")
    print(f"  Expected sales: {sales_prediction:.2f}M units")
    if sales_prediction >= 0.8:
        print("  💡 Close to threshold - niche success possible")
    else:
        print("  📉 Lower sales expected")

print("="*60)
