# Quick Reference: Classification vs Regression

## At a Glance

| Aspect | Classification Model | Regression Model |
|--------|---------------------|------------------|
| **Question** | "Will it be a Hit?" | "How much will it sell?" |
| **Output Type** | Binary (Hit/Not Hit) | Continuous (2.18M) |
| **Output Format** | Label + Probability | Number in millions |
| **Decision Making** | ✅ YES - determines success | ❌ NO - only estimates value |
| **Use For** | Go/No-Go decisions | Sales forecasting |
| **Example Output** | "Hit (75.3%)" | "2.18 million units" |
| **Threshold** | Built into model | N/A |

---

## When to Use Which

### Use Classification When:
- ❓ "Should we greenlight this game?"
- ❓ "Is this going to be successful?"
- ❓ "What's the probability of hitting our target?"
- ❓ "Should marketing budget be allocated?"

### Use Regression When:
- ❓ "How many units should we manufacture?"
- ❓ "What's our expected revenue?"
- ❓ "How much inventory do we need?"
- ❓ "What's the sales forecast for this quarter?"

### Use Both When:
- ❓ "Should we invest in this game AND how much can we expect to make?"
- ❓ "What's the risk/reward profile?"
- ❓ "Is it worth the development cost?"

---

## Code Examples

### Single Prediction

```python
# Load models
classifier = joblib.load('models/best_model.joblib')
regressor = joblib.load('models/best_regressor.joblib')

# Prepare input
game = pd.DataFrame([{
    'genre': 'action',
    'console': 'ps5',
    'publisher': 'sony',
    'developer': 'naughty dog',
    'critic_score': 9.0,
    'release_year': 2024
}])

# Classification: Hit or Not Hit?
hit_proba = classifier.predict_proba(game)[:, 1][0]
hit_label = "Hit" if hit_proba >= 0.5 else "Not Hit"
print(f"Classification: {hit_label} ({hit_proba:.1%})")
# Output: Classification: Hit (82.5%)

# Regression: How much will it sell?
predicted_sales = regressor.predict(game)[0]
print(f"Expected Sales: {predicted_sales:.2f}M units")
# Output: Expected Sales: 2.18M units

# ❌ DON'T DO THIS:
# regression_label = "Hit" if predicted_sales >= 1.0 else "Not Hit"
```

---

## In the CSV Output

### Columns Explanation

```csv
genre,console,hit_prediction,hit_probability,predicted_sales,insights
action,ps5,1,0.825,2.18,"✅ Hit predicted - Strong sales (2.18M)"
sports,switch,0,0.345,0.56,"📉 Not Hit - Lower sales expected (0.56M)"
```

| Column | From Model | Meaning |
|--------|-----------|---------|
| `hit_prediction` | Classification | 0=Not Hit, 1=Hit |
| `hit_probability` | Classification | Confidence (0.0-1.0) |
| `predicted_sales` | Regression | Sales in millions |
| `insights` | Both | Combined interpretation |

---

## Decision Framework

```
IF Classification = Hit AND Sales > 1.5M
  → High confidence, strong investment

IF Classification = Hit AND Sales 1.0-1.5M  
  → Moderate confidence, proceed with caution

IF Classification = Hit AND Sales < 1.0M
  → Mixed signals, investigate further

IF Classification = Not Hit AND Sales > 0.8M
  → Niche potential, consider target market

IF Classification = Not Hit AND Sales < 0.8M
  → Low confidence, reconsider
```

---

## Common Mistakes to Avoid

### ❌ WRONG:
```python
# Don't use regression for classification
if predicted_sales >= 1.0:
    hit_label = "Hit"  # WRONG!
```

### ✅ CORRECT:
```python
# Use classification for Hit/Not Hit
hit_label = "Hit" if classifier.predict_proba(game)[:, 1][0] >= 0.5 else "Not Hit"

# Use regression for sales estimate
sales = regressor.predict(game)[0]

# Use both for insights
if hit_label == "Hit":
    print(f"Expected to be a hit with {sales:.2f}M units")
```

---

## Testing Your Understanding

**Q1:** User asks "Will this game be successful?"
- **A:** Use **Classification Model** - it determines Hit/Not Hit

**Q2:** User asks "How many copies will sell?"
- **A:** Use **Regression Model** - it predicts sales value

**Q3:** User asks "Should we approve this project?"
- **A:** Use **Both Models** - classification for decision + regression for magnitude

**Q4:** You see `predicted_sales = 2.5M` - can you call it a Hit?
- **A:** **NO!** Only classification model determines Hit/Not Hit

---

## Quick Test Commands

```bash
# Test single prediction
python test_dual_prediction.py

# Test batch predictions
python predict_batch.py sample_games.csv output.csv

# Verify models working
python verify_models.py

# Run Streamlit app
python -m streamlit run src/app.py
```

---

**Remember:** 
- 🎯 Classification = Decision (yes/no)
- 📊 Regression = Magnitude (how much)
- 🤝 Together = Complete picture
