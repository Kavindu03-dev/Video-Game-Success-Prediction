# Unified Hit Threshold: Classification + Regression

## 🎯 Core Concept

Both models now use the **same Hit threshold of ≥1.0M units**:

1. **Classification Model**: Predicts probability that `total_sales ≥ 1.0M`
2. **Regression Model**: Predicts actual sales, then applies `≥1.0M` threshold

This unified approach makes predictions **directly comparable** and easier to interpret.

## 📊 How It Works

### Classification Model
```
Input → Model → Probability → Threshold (0.5) → Hit/Not Hit
                                    ↓
                        "Will it sell ≥1.0M units?"
```

### Regression Model  
```
Input → Model → Predicted Sales → Threshold (1.0M) → Hit/Not Hit
                                          ↓
                            "Predicted: X.XXM units"
                                   ↓
                    If ≥1.0M: Hit, else: Not Hit
```

## ✅ Benefits of Unified Threshold

1. **Consistent Definition**: Both models answer "Will this be a Hit?"
2. **Easy Comparison**: Direct model agreement/disagreement analysis
3. **Clear Interpretation**: Same success criteria across predictions
4. **Better Decisions**: Compare classification confidence with regression estimate

## 📈 Prediction Scenarios

### Scenario 1: Strong Agreement (Hit)
```
Classification: Hit (P=72%)
Regression: 1.50M units → Hit

Interpretation: ✅ Strong confidence - Both predict Hit
Recommendation: High confidence in success
```

### Scenario 2: Strong Agreement (Not Hit)
```
Classification: Not Hit (P=15%)
Regression: 0.36M units → Not Hit

Interpretation: ✅ Agreement - Both predict Not Hit
Recommendation: Low likelihood of success
```

### Scenario 3: Mixed Signals (Classification Higher)
```
Classification: Hit (P=62%)
Regression: 0.85M units → Not Hit

Interpretation: ⚠️ Mixed - Classification optimistic
Recommendation: May perform just below threshold, proceed with caution
```

### Scenario 4: Mixed Signals (Regression Higher)
```
Classification: Not Hit (P=41%)
Regression: 2.18M units → Hit

Interpretation: ⚠️ Mixed - Regression more optimistic
Recommendation: May perform above threshold, moderate confidence
```

## 🔍 Understanding Disagreements

When models disagree, it typically means:

- **Game is near the 1.0M threshold**
- **Higher uncertainty in prediction**
- **Consider additional factors** (market timing, competition, etc.)
- **Use conservative estimates** for risk management

### Example Analysis

```python
Game: Action on PS4, Score 9.5

Classification: Not Hit (41% probability)
→ Model is cautious, thinks success is unlikely

Regression: 2.18M units = Hit
→ Model predicts solid sales above threshold

Interpretation: Models disagree because prediction is 
uncertain. Sales could be 0.8M or 2.5M with similar 
confidence. Recommend further analysis.
```

## 📝 Updated Output Format

### Streamlit App Display

```
🎯 Hit Classification Model
Prediction: Hit
Probability: 72%
[Progress bar showing 72%]
Classification model predicts if total sales ≥ 1.0M units

📊 Regression Sales Prediction  
Predicted Sales: 1.50M units
Based on Sales: Hit (Above 1.0M threshold)
Regression model predicts actual sales value (Hit if ≥ 1.0M units)

🔍 Model Agreement Analysis
✅ Strong Confidence: Both models predict Hit
- Classification probability: 72%
- Regression prediction: 1.50M units
```

### Batch Prediction CSV Columns

| Column | Description |
|--------|-------------|
| `hit_prediction` | 0 or 1 (Classification) |
| `hit_prediction_label` | "Hit" or "Not Hit" (Classification) |
| `hit_probability` | 0.0 to 1.0 (Classification confidence) |
| `predicted_sales` | X.XX (Regression sales in M units) |
| `regression_hit_label` | "Hit" or "Not Hit" (Regression with ≥1.0M threshold) |
| `interpretation` | Agreement analysis |

### Sample CSV Output

```csv
genre,console,predicted_sales,regression_hit_label,hit_prediction_label,interpretation
action,ps4,2.18,Hit,Not Hit,"⚠️ Mixed: Classification=Not Hit, Regression=Hit (≥1.0M)"
sports,xone,1.50,Hit,Hit,"✅ Strong: Both predict Hit (≥1.0M)"
shooter,pc,0.36,Not Hit,Not Hit,"✅ Agreement: Both predict Not Hit (<1.0M)"
```

## 🛠️ Updated Functions

### Streamlit App (`src/app.py`)

```python
def predict_sales(regressor, ...):
    """Returns: (predicted_sales, hit_label)"""
    predicted_sales = regressor.predict(X)[0]
    hit_label = "Hit" if predicted_sales >= 1.0 else "Not Hit"
    return predicted_sales, hit_label
```

### Batch Script (`predict_batch.py`)

```python
# Hit threshold (same as classification)
HIT_THRESHOLD = 1.0

# Apply to regression predictions
result['regression_hit_label'] = (
    result['predicted_sales'] >= HIT_THRESHOLD
).map({False: 'Not Hit', True: 'Hit'})
```

### Test Script (`test_dual_prediction.py`)

```python
HIT_THRESHOLD = 1.0
regression_hit_label = "Hit" if sales >= HIT_THRESHOLD else "Not Hit"

# Compare labels directly
if hit_prediction == regression_hit_label:
    print("✅ Models agree")
else:
    print("⚠️ Mixed signals")
```

## 📚 Interpretation Guide

### When Both Predict Hit
- **High confidence** in success
- Expected sales well above 1.0M
- Classification probability usually >60%
- **Action**: Proceed with confidence

### When Both Predict Not Hit  
- **Low likelihood** of hitting threshold
- Expected sales below 1.0M
- Classification probability usually <40%
- **Action**: Consider alternatives or improvements

### When They Disagree
- **Moderate uncertainty**
- Predicted sales close to 1.0M threshold
- Classification probability typically 40-60%
- **Action**: 
  - Review game concept
  - Consider market research
  - Plan conservative budget
  - Have contingency plans

## 🎓 Key Takeaways

1. **Same Definition**: Both models use ≥1.0M as "Hit"
2. **Compare Directly**: Check if labels match
3. **Use Both Metrics**: Probability AND sales value
4. **Trust Agreement**: High confidence when models agree
5. **Investigate Disagreement**: Extra caution when they differ

## 📊 Example Decision Matrix

| Classification | Regression | Agreement | Decision |
|---------------|------------|-----------|----------|
| Hit (70%) | 1.8M = Hit | ✅ Yes | ✅ Greenlight |
| Hit (55%) | 1.1M = Hit | ✅ Yes | ✅ Proceed |
| Not Hit (35%) | 0.4M = Not Hit | ✅ Yes | ❌ Pass |
| Hit (52%) | 0.9M = Not Hit | ⚠️ No | ⚠️ Review |
| Not Hit (45%) | 1.2M = Hit | ⚠️ No | ⚠️ Analyze |

## 🚀 Quick Commands

```powershell
# Test single game with unified threshold
python test_dual_prediction.py

# Batch predict with model agreement
python predict_batch.py games.csv predictions.csv

# Run Streamlit with unified display
streamlit run src/app.py
```

## ✨ Summary

The unified threshold approach provides:
- ✅ Consistent Hit definition across models
- ✅ Clear agreement/disagreement signals
- ✅ Better decision-making framework
- ✅ Easier interpretation for stakeholders

**Both models now speak the same language: Hit = ≥1.0M units** 🎯
