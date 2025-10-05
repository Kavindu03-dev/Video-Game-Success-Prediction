# Model Roles and Responsibilities

## 🎯 Clear Separation of Concerns

### Classification Model (RandomForest)
**Purpose:** Determines if a game is Hit or Not Hit

- **Output:** Binary prediction (Hit/Not Hit) with probability
- **Decision:** Based on learned patterns from successful games
- **Accuracy:** 93%
- **Use Case:** "Will this game be a hit?"

**Example Output:**
```
Prediction: Hit
Probability: 75.3%
```

---

### Regression Model (GradientBoosting)
**Purpose:** Predicts actual sales value in millions

- **Output:** Continuous value (e.g., 2.18M units)
- **Decision:** Estimates how many copies will sell
- **Metrics:** R²=0.33, MAE=0.26M, RMSE=0.70M
- **Use Case:** "How much will this game sell?"

**Example Output:**
```
Predicted Sales: 2.18 million units
```

---

## 🔄 How They Work Together

### Scenario 1: High Confidence Hit
```
Classification: Hit (85% probability)
Regression: 2.5M units
→ Strong success predicted
```

### Scenario 2: Borderline Case
```
Classification: Not Hit (45% probability)
Regression: 0.9M units
→ Close to threshold, niche success possible
```

### Scenario 3: Clear Not Hit
```
Classification: Not Hit (15% probability)
Regression: 0.3M units
→ Lower sales expected
```

---

## 📊 In the Streamlit App

The app shows both predictions side by side:

1. **Classification Section:**
   - Hit/Not Hit label
   - Probability percentage
   - Visual progress bar

2. **Regression Section:**
   - Predicted sales value
   - No Hit/Not Hit label (this is classification's job!)

3. **Combined Insights:**
   - Classification determines success
   - Regression shows expected magnitude
   - Together they provide complete picture

---

## ✅ Key Takeaways

| Model | What It Answers | Output Type | Use For |
|-------|----------------|-------------|---------|
| **Classification** | Will it be a Hit? | Hit/Not Hit + Probability | Decision making |
| **Regression** | How much will it sell? | Sales in millions | Revenue forecasting |

**Remember:** 
- ✅ Classification = Hit/Not Hit determination
- ✅ Regression = Sales amount prediction
- ❌ Don't use regression for Hit/Not Hit classification
- ✅ Use both together for complete insights

---

## 🚀 Updated Files

All scripts now follow this separation:

- ✅ `src/app.py` - Streamlit app
- ✅ `test_dual_prediction.py` - Test script
- ✅ `predict_batch.py` - Batch predictions
- ✅ `verify_models.py` - Model verification

**Last Updated:** October 5, 2025
