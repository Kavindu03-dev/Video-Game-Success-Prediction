# Video Game Success Prediction

End-to-end ML project to predict whether a video game will be a Hit (total_sales ≥ 1.0) and explore sales trends via a Streamlit app.

This repository contains the working project under `video-game-success-prediction/`.

## 🚀 Quick Start

```powershell
cd "video-game-success-prediction"

# Setup environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train model & run app
python -m src.train_model
streamlit run app\app.py --server.port 8504
```

Then open http://localhost:8504 in your browser.

## 📁 Project Structure

```
video-game-success-prediction/
├── app/app.py              # Streamlit web application
├── data/vg_sales_2024.csv  # Dataset (place here)
├── notebooks/              # 01-04: EDA, preprocessing, modeling, explainability
├── src/                    # Training & preprocessing code
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── model.pkl              # Trained model (created after training)
```

## 📊 Dataset

Place `vg_sales_2024.csv` in `video-game-success-prediction/data/`.

**Required columns:** total_sales, genre, platform, publisher, critic_score, release_date/release_year
**Optional:** na_sales, eu_sales, jp_sales, other_sales

## 🎯 Features

**Streamlit App:**
- Predict Hit/Not Hit from game attributes
- Visualize sales by genre, platform, region
- Interactive charts and dashboards

**Notebooks:** Complete analysis pipeline from EDA to model explainability

**Model:** Random Forest with OneHot encoding, ~80% accuracy

## 🔧 Troubleshooting

- **"Model not found":** Run `python -m src.train_model` first
- **Memory warnings:** Large dataset with high-cardinality features (normal)
- **Port conflicts:** Use different port: `--server.port 8505`
- **Package errors:** Use Python 3.11+ for pre-built wheels

## 🧪 Testing

```powershell
pip install pytest
pytest -q
```

## License

Educational use only. Comply with dataset license terms.