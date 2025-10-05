"""
Batch prediction script for both classification and regression.
Reads games from CSV, predicts both hit probability and total sales.

Usage:
    python predict_batch.py input.csv output.csv
"""
import sys
import joblib
import pandas as pd
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_batch.py input.csv output.csv")
        print("\nInput CSV should have columns:")
        print("  - genre (e.g., 'action', 'sports')")
        print("  - console (e.g., 'ps4', 'xone')")
        print("  - publisher (e.g., 'nintendo', 'ea')")
        print("  - developer (e.g., 'nintendo', 'ubisoft')")
        print("  - critic_score (0-10)")
        print("  - release_year (e.g., 2020)")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Load models
    project_root = Path(__file__).parent
    classifier_path = project_root / 'models' / 'best_model.joblib'
    regressor_path = project_root / 'models' / 'best_regressor.joblib'
    
    if not classifier_path.exists():
        print(f"Error: Classifier not found: {classifier_path}")
        print("Run: python src/train.py")
        sys.exit(1)
    
    if not regressor_path.exists():
        print(f"Error: Regressor not found: {regressor_path}")
        print("Run: python src/train_regression.py")
        sys.exit(1)
    
    print("Loading models...")
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    
    # Load input data
    print(f"Reading input from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Check required columns
    required = ['genre', 'console', 'publisher', 'developer', 'critic_score', 'release_year']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)
    
    # Normalize text features
    for col in ['genre', 'console', 'publisher', 'developer']:
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip().str.lower()
    
    # Ensure numeric types
    if 'critic_score' in df.columns:
        df['critic_score'] = pd.to_numeric(df['critic_score'], errors='coerce').clip(0, 10)
    if 'release_year' in df.columns:
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(2020).astype(int)
    
    print(f"Predicting for {len(df)} games...")
    
    # Make predictions
    X = df[required]
    
    # Classification
    if hasattr(classifier, 'predict_proba'):
        hit_proba = classifier.predict_proba(X)[:, 1]
    else:
        hit_proba = classifier.predict(X).astype(float)
    
    hit_pred = (hit_proba >= 0.5).astype(int)
    
    # Regression
    sales_pred = regressor.predict(X)
    sales_pred = sales_pred.clip(min=0)  # No negative sales
    
    # Add predictions to dataframe
    result = df.copy()
    result['hit_prediction'] = hit_pred
    result['hit_prediction_label'] = result['hit_prediction'].map({0: 'Not Hit', 1: 'Hit'})
    result['hit_probability'] = hit_proba
    result['predicted_sales'] = sales_pred
    
    # Add insights based on classification and sales prediction
    def get_insights(row):
        class_label = 'Hit' if row['hit_prediction'] == 1 else 'Not Hit'
        sales = row['predicted_sales']
        
        if class_label == 'Hit':
            if sales >= 1.5:
                return f'✅ Hit predicted - Strong sales potential ({sales:.2f}M)'
            elif sales >= 1.0:
                return f'✅ Hit predicted - Moderate sales ({sales:.2f}M)'
            else:
                return f'⚠️ Hit predicted but lower sales estimate ({sales:.2f}M)'
        else:
            if sales >= 0.8:
                return f'📊 Not Hit - Close to threshold ({sales:.2f}M)'
            else:
                return f'📉 Not Hit - Lower sales expected ({sales:.2f}M)'
    
    result['insights'] = result.apply(get_insights, axis=1)
    
    # Save output
    print(f"Saving results to: {output_path}")
    result.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total games:        {len(result)}")
    print(f"\nClassification Model (Hit/Not Hit):")
    print(f"  Predicted Hit:     {(result['hit_prediction'] == 1).sum()} ({(result['hit_prediction'] == 1).mean():.1%})")
    print(f"  Predicted Not Hit: {(result['hit_prediction'] == 0).sum()} ({(result['hit_prediction'] == 0).mean():.1%})")
    print(f"\nRegression Model (Sales Predictions):")
    print(f"  Average: {result['predicted_sales'].mean():.2f}M units")
    print(f"  Median:  {result['predicted_sales'].median():.2f}M units")
    print(f"  Max:     {result['predicted_sales'].max():.2f}M units")
    print(f"  Min:     {result['predicted_sales'].min():.2f}M units")
    print(f"\nGames above 1.0M:   {(result['predicted_sales'] >= 1.0).sum()} ({(result['predicted_sales'] >= 1.0).mean():.1%})")
    print(f"Games below 1.0M:   {(result['predicted_sales'] < 1.0).sum()} ({(result['predicted_sales'] < 1.0).mean():.1%})")
    print("\nInsights Distribution:")
    print(result['insights'].value_counts().to_string())
    print("="*60)
    print(f"\n✓ Results saved to: {output_path}")

if __name__ == '__main__':
    main()
