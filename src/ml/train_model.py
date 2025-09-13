import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.ml.features import add_technical_indicators
from src.ml.labels import make_targets

def prepare_xy(df):
    df2 = add_technical_indicators(df)
    df2 = make_targets(df2, mode='classification')
    feature_cols = [c for c in df2.columns if c not in ('Date','label','next_return')]
    X = df2[feature_cols]
    y = df2['label']
    return df2, X, y, feature_cols

def train_and_save(df: pd.DataFrame, model_path='models/rf_model.joblib'):
    df2, X, y, feature_cols = prepare_xy(df)
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)

    joblib.dump({'pipeline': pipeline, 'feature_cols': feature_cols}, model_path)
    print(f"✅ Model trained and saved to {model_path}")
    return pipeline
