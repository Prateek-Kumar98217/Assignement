import os
import joblib
import pandas as pd
import numpy as np
from preprocess import preprocessor_textmeta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold  # Added
from xgboost import XGBClassifier

# Load the data
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Define labels and features
state_label = train_df["emotional_state"]
intensity_label = train_df["intensity"]
train_features = train_df.drop(columns=["id", "emotional_state", "intensity"])

# Encode labels
state_encoder = LabelEncoder()
state_label_encoded = state_encoder.fit_transform(state_label)
intensity_encoder = LabelEncoder()
intensity_label_encoded = intensity_encoder.fit_transform(intensity_label)

# Define models
classification_model = XGBClassifier(
    random_state=42, eval_metric="mlogloss", max_depth=6, n_estimators=100
)
classification_model_intensity = XGBClassifier(
    random_state=42, eval_metric="mlogloss", max_depth=6, n_estimators=100
)

# Create pipelines
classification_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_textmeta),
        ("classifier", classification_model),
    ]
)
classification_pipeline_intensity = Pipeline(
    steps=[
        ("preprocessor", preprocessor_textmeta),
        ("classifier", classification_model_intensity),
    ]
)

# K-fold cross-validation for OOF predictions
print("\nGenerating Out-of-Fold (OOF) predictions via 5-Fold CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Predictions for Emotional State
state_preds_encoded = cross_val_predict(
    classification_pipeline,
    train_features,
    state_label_encoded,
    cv=skf,
    method="predict",
)
state_probs = cross_val_predict(
    classification_pipeline,
    train_features,
    state_label_encoded,
    cv=skf,
    method="predict_proba",
)

# Predictions for Intensity (now as a classifier)
intensity_preds_encoded = cross_val_predict(
    classification_pipeline_intensity,
    train_features,
    intensity_label_encoded,
    cv=skf,
    method="predict",
)
intensity_probs = cross_val_predict(
    classification_pipeline_intensity,
    train_features,
    intensity_label_encoded,
    cv=skf,
    method="predict_proba",
)

# Process predictions and confidence scores
state_predictions = state_encoder.inverse_transform(state_preds_encoded)
confidence_scores = np.max(state_probs, axis=1)
uncertainty_flag = confidence_scores < 0.5

intensity_predictions = intensity_encoder.inverse_transform(intensity_preds_encoded)
confidence_scores_intensity = np.max(intensity_probs, axis=1)
uncertainty_flag_intensity = confidence_scores_intensity < 0.5

# Save the Comparison (Non-leaked predictions)
print("Saving OOF predictions for comparison...")
comparison_dftrain = pd.DataFrame(
    {
        "id": train_df["id"],
        "actual_emotional_state": state_label,
        "predicted_emotional_state": state_predictions,
        "confidence_score": confidence_scores,
        "uncertainty_flag": uncertainty_flag,
        "actual_intensity": intensity_label,
        "predicted_intensity": intensity_predictions,
        "confidence_score_intensity": confidence_scores_intensity,
        "uncertainty_flag_intensity": uncertainty_flag_intensity,
    }
)

os.makedirs("predictions-textmeta", exist_ok=True)
comparison_dftrain.to_csv(
    "predictions-textmeta/train_predictions_xgboost_clf.csv", index=False
)

# Final model training on full data
print("Training final models on the full dataset...")
classification_pipeline.fit(train_features, state_label_encoded)
classification_pipeline_intensity.fit(train_features, intensity_label_encoded)

print("Saving the trained models...")
os.makedirs("saved-models/xgboost_clf", exist_ok=True)
joblib.dump(classification_pipeline, "saved-models/xgboost_clf/state.joblib")
joblib.dump(
    classification_pipeline_intensity, "saved-models/xgboost_clf/intensity.joblib"
)
joblib.dump(state_encoder, "saved-models/xgboost_clf/state_encoder.joblib")
joblib.dump(intensity_encoder, "saved-models/xgboost_clf/intensity_encoder.joblib")
print("Process complete.")