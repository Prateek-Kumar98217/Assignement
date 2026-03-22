import os
import joblib
import pandas as pd
import numpy as np
from preprocess import preprocessor_textmeta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier

# --- Load and Setup ---
train_df = pd.read_csv("data/train.csv")
state_label = train_df["emotional_state"]
intensity_label = train_df["intensity"]
train_features = train_df.drop(columns=["id", "emotional_state", "intensity"])

state_encoder = LabelEncoder()
state_label_encoded = state_encoder.fit_transform(state_label)
intensity_encoder = LabelEncoder()
intensity_label_encoded = intensity_encoder.fit_transform(intensity_label)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Stage 1: Emotional State (Base Model) ---
print("Stage 1: Generating OOF predictions for Emotional State...")
state_clf = XGBClassifier(
    random_state=42, eval_metric="mlogloss", max_depth=6, n_estimators=100
)
state_pipe = Pipeline(
    [("preprocessor", preprocessor_textmeta), ("classifier", state_clf)]
)

# OOF predictions for State (to be used as a feature for Intensity)
oof_state_preds = cross_val_predict(
    state_pipe, train_features, state_label_encoded, cv=skf, method="predict"
)
state_probs = cross_val_predict(
    state_pipe, train_features, state_label_encoded, cv=skf, method="predict_proba"
)

# --- Stage 2: Intensity (Sequential Model) ---
print("Stage 2: Generating OOF predictions for Intensity using State predictions...")

# We add the OOF state predictions as a new feature column
# Note: We use the encoded integers so the XGBoost model can process them
train_features_with_state = train_features.copy()
train_features_with_state["predicted_state_feat"] = oof_state_preds

intensity_clf = XGBClassifier(
    random_state=42, eval_metric="mlogloss", max_depth=6, n_estimators=100
)
intensity_pipe = Pipeline(
    [("preprocessor", preprocessor_textmeta), ("classifier", intensity_clf)]
)

# OOF predictions for Intensity
oof_intensity_preds = cross_val_predict(
    intensity_pipe,
    train_features_with_state,
    intensity_label_encoded,
    cv=skf,
    method="predict",
)
intensity_probs = cross_val_predict(
    intensity_pipe,
    train_features_with_state,
    intensity_label_encoded,
    cv=skf,
    method="predict_proba",
)

# --- Process Results ---
state_predictions = state_encoder.inverse_transform(oof_state_preds)
confidence_scores = np.max(state_probs, axis=1)
intensity_predictions = intensity_encoder.inverse_transform(oof_intensity_preds)
confidence_scores_intensity = np.max(intensity_probs, axis=1)

# Save Comparison
comparison_dftrain = pd.DataFrame(
    {
        "id": train_df["id"],
        "actual_emotional_state": state_label,
        "predicted_emotional_state": state_predictions,
        "confidence_score": confidence_scores,
        "actual_intensity": intensity_label,
        "predicted_intensity": intensity_predictions,
        "confidence_score_intensity": confidence_scores_intensity,
    }
)
os.makedirs("predictions-test", exist_ok=True)
comparison_dftrain.to_csv(
    "predictions-test/train_predictions_sequential_clf.csv", index=False
)

# --- Final Production Training ---
print("Training final sequential models on full dataset...")

# 1. Fit State Model on full data
state_pipe.fit(train_features, state_label_encoded)

# 2. Get State predictions on full data to train the Intensity Model
full_train_state_preds = state_pipe.predict(train_features)
train_features_with_state_full = train_features.copy()
train_features_with_state_full["predicted_state_feat"] = full_train_state_preds

# 3. Fit Intensity Model on full data (including the predicted state feature)
intensity_pipe.fit(train_features_with_state_full, intensity_label_encoded)

# --- Save Models ---
os.makedirs("saved-models/xgboost_sequential", exist_ok=True)
joblib.dump(state_pipe, "saved-models/xgboost_sequential/state.joblib")
joblib.dump(intensity_pipe, "saved-models/xgboost_sequential/intensity.joblib")
print("Sequential models saved.")
