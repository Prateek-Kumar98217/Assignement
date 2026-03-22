import os
import joblib
import pandas as pd
import numpy as np
from preprocess import preprocessor_textmeta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict

# Load the data as dataframes
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Define labels and features
state_label = train_df["emotional_state"]
intensity_label = train_df["intensity"]
train_features = train_df.drop(columns=["id", "emotional_state", "intensity"])
test_features = test_df.drop(columns=["id"])

# Encode labels
# For intensity, we can treat it as a regression problem, so we will not encode it.
state_encoder = LabelEncoder()
state_label_encoded = state_encoder.fit_transform(state_label)

# Create seprate models for emotional state and intensity prediction
classification_model = XGBClassifier(
    random_state=42, eval_metric="mlogloss", max_depth=6, n_estimators=100
)
regression_model = XGBRegressor(
    random_state=42, eval_metric="rmse", max_depth=6, n_estimators=100
)

# create pipelines for both models
classification_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_textmeta),
        ("classifier", classification_model),
    ]
)
regression_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor_textmeta), ("regressor", regression_model)]
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Generating Out-of-Fold (OOF) predictions via 5-Fold CV...")

# 2. Generate OOF Class Predictions and Probabilities
# predict_proba gives us the scores for the confidence/uncertainty logic
state_predictions_encoded = cross_val_predict(
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

# 3. Generate OOF Regression Predictions
intensity_predictions = cross_val_predict(
    regression_pipeline, train_features, intensity_label, cv=kf
)

# --- Process Results ---

# Decode classification
state_predictions = state_encoder.inverse_transform(state_predictions_encoded)
confidence_scores = np.max(state_probs, axis=1)
uncertainty_flag = confidence_scores < 0.5

# Round and clip regression
intensity_predictions_rounded = np.clip(np.round(intensity_predictions), 1, 5).astype(
    int
)

# 4. Save the Comparison (These are now "honest" predictions)
print("Saving OOF predictions for comparison...")
comparison_dftrain = pd.DataFrame(
    {
        "id": train_df["id"],
        "actual_emotional_state": state_label,
        "predicted_emotional_state": state_predictions,
        "actual_intensity": intensity_label,
        "predicted_intensity": intensity_predictions_rounded,
        "confidence_score": confidence_scores,
        "uncertainty_flag": uncertainty_flag,
    }
)
os.makedirs("predictions-test", exist_ok=True)
comparison_dftrain.to_csv(
    "predictions-test/train_predictions_xgboost_reg.csv", index=False
)

# 5. Final Fit and Save
# We still do a final fit on the WHOLE dataset so the saved .joblib file is as "smart" as possible for the future
print("Training final models on full data for export...")
classification_pipeline.fit(train_features, state_label_encoded)
regression_pipeline.fit(train_features, intensity_label)

os.makedirs("saved-models/xgboost_reg", exist_ok=True)
joblib.dump(classification_pipeline, "saved-models/xgboost_reg/state.joblib")
joblib.dump(regression_pipeline, "saved-models/xgboost_reg/intensity.joblib")
print("Process complete.")
