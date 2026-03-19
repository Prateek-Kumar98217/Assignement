import os
import joblib
import pandas as pd
import numpy as np
from preprocess import preprocessor_textmeta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier

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

# Train the models
print("Training the models...")
classification_pipeline.fit(train_features, state_label_encoded)
regression_pipeline.fit(train_features, intensity_label)

# generate predictions
print(
    "Generating predictions for training data to see performance and check underfitting..."
)
# state predictions are encoded, we will decode them back to original labels later for better interpretability
state_predictions_encoded = classification_pipeline.predict(train_features)
state_predictions = state_encoder.inverse_transform(state_predictions_encoded)
state_probabilities = classification_pipeline.predict_proba(train_features)
confidence_scores = np.max(state_probabilities, axis=1)
uncertainty_flag = confidence_scores < 0.5
# intensity predictions are continuous values, thus we round them to the nearest integer and clip to the range of 1-5
intensity_predictions = regression_pipeline.predict(train_features)
intensity_predictions_rounded = np.clip(np.round(intensity_predictions), 1, 5).astype(
    int
)

# compare predictions with actual labels to see performance and check underfitting
print("Comparing predictions with actual labels...")
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
comparison_dftrain.to_csv(
    "predictions-textmeta/train_predictions_xgboost_reg.csv", index=False
)
print(
    "Comparison for training data saved to predictions-textmeta/train_predictions_xgboost_reg.csv"
)
print("Saving the trained models...")
os.makedirs("saved-models/xgboost_reg", exist_ok=True)
joblib.dump(classification_pipeline, "saved-models/xgboost_reg/state.joblib")
joblib.dump(regression_pipeline, "saved-models/xgboost_reg/intensity.joblib")
print("Trained models saved.")
