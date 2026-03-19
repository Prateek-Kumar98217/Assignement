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
# Treating both emotional state and intensity as categorical variables, we will encode them.
state_encoder = LabelEncoder()
state_label_encoded = state_encoder.fit_transform(state_label)
intensity_encoder = LabelEncoder()
intensity_label_encoded = intensity_encoder.fit_transform(intensity_label)

# Create seprate models for emotional state and intensity prediction
classification_model = XGBClassifier(
    random_state=42, eval_metric="mlogloss", max_depth=6, n_estimators=100
)
classification_model_intensity = XGBClassifier(
    random_state=42, eval_metric="mlogloss", max_depth=6, n_estimators=100
)

# create pipelines for both models
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

# Train the models
print("Training the models...")
classification_pipeline.fit(train_features, state_label_encoded)
classification_pipeline_intensity.fit(train_features, intensity_label_encoded)

# generate predictions
print(
    "Generating predictions for training data to see performance and check underfitting..."
)
# state predictions are encoded, we will decode them back to original labels later
state_predictions_encoded = classification_pipeline.predict(train_features)
state_predictions = state_encoder.inverse_transform(state_predictions_encoded)
state_probabilities = classification_pipeline.predict_proba(train_features)
confidence_scores = np.max(state_probabilities, axis=1)
uncertainty_flag = confidence_scores < 0.5
# concidering intesity values as categorical, we will decode them back to original labels later
intensity_predictions_encoded = classification_pipeline_intensity.predict(
    train_features
)
intensity_predictions = intensity_encoder.inverse_transform(
    intensity_predictions_encoded
)
intensity_probabilities = classification_pipeline_intensity.predict_proba(
    train_features
)
confidence_scores_intensity = np.max(intensity_probabilities, axis=1)
uncertainty_flag_intensity = confidence_scores_intensity < 0.5

# compare predictions with actual labels to see performance and check underfitting
print("Comparing predictions with actual labels...")
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
comparison_dftrain.to_csv(
    "predictions-textmeta/train_predictions_xgboost_clf.csv", index=False
)
print(
    "Comparison for training data saved to predictions-textmeta/train_predictions_xgboost_clf.csv"
)
