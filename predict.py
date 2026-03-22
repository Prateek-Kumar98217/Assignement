"""A script to generate predictions for the test and train data using the best trained models"""

import os
import joblib
import pandas as pd
import numpy as np

test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

output_dir = "final-prediction"

state_pipeline = joblib.load("saved-models/xgboost_clf/state.joblib")
intensity_pipeline = joblib.load("saved-models/xgboost_clf/intensity.joblib")
state_encoder = joblib.load("saved-models/xgboost_clf/state_encoder.joblib")
intensity_encoder = joblib.load("saved-models/xgboost_clf/intensity_encoder.joblib")
#predictions on training data
state_predictions_train = state_pipeline.predict(train_df)
state_prediction_confidence_train = np.max(state_pipeline.predict_proba(train_df), axis = 1)
intensity_predictions_train = intensity_pipeline.predict(train_df)
intensity_prediction_confidence_train = np.max(intensity_pipeline.predict_proba(train_df), axis = 1)

#predictions on test data
state_predictions_test = state_pipeline.predict(test_df)
state_prediction_confidence_test = np.max(state_pipeline.predict_proba(test_df), axis = 1)
intensity_predictions_test = intensity_pipeline.predict(test_df)
intensity_prediction_confidence_test = np.max(intensity_pipeline.predict_proba(test_df), axis = 1)

# Decision engine: Reasons for such implementation discussed in README.md
def apply_decision_engine(state, intensity, is_uncertain):
    if is_uncertain:
        return pd.Series(["prompt_manual_reflection", "end_of_day"])
    
    if state in ["calm", "focused", "neutral"]:
        if intensity <= 3:
            return pd.Series(["no_action_needed", "none"])
        else:
            return pd.Series(["maintain_momentum", "none"])
            
    if state in ["overwhelmed", "restless"]:
        if intensity >= 4:
            return pd.Series(["suggest_short_break", "immediate"])
        else:
            return pd.Series(["review_daily_schedule", "next_transition"])
            
    if state == "mixed":
        return pd.Series(["log_gratitude_journal", "evening"])
        
    return pd.Series(["no_action_needed", "none"])

combined_confidence_train = (state_prediction_confidence_train + intensity_prediction_confidence_train) / 2
is_uncertain_train = combined_confidence_train < 0.5

combined_confidence_test = (state_prediction_confidence_test + intensity_prediction_confidence_test) / 2
is_uncertain_test = combined_confidence_test < 0.5

final_state_train = state_encoder.inverse_transform(state_predictions_train)
final_intensity_train = intensity_encoder.inverse_transform(intensity_predictions_train)
final_state_test = state_encoder.inverse_transform(state_predictions_test)
final_intensity_test = intensity_encoder.inverse_transform(intensity_predictions_test)

train_predictions = pd.DataFrame({
    "id": train_df["id"],
    "predicted_state": final_state_train,
    "predicted_intensity": final_intensity_train,
    "confidence": combined_confidence_train,
    "uncertain_flag": is_uncertain_train
})

test_predictions = pd.DataFrame({
    "id": test_df["id"],
    "predicted_state": final_state_test,
    "predicted_intensity": final_intensity_test,
    "confidence": combined_confidence_test,
    "uncertain_flag": is_uncertain_test
})

train_predictions[["what_to_do", "when_to_do"]] = train_predictions.apply(
    lambda row: apply_decision_engine(
        row["predicted_state"],
        row["predicted_intensity"],
        row["uncertain_flag"]
        ), axis=1
)

test_predictions[["what_to_do", "when_to_do"]] = test_predictions.apply(
    lambda row: apply_decision_engine(
        row["predicted_state"],
        row["predicted_intensity"],
        row["uncertain_flag"]
    ), axis=1
)

os.makedirs(output_dir, exist_ok=True)
train_predictions.to_csv(os.path.join(output_dir, "train_predictions.csv"), index=False)
test_predictions.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

print("Predictions saved to", output_dir)