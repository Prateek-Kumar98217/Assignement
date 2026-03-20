import os
import pandas as pd
from eval import generate_eval_artifacts

# Load the data
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
# Note: Pointing to the new sequential CSV we generated in the previous step
pred_df = pd.read_csv("predictions-textmeta/train_predictions_sequential_clf.csv")

# Merge and clean up
analysis_df = pd.merge(train_df, pred_df, on="id", how="inner")

# We drop the 'actual_' prefixed columns from the pred_df to keep the
# ground truth from train_df (emotional_state, intensity) as our reference.
cols_to_drop = [col for col in analysis_df.columns if col.startswith("actual_")]
print(f"Dropping redundant columns: {cols_to_drop}")
analysis_df = analysis_df.drop(columns=cols_to_drop)


def flag_sequential_errors(row):
    errors = []
    state_wrong = row["emotional_state"] != row["predicted_emotional_state"]
    intensity_wrong = row["intensity"] != row["predicted_intensity"]

    # 1. Check for Sequential/Cascade Error
    if state_wrong and intensity_wrong:
        errors.append("Cascade_Error (State & Intensity)")
    elif state_wrong:
        errors.append("State_Only_Wrong")
    elif intensity_wrong:
        errors.append("Intensity_Only_Wrong")

    # 2. Check Uncertainty Flags
    # Since it's sequential, we check both confidence scores
    if row.get("confidence_score", 1.0) < 0.5:
        errors.append("Uncertain_State")
    if row.get("confidence_score_intensity", 1.0) < 0.5:
        errors.append("Uncertain_Intensity")

    return " | ".join(errors) if errors else "Correct"


# Apply the refined logic
analysis_df["error_type"] = analysis_df.apply(flag_sequential_errors, axis=1)

# Generate artifacts (Confusion matrices, classification reports, etc.)
# We use a new suffix to distinguish from the standard classifier
error_counts = generate_eval_artifacts(analysis_df, "xgboost_clf_textmeta_seq")

print("\n--- Error Distribution ---")
print(analysis_df["error_type"].value_counts())
