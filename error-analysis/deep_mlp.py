import pandas as pd
from eval import generate_eval_artifacts

# Load the data as dataframes
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
pred_df = pd.read_csv("predictions-test/train_predictions_dl_mlp.csv")

# Merge the actual and predicted dataframes on the 'id' column
analysis_df = pd.merge(train_df, pred_df, on="id", how="inner")
cols_to_drop = [col for col in analysis_df.columns if col.startswith("actual")]
print(f"Dropping columns: {cols_to_drop}")
analysis_df = analysis_df.drop(columns=cols_to_drop)


# Function to mark the errors and uncertainty in the predictions
def flag_error_type(row):
    errors = []
    if row["emotional_state"] != row["predicted_emotional_state"]:
        errors.append("State_Wrong")
    if row["intensity"] != row["predicted_intensity"]:
        errors.append("Intensity_Wrong")
    # Dynamically catch any column that implies uncertainty
    for col in row.index:
        if "uncertain" in col.lower() and row[col] == True:
            label = (
                col.replace("_flag", "").replace("_uncertainty", "_Uncertain").title()
            )
            errors.append(label)
    return " & ".join(errors) if errors else "Correct"


analysis_df["error_type"] = analysis_df.apply(flag_error_type, axis=1)

error_counts = generate_eval_artifacts(analysis_df, "deep_mlp_textmeta")
print(error_counts)
