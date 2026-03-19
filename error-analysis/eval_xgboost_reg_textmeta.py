import pandas as pd
from eval import generate_eval_artifacts

# Load the data as dataframes
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
pred_df = pd.read_csv("predictions-textmeta/train_predictions_xgboost_reg.csv")

# Merge the actual and predicted dataframes on the 'id' column and drop any columns that start with 'actual' to avoid confusion in the analysis
analysis_df = pd.merge(train_df, pred_df, on="id", how="inner")
cols_to_drop = [col for col in analysis_df.columns if col.startswith("actual")]
print(f"Dropping columns: {cols_to_drop}")
analysis_df = analysis_df.drop(columns=cols_to_drop)


# function to mark the errors and uncertainty in the predictions, we will use this to analyze the error distribution and confusion matrix later
def flag_error_type(row):
    errors = []
    if row["emotional_state"] != row["predicted_emotional_state"]:
        errors.append("State")
    if row["intensity"] != row["predicted_intensity"]:
        errors.append("Intensity")
    if row["uncertainty_flag"]:
        errors.append("Uncertain")
    return " & ".join(errors)


analysis_df["error_type"] = analysis_df.apply(flag_error_type, axis=1)
error_counts = generate_eval_artifacts(analysis_df, "xgboost_reg_textmeta")
print(error_counts)
