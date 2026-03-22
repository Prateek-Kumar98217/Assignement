"""A single function to tabulate and plot the evaluation metrics for a given model"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os


def generate_eval_artifacts(df, model_name):
    os.makedirs(f"artifacts/{model_name}/plots", exist_ok=True)
    os.makedirs(f"artifacts/{model_name}/tables", exist_ok=True)

    eval_df = df.copy()

    errors_only = eval_df[
        (eval_df["error_type"] != "Correct") & (eval_df["error_type"] != "")
    ]
    error_counts = errors_only["error_type"].value_counts().reset_index()
    error_counts.columns = ["error_type", "count"]

    # Error distribution table
    table_path = f"artifacts/{model_name}/tables/error_counts.csv"
    error_counts.to_csv(table_path, index=False)
    print(f"Error distribution for {model_name} saved to {table_path}")

    # Error distribution plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="count", y="error_type", data=error_counts, palette="viridis")
    plt.title(
        f"Distribution of Prediction Errors for {model_name}", fontweight="bold", pad=15
    )
    plt.xlabel("Error Count", labelpad=10)
    plt.ylabel("")
    for index, value in enumerate(error_counts["count"]):
        plt.text(value, index, str(value), va="center")
    plt.tight_layout()
    plot_path = f"artifacts/{model_name}/plots/error_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Error distribution plot for {model_name} saved to {plot_path}")

    # State confusion matrix
    state_labels = sorted(eval_df["emotional_state"].unique())
    cm_state = confusion_matrix(
        eval_df["emotional_state"],
        eval_df["predicted_emotional_state"],
        labels=state_labels,
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_state,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=state_labels,
        yticklabels=state_labels,
    )
    plt.title(
        f"Confusion Matrix for Emotional State - {model_name}",
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Predicted state", labelpad=10)
    plt.ylabel("Actual state", labelpad=10)
    plt.tight_layout()
    cm_state_path = f"artifacts/{model_name}/plots/confusion_matrix_state.png"
    plt.savefig(cm_state_path, dpi=300)
    plt.close()
    print(
        f"Confusion matrix for emotional state of {model_name} saved to {cm_state_path}"
    )

    # Intensity confusion matrix
    intensity_labels = sorted(eval_df["intensity"].unique())
    cm_intensity = confusion_matrix(
        eval_df["intensity"], eval_df["predicted_intensity"], labels=intensity_labels
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_intensity,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=intensity_labels,
        yticklabels=intensity_labels,
    )
    plt.title(
        f"Confusion Matrix for Intensity - {model_name}",
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Predicted intensity", labelpad=10)
    plt.ylabel("Actual intensity", labelpad=10)
    plt.tight_layout()
    cm_intensity_path = f"artifacts/{model_name}/plots/confusion_matrix_intensity.png"
    plt.savefig(cm_intensity_path, dpi=300)
    plt.close()
    print(
        f"Confusion matrix for intensity of {model_name} saved to {cm_intensity_path}"
    )

    print(f"\nGenerating F1 Scores and Classification Reports for {model_name}...")

    # 1. Emotional State Report
    state_report_dict = classification_report(
        eval_df["emotional_state"],
        eval_df["predicted_emotional_state"],
        labels=state_labels,
        output_dict=True,
        zero_division=0,
    )
    state_report_df = pd.DataFrame(state_report_dict).transpose()
    state_report_path = f"artifacts/{model_name}/tables/classification_report_state.csv"
    state_report_df.to_csv(state_report_path)

    print(f"\n[EMOTIONAL STATE METRICS]")
    print(
        classification_report(
            eval_df["emotional_state"],
            eval_df["predicted_emotional_state"],
            zero_division=0,
        )
    )

    # 2. Intensity Report
    intensity_report_dict = classification_report(
        eval_df["intensity"],
        eval_df["predicted_intensity"],
        labels=intensity_labels,
        output_dict=True,
        zero_division=0,
    )
    intensity_report_df = pd.DataFrame(intensity_report_dict).transpose()
    intensity_report_path = (
        f"artifacts/{model_name}/tables/classification_report_intensity.csv"
    )
    intensity_report_df.to_csv(intensity_report_path)

    print(f"\n[INTENSITY METRICS]")
    print(
        classification_report(
            eval_df["intensity"], eval_df["predicted_intensity"], zero_division=0
        )
    )

    print(f"Saved classification reports to artifacts/{model_name}/tables/")

    return error_counts
