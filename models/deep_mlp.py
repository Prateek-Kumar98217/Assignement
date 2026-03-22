import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from preprocess import preprocessor_textmeta

print("Loading data...")
train_df = pd.read_csv("data/train.csv")

state_label = train_df["emotional_state"]
intensity_label = train_df["intensity"]
train_features = train_df.drop(["id", "emotional_state", "intensity"], axis=1)

# label encoding for emotional state and intensity
state_encoder = LabelEncoder()
state_labels_encoded = state_encoder.fit_transform(state_label)
intensity_encoder = LabelEncoder()
intensity_labels_encoded = intensity_encoder.fit_transform(intensity_label)


class DualTargetMLP(nn.Module):
    def __init__(self, input_dim, state_classes, intensity_classes):
        super().__init__()

        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.3),
        )

        self.state_head = nn.Linear(64, state_classes)
        self.intensity_head = nn.Sequential(
            nn.Linear(64 + state_classes, 32),
            nn.SiLU(),
            nn.Linear(32, intensity_classes),
        )

    def forward(self, x):
        shared_rep = self.shared_layer(x)
        state_out = self.state_head(shared_rep)
        intensity_input = torch.cat([shared_rep, state_out], dim=1)
        intensity_out = self.intensity_head(intensity_input)
        return state_out, intensity_out


def train_fold(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y_state, batch_y_int in train_loader:
            optimizer.zero_grad()
            state_logits, int_logits = model(batch_x)

            # Combine losses natively
            loss = criterion(state_logits, batch_y_state) + criterion(
                int_logits, batch_y_int
            )
            loss.backward()
            optimizer.step()
    return model


def evaluate_fold(model, val_loader):
    model.eval()
    val_state_probs = []
    val_int_probs = []
    with torch.no_grad():
        for batch_x in val_loader:
            state_logits, int_logits = model(batch_x[0])
            val_state_probs.append(torch.softmax(state_logits, dim=1).numpy())
            val_int_probs.append(torch.softmax(int_logits, dim=1).numpy())

    # Return the stacked probabilities
    return np.vstack(val_state_probs), np.vstack(val_int_probs)


# main script
X_processed_full = preprocessor_textmeta.fit_transform(train_features).toarray()
input_dim = X_processed_full.shape[1]
print(f"Input dimension after preprocessing: {input_dim}")
num_state_classes = len(state_encoder.classes_)
num_int_classes = len(intensity_encoder.classes_)

X_tensor = torch.FloatTensor(X_processed_full)
y_state_tensor = torch.LongTensor(state_labels_encoded)
y_int_tensor = torch.LongTensor(intensity_labels_encoded)

# Setup K-Fold Arrays
epochs = 30
batch_size = 32
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_state_probs = np.zeros((len(train_df), num_state_classes))
oof_intensity_probs = np.zeros((len(train_df), num_int_classes))

print("Starting 5-Fold CV for PyTorch MLP...")
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_processed_full, state_labels_encoded)
):
    print(f"--- FOLD {fold + 1} ---")

    # Dataloaders
    train_dataset = TensorDataset(
        X_tensor[train_idx], y_state_tensor[train_idx], y_int_tensor[train_idx]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_tensor[val_idx])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = DualTargetMLP(
        input_dim=input_dim,
        state_classes=num_state_classes,
        intensity_classes=num_int_classes,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Train and Evaluate using our clean functions!
    trained_model = train_fold(model, train_loader, criterion, optimizer, epochs)
    fold_state_probs, fold_int_probs = evaluate_fold(trained_model, val_loader)

    # Store results
    oof_state_probs[val_idx] = fold_state_probs
    oof_intensity_probs[val_idx] = fold_int_probs

# 4. Process Predictions
state_preds_encoded = np.argmax(oof_state_probs, axis=1)
state_predictions = state_encoder.inverse_transform(state_preds_encoded)
confidence_scores = np.max(oof_state_probs, axis=1)
uncertainty_flag = confidence_scores < 0.5

intensity_preds_encoded = np.argmax(oof_intensity_probs, axis=1)
intensity_predictions = intensity_encoder.inverse_transform(intensity_preds_encoded)
confidence_scores_intensity = np.max(oof_intensity_probs, axis=1)
uncertainty_flag_intensity = confidence_scores_intensity < 0.5

# 5. Save the OOF predictions
print("Saving PyTorch OOF predictions...")
os.makedirs("predictions-textmeta", exist_ok=True)
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
    "predictions-textmeta/train_predictions_dl_mlp.csv", index=False
)

# Final model
print("Training final PyTorch model on 100% of data...")
final_model = DualTargetMLP(
    input_dim=input_dim,
    state_classes=num_state_classes,
    intensity_classes=num_int_classes,
)
final_optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=0.01)
final_loader = DataLoader(
    TensorDataset(X_tensor, y_state_tensor, y_int_tensor),
    batch_size=batch_size,
    shuffle=True,
)

final_model = train_fold(
    final_model, final_loader, nn.CrossEntropyLoss(), final_optimizer, epochs
)

print("Saving final model and preprocessor...")
os.makedirs("saved-models/pytorch_mlp", exist_ok=True)
torch.save(final_model.state_dict(), "saved-models/pytorch_mlp/model.pth")
joblib.dump(preprocessor_textmeta, "saved-models/pytorch_mlp/preprocessor.joblib")
print(
    "Done! You can now run eval.py on predictions-textmeta/train_predictions_dl_mlp.csv"
)
