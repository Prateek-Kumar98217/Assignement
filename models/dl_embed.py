import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from preprocess import preprocessor_nontext

print("Loading data...")
train_df = pd.read_csv("data/train.csv")

state_label = train_df["emotional_state"]
intensity_label = train_df["intensity"]
train_features = train_df.drop(["id", "emotional_state", "intensity"], axis=1)

# 1. Generate Dense Text Embeddings
print("Loading MiniLM and encoding text...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dense_text_embeddings = embedding_model.encode(
    train_df["journal_text"].tolist(), show_progress_bar=True
)

# 2. Process Non-Text Features
print("Processing non-text features...")
train_features_nontext = preprocessor_nontext.fit_transform(train_features)
# Safety check: if the non-text preprocessor returns sparse, convert to dense
if hasattr(train_features_nontext, "toarray"):
    train_features_nontext = train_features_nontext.toarray()

# 3. Combine Features (This is our true X_processed_full now!)
print("Fusing semantic embeddings with metadata...")
X_processed_full = np.hstack((train_features_nontext, dense_text_embeddings))
input_dim = X_processed_full.shape[1]
print(f"Final input dimension after fusion: {input_dim}")

# 4. Label Encoding
state_encoder = LabelEncoder()
state_labels_encoded = state_encoder.fit_transform(state_label)
intensity_encoder = LabelEncoder()
intensity_labels_encoded = intensity_encoder.fit_transform(intensity_label)

num_state_classes = len(state_encoder.classes_)
num_int_classes = len(intensity_encoder.classes_)


# 5. Cascaded Model Definition using SiLU
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

        # Conditional Probability Integration
        intensity_input = torch.cat([shared_rep, state_out], dim=1)
        intensity_out = self.intensity_head(intensity_input)
        return state_out, intensity_out


# 6. Helper Functions
def train_fold(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y_state, batch_y_int in train_loader:
            optimizer.zero_grad()
            state_out, intensity_out = model(batch_x)
            loss_state = criterion(state_out, batch_y_state)
            loss_intensity = criterion(intensity_out, batch_y_int)
            loss = loss_state + loss_intensity
            loss.backward()
            optimizer.step()
    return model


def evaluate_fold(model, val_loader):
    model.eval()
    val_state_probs = []
    val_int_probs = []
    with torch.no_grad():
        # Fixed unpacking: val_loader only yields batch_x
        for batch in val_loader:
            batch_x = batch[0]
            state_out, intensity_out = model(batch_x)
            # Return probabilities, NOT argmax, so uncertainty scores work!
            val_state_probs.append(torch.softmax(state_out, dim=1).cpu().numpy())
            val_int_probs.append(torch.softmax(intensity_out, dim=1).cpu().numpy())

    return np.vstack(val_state_probs), np.vstack(val_int_probs)


# 7. Setup Tensors and K-Fold
X_tensor = torch.FloatTensor(X_processed_full)
y_state_tensor = torch.LongTensor(state_labels_encoded)
y_int_tensor = torch.LongTensor(intensity_labels_encoded)

epochs = 30
batch_size = 32
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_state_probs = np.zeros((len(train_df), num_state_classes))
oof_intensity_probs = np.zeros((len(train_df), num_int_classes))

print("Starting 5-Fold CV for PyTorch MLP (Semantic MiniLM)...")
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_processed_full, state_labels_encoded)
):
    print(f"--- FOLD {fold + 1} ---")

    train_dataset = TensorDataset(
        X_tensor[train_idx], y_state_tensor[train_idx], y_int_tensor[train_idx]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_tensor[val_idx])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DualTargetMLP(
        input_dim=input_dim,
        state_classes=num_state_classes,
        intensity_classes=num_int_classes,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    trained_model = train_fold(model, train_loader, criterion, optimizer, epochs)
    fold_state_probs, fold_int_probs = evaluate_fold(trained_model, val_loader)

    oof_state_probs[val_idx] = fold_state_probs
    oof_intensity_probs[val_idx] = fold_int_probs

# 8. Process Predictions
state_preds_encoded = np.argmax(oof_state_probs, axis=1)
state_predictions = state_encoder.inverse_transform(state_preds_encoded)
confidence_scores = np.max(oof_state_probs, axis=1)
uncertainty_flag = confidence_scores < 0.5

intensity_preds_encoded = np.argmax(oof_intensity_probs, axis=1)
intensity_predictions = intensity_encoder.inverse_transform(intensity_preds_encoded)
confidence_scores_intensity = np.max(oof_intensity_probs, axis=1)
uncertainty_flag_intensity = confidence_scores_intensity < 0.5

# 9. Save the OOF predictions
print("Saving PyTorch OOF predictions...")
os.makedirs("predictions-test", exist_ok=True)
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
    "predictions-test/train_predictions_dl_transformer.csv", index=False
)

# 10. Train Final Model
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
os.makedirs("saved-models/pytorch_transformer", exist_ok=True)
torch.save(final_model.state_dict(), "saved-models/pytorch_transformer/model.pth")
joblib.dump(
    preprocessor_nontext, "saved-models/pytorch_transformer/preprocessor_nontext.joblib"
)
print(
    "Done! You can now run eval.py on predictions-textmeta/train_predictions_dl_transformer.csv"
)
