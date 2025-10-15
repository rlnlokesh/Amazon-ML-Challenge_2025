# -----------------------------
# 0) Imports
# -----------------------------
import pandas as pd
import numpy as np
from ast import literal_eval
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# -----------------------------
# 1) Load CSV
# -----------------------------
CSV_PATH = "D:/DA/AAA/train_embed_16.csv"  # replace with your path
df = pd.read_csv(CSV_PATH)


# -----------------------------
# 2) Convert embeddings to numpy arrays and handle missing
# -----------------------------
def to_array_fill_zero(x, expected_dim=1024):
    try:
        if isinstance(x, str):
            arr = np.array(literal_eval(x), dtype=np.float32)
        else:
            arr = np.array(x, dtype=np.float32)
    except Exception:
        arr = np.zeros(expected_dim, dtype=np.float32)
    if arr.size != expected_dim:
        return np.zeros(expected_dim, dtype=np.float32)
    return arr


df['embed_array'] = df['concatenate_embedding'].apply(lambda x: to_array_fill_zero(x, 1024))
X_base = np.vstack(df['embed_array'].values)

# Add extra features: norm, mean, std
extra_features = np.vstack([np.linalg.norm(X_base, axis=1),
                            np.mean(X_base, axis=1),
                            np.std(X_base, axis=1)]).T
X = np.hstack([X_base, extra_features])  # 1027-dim
y = np.log1p(df['price'].values)  # log-transform target

# -----------------------------
# 3) Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# 4) PyTorch Dataset
# -----------------------------
class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = PriceDataset(X_scaled, y)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1) Improved SMAPE Loss
# -----------------------------
class SMAPELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super()._init_()
        self.eps = eps

    def forward(self, y_pred, y_true):
        abs_diff = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2 + self.eps
        return torch.mean(abs_diff / denominator)


criterion = SMAPELoss()


# -----------------------------
# 2) Enhanced MLP Model
# -----------------------------
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[2048, 1024, 512, 256, 128], dropouts=[0.4, 0.4, 0.3, 0.2, 0.1]):
        super()._init_()

        layers = []
        prev_dim = input_dim

        for i, (hidden_dim, dropout) in enumerate(zip(hidden_dims, dropouts)):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


# -----------------------------
# 3) Improved Training Function
# -----------------------------
def train_mlp_improved(model, train_loader, val_loader, n_epochs=150, lr=1e-3, patience=25):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

    best_val_smape = np.inf
    stop_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        y_val_pred_list, y_val_true_list = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss = criterion(pred, yb)
                val_losses.append(val_loss.item())
                y_val_pred_list.append(pred.cpu().numpy())
                y_val_true_list.append(yb.cpu().numpy())

        val_loss = np.mean(val_losses)

        # Calculate SMAPE on original scale
        y_val_pred_np = np.vstack(y_val_pred_list)
        y_val_true_np = np.vstack(y_val_true_list)

        # Convert back from log1p if used during preprocessing
        y_val_pred_orig = np.expm1(y_val_pred_np)  # Remove if no log transformation
        y_val_true_orig = np.expm1(y_val_true_np)  # Remove if no log transformation

        val_smape_epoch = 100 * np.mean(
            2 * np.abs(y_val_pred_orig - y_val_true_orig) /
            (np.abs(y_val_true_orig) + np.abs(y_val_pred_orig) + 1e-8)
        )

        scheduler.step(val_loss)

        # Early stopping
        if val_smape_epoch < best_val_smape:
            best_val_smape = val_smape_epoch
            stop_counter = 0
            torch.save(model.state_dict(), f"best_mlp_{id(model)}.pt")
        else:
            stop_counter += 1

        if stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_loss={val_loss:.4f}, val_smape={val_smape_epoch:.2f}")

    # Load best model
    model.load_state_dict(torch.load(f"best_mlp_{id(model)}.pt"))
    return model, best_val_smape


# -----------------------------
# 4) Create Ensemble
# -----------------------------
def create_ensemble(input_dim):
    models = []
    # Different architectures
    models.append(ImprovedMLP(input_dim, [2048, 1024, 512, 256, 128], [0.4, 0.4, 0.3, 0.2, 0.1]))
    models.append(ImprovedMLP(input_dim, [1536, 768, 384, 192, 96], [0.35, 0.35, 0.25, 0.15, 0.05]))
    models.append(ImprovedMLP(input_dim, [1024, 1024, 512, 256, 128, 64], [0.3, 0.3, 0.25, 0.2, 0.15, 0.1]))
    models.append(ImprovedMLP(input_dim, [2560, 1280, 640, 320, 160], [0.45, 0.4, 0.35, 0.25, 0.15]))
    models.append(ImprovedMLP(input_dim, [1024, 512, 256, 128, 64, 32], [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]))

    return [model.to(device) for model in models]


# -----------------------------
# 5) Improved Ensemble Prediction
# -----------------------------
def ensemble_predict_improved(models, X_tensor):
    """Ensemble prediction with median + mean combination"""
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            pred = m(X_tensor).cpu().numpy().flatten()
            preds.append(pred)

    preds_array = np.array(preds)

    # Use median to reduce outlier effects, then average
    median_preds = np.median(preds_array, axis=0)
    mean_preds = np.mean(preds_array, axis=0)

    # Combine median and mean for robustness
    final_preds = 0.7 * median_preds + 0.3 * mean_preds
    return final_preds


# -----------------------------
# 6) MAIN TRAINING PIPELINE
# -----------------------------
print("Creating ensemble...")
ensemble_models = create_ensemble(X_scaled.shape[1])
ensemble_val_smapes = []

print("\nTraining ensemble models...")
for i, model in enumerate(ensemble_models):
    print(f"\nTraining model {i + 1}/{len(ensemble_models)}")
    trained_model, best_smape = train_mlp_improved(
        model, train_loader, val_loader, n_epochs=150, lr=1e-3, patience=25
    )
    ensemble_val_smapes.append(best_smape)
    print(f"Model {i + 1} best validation SMAPE: {best_smape:.2f}")

# -----------------------------
# 7) FINAL EVALUATION
# -----------------------------
print("\n=== Final Ensemble Evaluation ===")

# Prepare validation data
X_val_tensor = torch.tensor(np.vstack([x for x, _ in val_dataset]), dtype=torch.float32).to(device)
y_val_true_log = np.vstack([y for _, y in val_dataset]).flatten()

# Get ensemble predictions (in log scale)
y_val_pred_log = ensemble_predict_improved(ensemble_models, X_val_tensor)

# Convert back to original scale
y_val_pred_orig = np.expm1(y_val_pred_log)
y_val_true_orig = np.expm1(y_val_true_log)


# Final SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


val_smape = smape(y_val_true_orig, y_val_pred_orig)
print(f"Individual model SMAPEs: {[f'{s:.2f}' for s in ensemble_val_smapes]}")
print(f"Ensemble Validation SMAPE: {val_smape:.2f}")

# # -----------------------------
# # 8) Additional: Check predictions
# # -----------------------------
# print("\n=== Prediction Analysis ===")
# print(f"Prediction range: [{y_val_pred_orig.min():.2f}, {y_val_pred_orig.max():.2f}]")
# print(f"True range: [{y_val_true_orig.min():.2f}, {y_val_true_orig.max():.2f}]")

# # Check for any extreme predictions
# extreme_preds = np.sum((y_val_pred_orig > 1000) | (y_val_pred_orig < 0))
# print(f"Extreme predictions: {extreme_preds}/{len(y_val_pred_orig)}")


# -----------------------------
# 0) Helper to convert embedding and fill empty with zeros
# -----------------------------
import numpy as np
from ast import literal_eval


def to_array_fill_zero(x, size=1024):
    if isinstance(x, str) and len(x) > 2:
        return np.array(literal_eval(x), dtype=np.float32)
    else:
        return np.zeros(size, dtype=np.float32)


# -----------------------------
# 1) Load new CSV
# -----------------------------
import pandas as pd

NEW_CSV = "D:/DA/AAA/test_embed_16.csv"  # replace with your path
df_test = pd.read_csv(NEW_CSV)

# -----------------------------
# 2) Convert embeddings
# -----------------------------
df_test['embed_array'] = df_test['concatenate_embedding'].apply(lambda x: to_array_fill_zero(x, 1024))
X_base_test = np.vstack(df_test['embed_array'].values)

# -----------------------------
# 3) Extra features (optional, same as training)
# -----------------------------
extra_features_test = np.vstack([
    np.linalg.norm(X_base_test, axis=1),
    np.mean(X_base_test, axis=1),
    np.std(X_base_test, axis=1)
]).T

X_test = np.hstack([X_base_test, extra_features_test])
X_test_scaled = scaler.transform(X_test)  # use the scaler from training

# -----------------------------
# 4) Ensemble prediction
# -----------------------------
import torch

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)


def ensemble_predict(models, X_tensor):
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(X_tensor).cpu().numpy().flatten())
    return np.mean(np.vstack(preds), axis=0)


pred_log = ensemble_predict(ensemble_models, X_test_tensor)  # ensemble of 5 models
pred_price = np.expm1(pred_log)  # inverse of log1p if used during training

# -----------------------------
# 5) Save CSV
# -----------------------------
df_test['price'] = pred_price
df_test[['sample_id', 'price']].to_csv("final_test_out.csv", index=False)