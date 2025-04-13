import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR



# === Load and prepare dataset ===
df = pd.read_csv("/home/petrn/python/rackathon/main/viktor/processed_dataset.csv")

# Map to binary classification: relapse (1) vs. no relapse (0)
def get_binary_label(row):
    if row['no_relaps'] == 1.0:
        return 0  # No relapse
    else:
        return 1  # Any relapse

df["label"] = df.apply(get_binary_label, axis=1)
X = df.drop(columns=["relaps_1_5", "relaps_5_10", "relaps_10_15", "no_relaps", "label"])
y = df["label"]

print(f"Class distribution (original): {pd.Series(y).value_counts().sort_index()}")
print(f"0: No relapse, 1: Relapse")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Class distribution in training data: {pd.Series(y_train).value_counts().sort_index()}")

# === Dataset & DataLoader ===
class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Použití původních dat pro trénování, bez oversamplingu
train_ds = CancerDataset(X_train, y_train)
test_ds = CancerDataset(X_test, y_test)

# Create validation loader for monitoring 
val_size = int(0.1 * len(X_train))
train_indices = range(len(X_train) - val_size)
val_indices = range(len(X_train) - val_size, len(X_train))

val_ds = torch.utils.data.Subset(train_ds, val_indices)
train_ds_final = torch.utils.data.Subset(train_ds, train_indices)

train_loader = DataLoader(train_ds_final, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

# === Define Neural Network ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

model = MLP(input_dim=X.shape[1])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Add scheduler AFTER optimizer is defined
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# === Training loop ===
def train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, epochs=10):
    model.train()
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)
                total_val_loss += loss.item()
        
        # Step the scheduler based on validation loss
        scheduler.step(total_val_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# Call the updated training function with scheduler
train(model, train_loader, val_loader, optimizer, scheduler, loss_fn)

def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())
    print(classification_report(y_true, y_pred, target_names=[
        "no_relapse", "relapse"
    ]))

evaluate(model, test_loader)