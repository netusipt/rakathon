import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from trainable_module import TrainableModule
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, HuberLoss, MSELoss
from torch.utils.data import random_split



def get_X(df):
    map_age = lambda x: x - 102
    age = df["vekova_kategorie_10let_dg"].apply(map_age)

    stadium_map = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "X": 4,
        "Y": 5,
    }
    stadium = df["stadium"].apply(lambda x: stadium_map[x] if x in stadium_map else 0)

    tumor_size_map = {
        "0": 0,
        "is": 0,
        "isD": 0,
        "isL": 0,
        "isP": 0,
        "1m": 2,
        "1a": 3,
        "1a2": 3,
        "1b": 4,
        "1c": 5,
        "1": 5,
        "2": 6,
        "2a": 6,
        "2b": 6,
        "2c": 6,
        "3": 7,
        "3b": 7,
        "4": 8,
        "4a": 8,
        "4b": 8,
        "4c": 8,
        "4d": 8,
        "X": 0,
        "a": 0,
    }

    tumour_map = lambda x: tumor_size_map[x] if x in tumor_size_map else 0
    tumour_size = df["tnm_klasifikace_t_kod"].apply(tumour_map)

    er_status = df["je_pl_hormo"]

    map_grading = lambda x: x - 1
    grading = df["grading"].apply(map_grading)

    modality = [75, 77, 79, 81, 83, 85, 87, 89, 91, 93]
    symbols = ['O', 'R', 'T', 'C', 'H', 'I']
    for symbol in symbols:
        df[symbol + '_count'] = df.iloc[:, modality].apply(lambda row: list(row).count(symbol), axis=1)

    O_count = torch.tensor(df["O_count"].values.astype(np.float32)).unsqueeze(1)
    R_count = torch.tensor(df["R_count"].values.astype(np.float32)).unsqueeze(1)
    T_count = torch.tensor(df["T_count"].values.astype(np.float32)).unsqueeze(1)
    C_count = torch.tensor(df["C_count"].values.astype(np.float32)).unsqueeze(1)
    H_count = torch.tensor(df["H_count"].values.astype(np.float32)).unsqueeze(1)
    I_count = torch.tensor(df["I_count"].values.astype(np.float32)).unsqueeze(1)

    pl_delka = torch.tensor(df["pl_delka"].values.astype(np.float32)).unsqueeze(1)

    stadium_tensor = torch.tensor(stadium.values.astype(np.float32)).unsqueeze(1)
    age_tensor = torch.tensor(age.values.astype(np.float32)).unsqueeze(1)
    tumour_size_tensor = torch.tensor(tumour_size.values.astype(np.float32)).unsqueeze(1)
    er_status_tensor = torch.tensor(er_status.values.astype(np.float32)).unsqueeze(1)
    grading_tensor = torch.tensor(grading.values.astype(np.float32)).unsqueeze(1)


    X = torch.cat(
            (stadium_tensor, age_tensor, tumour_size_tensor, er_status_tensor, grading_tensor, O_count, R_count,
            T_count, C_count, H_count, I_count, pl_delka),
        dim=1)

    return X

def get_y(df):
    return torch.tensor(df["time_datum_dg_to_zahajeni_nl"].values.astype(np.float32)).unsqueeze(1)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    torch.manual_seed(42)
    data_path = "/home/petrn/python/rackathon/main/viktor/new/rakathon/onco_data.csv"
    df = pd.read_csv(data_path)

    df = df[(df["je_pl"] == True)]
    df = df[~df["time_datum_dg_to_zahajeni_nl"].isna()]

    X = get_X(df)
    y = get_y(df)

    X_mean = X.mean(dim=0, keepdim=True)  # shape: (1, num_features)
    X_std = X.std(dim=0, keepdim=True)    # shape: (1, num_features)

    # To avoid division by zero (in case some feature has zero variance), add a tiny value like 1e-8.
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    X = X_normalized

    dataset = MyDataset(
        X,
        y,
    )

    total_size = len(dataset)
    train_size = int(0.84 * total_size)
    val_size = total_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)

    class Model(TrainableModule):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, 256)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(256, 64)
            self.fc3 = torch.nn.Linear(64, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    model = Model(input_dim=12, output_dim=1)



    model.configure(
        optimizer=AdamW(model.parameters(), lr=0.001),
        # loss=HuberLoss(delta=0.1),
        loss=MSELoss(
            # weight=torch.tensor([1.0, 2.60]),
            # label_smoothing=0.1,
            ),
        # metrics=["accuracy"],
    )

    model.fit(train_loader, epochs=20)

    x, y = next(iter(val_loader))
    y_hat = model(x)

    print("loss:", ((y_hat - y)**2).sum(dim=[0,-1])/len(y_hat))


    model.save_weights("model_weights.pth")

    diff = y_hat.squeeze() - y.squeeze()
    diff = (diff).numpy(force=True)
    sigma = np.std(diff, ddof=1)

    print(sigma)

    with open("avg_std.txt", "w") as f:
        f.write(str(float(sigma)))

