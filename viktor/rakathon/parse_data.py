import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

import numpy as np

class OnkoDataset(Dataset):
    def __init__(self, age, er_status, tumour_size, grading, y):
        self.age = age
        self.er_status = er_status
        self.tumour_size = tumour_size
        self.grading = grading

        self.y = y

        assert len(age) == len(er_status) == len(tumour_size) == len(y), "All input tensors must have the same length."
        assert y.sum(axis=1).max() == 1, "Output tensor must be one-hot encoded."
        assert y.sum(axis=1).mean() == 1, "Each row must sum to one."

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.age[idx], self.er_status[idx], self.tumour_size[idx], self.grading[idx]), self.y[idx]

tumor_size_map = {
    "0":  0,
    "is": 0,
    "isD": 0,
    "isL": 0,
    "isP": 0,

    "1m": 2,
    "1a": 3,
    "1a2": 3,
    "1b": 4,
    "1c": 5,
    "1":  5,

    "2":  6,
    "2a":  6,
    "2b":  6,
    "2c":  6,

    "3":  7,
    "3b":  7,

    "4":  8,
    "4a": 8,
    "4b": 8,
    "4c": 8,
    "4d": 8,

    "X":   0,
    "a":   0
}

def get_y(df):
    df["relaps"] = ((df[["je_disp", "je_nl"]].sum(axis=1) == 2) & (df.iloc[:, 150] / 365 > 1)).astype(int)
    df["relaps_1_5"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 <= 5)).astype(int)
    df["relaps_5_10"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 > 5) & (df.iloc[:, 150] / 365 <= 10)).astype(int)
    df["relaps_10_15"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 > 10) & (df.iloc[:, 150] / 365 <= 15)).astype(int)

    s1 = df["relaps_1_5"]
    s2 = df["relaps_5_10"]
    s3 = df["relaps_10_15"]

    col4 = ~(s1 | s2 | s3).astype(bool).values
    col1 = s1.astype(int).values
    col2 = s2.astype(int).values
    col3 = s3.astype(int).values

    one_hot_np = np.column_stack([col1, col2, col3, col4])
    y = torch.tensor(one_hot_np, dtype=torch.float32)

    return y

tumour_map = lambda x: tumor_size_map[x] if x in tumor_size_map else 0
map_age = lambda x: x - 102
map_grading = lambda x: x-1


def to_one_hot(series, max_value=None) -> torch.Tensor:
    if not max_value:
        max_value = series.max()+1
    x = torch.tensor(series.values, dtype=torch.long)
    x = one_hot(x, num_classes=max_value)
    return x


def create_dataset(df: pd.DataFrame) -> OnkoDataset:
    age = to_one_hot(df["vekova_kategorie_10let_dg"].apply(map_age))
    er_status = to_one_hot(df["je_pl_hormo"])
    tumour_size = to_one_hot(df["tnm_klasifikace_t_kod"].apply(tumour_map))
    grading = to_one_hot(df["grading"].apply(map_grading))

    y = get_y(df)

    return OnkoDataset(age, er_status, tumour_size, grading, y)


def save_dataset_to_csv(dataset: OnkoDataset, filepath: str) -> None:
    """
    Save the OnkoDataset to a CSV file.
    
    Args:
        dataset: The OnkoDataset to save
        filepath: Path where the CSV will be saved
    """
    # Convert tensors to numpy arrays
    age_np = dataset.age.numpy()
    er_status_np = dataset.er_status.numpy()
    tumour_size_np = dataset.tumour_size.numpy()
    grading_np = dataset.grading.numpy()
    y_np = dataset.y.numpy()
    
    # Create column names for features
    age_cols = [f'age_{i}' for i in range(age_np.shape[1])]
    er_cols = [f'er_status_{i}' for i in range(er_status_np.shape[1])]
    tumour_cols = [f'tumour_size_{i}' for i in range(tumour_size_np.shape[1])]
    grading_cols = [f'grading_{i}' for i in range(grading_np.shape[1])]
    y_cols = ['relaps_1_5', 'relaps_5_10', 'relaps_10_15', 'no_relaps']
    
    # Create DataFrames for each feature
    age_df = pd.DataFrame(age_np, columns=age_cols)
    er_df = pd.DataFrame(er_status_np, columns=er_cols)
    tumour_df = pd.DataFrame(tumour_size_np, columns=tumour_cols)
    grading_df = pd.DataFrame(grading_np, columns=grading_cols)
    y_df = pd.DataFrame(y_np, columns=y_cols)
    
    # Concatenate all DataFrames
    result_df = pd.concat([age_df, er_df, tumour_df, grading_df, y_df], axis=1)
    
    # Save to CSV
    result_df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")


if __name__ == "__main__":
    data_path = "/home/petrn/python/rackathon/main/viktor/rakathon/onco_data.csv"
    df = pd.read_csv(data_path)

    dataset = create_dataset(df)
    
    # Save dataset to CSV
    output_path = "processed_dataset.csv"
    save_dataset_to_csv(dataset, output_path)
    
    # Existing code for demonstration
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    x, y = next(iter(dataloader))
    print(x)
    print(y)