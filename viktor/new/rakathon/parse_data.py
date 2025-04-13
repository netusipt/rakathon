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

        # assert (
        #     len(age) == len(er_status) == len(tumour_size) == len(y)
        # ), "All input tensors must have the same length."
        # assert y.sum(axis=1).max() == 1, "Output tensor must be one-hot encoded."
        # assert y.sum(axis=1).mean() == 1, "Each row must sum to one."

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.age[idx],
            self.er_status[idx],
            self.tumour_size[idx],
            self.grading[idx],
        ), self.y[idx]


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


def get_y(df):
    df["relaps"] = (
        (df[["je_disp", "je_nl"]].sum(axis=1) == 2) & (df.iloc[:, 150] / 365 > 1)
    ).astype(int)
    df["relaps_1_5"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 <= 5)).astype(int)
    df["relaps_5_10"] = (
        (df["relaps"] == 1)
        & (df.iloc[:, 150] / 365 > 5)
        & (df.iloc[:, 150] / 365 <= 10)
    ).astype(int)
    df["relaps_10_15"] = (
        (df["relaps"] == 1)
        & (df.iloc[:, 150] / 365 > 10)
        & (df.iloc[:, 150] / 365 <= 15)
    ).astype(int)

    s1 = df["relaps_1_5"]
    s2 = df["relaps_5_10"]
    s3 = df["relaps_10_15"]

    col4 = ~(s1 | s2 | s3).astype(bool).values
    col1 = s1.astype(int).values
    col2 = s2.astype(int).values
    col3 = s3.astype(int).values

    one_hot_np = np.column_stack([col1, col2, col3, col4])
    y = torch.tensor(one_hot_np, dtype=torch.float32)
    y = torch.argmax(y, dim=1)

    return y


tumour_map = lambda x: tumor_size_map[x] if x in tumor_size_map else 0
map_age = lambda x: x - 102
map_grading = lambda x: x - 1


def to_one_hot(series, max_value=None) -> torch.Tensor:
    if not max_value:
        max_value = series.max() + 1
    x = torch.tensor(series.values, dtype=torch.long)
    x = one_hot(x, num_classes=max_value)
    return x


# def create_dataset(df: pd.DataFrame) -> OnkoDataset:
#     age = to_one_hot(df["vekova_kategorie_10let_dg"].apply(map_age))
#     er_status = to_one_hot(df["je_pl_hormo"])
#     tumour_size = to_one_hot(df["tnm_klasifikace_t_kod"].apply(tumour_map))
#     grading = to_one_hot(df["grading"].apply(map_grading))

#     y = get_y(df)

#     return OnkoDataset(age, er_status, tumour_size, grading, y)


def to_index_tensor(series) -> torch.Tensor:
    return torch.tensor(series.values, dtype=torch.long)


def create_dataset(df: pd.DataFrame) -> OnkoDataset:
    age = to_index_tensor(df["vekova_kategorie_10let_dg"].apply(map_age))
    er_status = to_index_tensor(df["je_pl_hormo"])
    tumour_size = to_index_tensor(df["tnm_klasifikace_t_kod"].apply(tumour_map))
    grading = to_index_tensor(df["grading"].apply(map_grading))

    y = get_y(df)

    return OnkoDataset(age, er_status, tumour_size, grading, y)


if __name__ == "__main__":
    data_path = "onco_data.csv"
    df = pd.read_csv(data_path)

    dataset = create_dataset(df)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    x, y = next(iter(dataloader))
    print(x)
    print(y)
