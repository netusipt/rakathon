import csv
import sys
import numpy as np
import pandas as pd
import torch
from time_prediction import MyDataset, get_X, get_y
from trainable_module import TrainableModule
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from scipy.stats import norm
import argparse


def predct(model, x):
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
    return y_hat

def get_bin_probs(th, y_hat, std):
    bins = [-float("inf")] + th + [float("inf")]

    probs = []
    # Calculate the probability for each bin
    for i in range(1, len(bins)):
        # Probability in the bin [bins[i-1], bins[i])
        p = norm.cdf(bins[i], loc=y_hat, scale=std) - norm.cdf(bins[i-1], loc=y_hat, scale=std)
        probs.append(p)

    return probs


def write_values_to_csv(values, filename="output.csv"):
    values = [float(v) for v in values]
    
    # Open the file in write mode and use the csv writer to output the values in one row.
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(values)

def test_loss(model):
    data_path = "onco_data.csv"
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

    x, y = next(iter(val_loader))
    y_hat = model(x)

    print("loss:", ((y_hat - y)**2).sum(dim=[0,-1])/len(y_hat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data provided as command-line arguments.')
    parser.add_argument('--x', type=float, required=True, help='A floating point number')
    parser.add_argument('--out', default="output.csv", type=str)
    args = parser.parse_args()
    print(args.x)

    # must be same architecture
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

    # load weights
    model.load_weights("/home/petrn/python/rackathon/main/model_weights.pth")
    mo

    with open("avg_std.txt", "r") as f:
        avg_std = float(f.read())

    print(avg_std)

    torch.manual_seed(42)
    # test_loss(model)


    th = [365, 365 * 4, 365 * 8]
    res = get_bin_probs(th, args.x, avg_std)
    print(res)

    write_values_to_csv(res, args.out)




