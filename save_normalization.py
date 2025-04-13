import pandas as pd
import numpy as np
import torch

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

if __name__ == "__main__":
    data_path = "onco_data.csv"
    df = pd.read_csv(data_path)

    df = df[(df["je_pl"] == True)]
    df = df[~df["time_datum_dg_to_zahajeni_nl"].isna()]

    X = get_X(df)

    X_mean = X.mean(dim=0).numpy()
    X_std = X.std(dim=0).numpy()

    # Save normalization parameters
    np.save("feature_means.npy", X_mean)
    np.save("feature_stds.npy", X_std)
    
    print("Normalization parameters saved!") 