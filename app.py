from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
import os

app = Flask(__name__)

# Define the same model architecture as in time_prediction.py
class Model(torch.nn.Module):
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
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

# Function to calculate bin probabilities
def get_bin_probs(th, y_hat, std):
    bins = [-float("inf")] + th + [float("inf")]
    probs = []
    # Calculate the probability for each bin
    for i in range(1, len(bins)):
        # Probability in the bin [bins[i-1], bins[i])
        p = norm.cdf(bins[i], loc=y_hat, scale=std) - norm.cdf(bins[i-1], loc=y_hat, scale=std)
        probs.append(p)
    return probs

# Load the clinical risk factor model
try:
    rf_model = joblib.load('clinical_rf_model.pkl')
    print("Clinical RF model loaded successfully")
except Exception as e:
    print(f"Error loading RF model: {e}")
    rf_model = None

# Load the time prediction model
try:
    time_model = Model(input_dim=12, output_dim=1)
    time_model.load_weights("/home/petrn/python/rackathon/main/model_weights.pth")
    time_model.eval()  # Set to evaluation mode
    print("Time prediction model loaded successfully")
    
    # Load the standard deviation for the time prediction
    avg_std = 477.0
    print(f"Average std loaded: {avg_std}")
except Exception as e:
    print(f"Error loading time prediction model: {e}")
    time_model = None
    avg_std = None

# # Try to load feature normalization parameters
# try:
#     feature_means = np.load("feature_means.npy")
#     feature_stds = np.load("feature_stds.npy")
#     print("Feature normalization parameters loaded")
# except Exception as e:
#     print(f"Error loading feature normalization parameters: {e}")
#     feature_means = None
#     feature_stds = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predicting...")
    if rf_model is None:
        return render_template('error.html', error="Clinical risk model not available")
    
    try:
        # Extract form data
        vek = int(request.form.get('vek', 2))
        er_status = int(request.form.get('er_status', 1))
        tumour_size = int(request.form.get('tumour_size', 2))
        grading = int(request.form.get('grading', 2))
        lym = int(request.form.get('lym', 0))
        stadium = int(request.form.get('stadium', 2))
        pl_delka = int(request.form.get('pl_delka', 3000)) / 365
        pl_mamo = int(request.form.get('pl_mamo', 0))
        
        # Treatment modalities
        o_count = int(request.form.get('O_count', 0))
        r_count = int(request.form.get('R_count', 0))
        t_count = int(request.form.get('T_count', 0))
        c_count = int(request.form.get('C_count', 0))
        h_count = int(request.form.get('H_count', 0))
        i_count = int(request.form.get('I_count', 0))
        
        # Create feature array for RF model (in the same order as expected by the model)
        rf_features = np.array([
            vek, er_status, tumour_size, grading, lym, stadium, 
            pl_delka, pl_mamo, o_count, r_count, t_count, 
            c_count, h_count, i_count
        ]).reshape(1, -1)
        
        # Get prediction from RF model
        risk_probability = rf_model.predict_proba(rf_features)[0, 1]
        risk_score = round(risk_probability * 100, 1)
        

        
        # Create feature array for time prediction model
        # Map values according to time_prediction.py feature encoding
        stadium_val = min(stadium, 5)  # Map to {0,1,2,3,4,5}
        tumour_size_mapped = min(tumour_size + 2, 8)  # Adjust mapping
        
        print("--------------------------------------------------")
        print(stadium_val, vek-2, tumour_size_mapped, er_status, 
            grading, o_count, r_count, t_count, c_count, h_count, 
            i_count, pl_delka)
        print("--------------------------------------------------")
        # Create feature tensor
        time_features = np.array([
            stadium_val, vek-2, tumour_size_mapped, er_status, 
            grading, o_count, r_count, t_count, c_count, h_count, 
            i_count, pl_delka
        ]).reshape(1, -1)

        # stadium_tensor = torch.tensor(stadium).unsqueeze(0)
        # age_tensor = torch.tensor(vek).unsqueeze(0)
        # tumour_size_tensor = torch.tensor(tumour_size).unsqueeze(0)
        # er_status_tensor = torch.tensor(er_status).unsqueeze(0)
        # grading_tensor = torch.tensor(grading).unsqueeze(0)
        # O_count = torch.tensor(o_count)
        # R_count = torch.tensor(r_count)
        # T_count = torch.tensor(t_count)
        # C_count = torch.tensor(c_count)
        # H_count = torch.tensor(h_count)
        # I_count = torch.tensor(i_count)
        # pl_delka = torch.tensor(pl_delka).unsqueeze(0)

        X = torch.tensor(
    [stadium, vek, tumour_size, er_status, grading, o_count, r_count,
    t_count, c_count, h_count, i_count, pl_delka * 365], dtype=torch.float32).unsqueeze(0)
        
        # Normalize features
        # time_features_norm = (time_features - feature_means) / (feature_stds + 1e-8)
        time_features_tensor = torch.tensor(time_features, dtype=torch.float32)
        
        # Get prediction
        with torch.no_grad():
            predicted_time = time_model(X).item()

            print(predicted_time)
        # predicted_time = 365 * 4
        
        # Get bin probabilities
        thresholds = [365, 365 * 4, 365 * 8]  # 1 year, 4 years, 8 years
        print("--------------------------------------------------")
        bin_probs = get_bin_probs(thresholds, predicted_time, avg_std)
        print("--------------------------------------------------")
        print(bin_probs)

        # Format time bins as percentages
        time_bins = {
            "1-4 roky": round((bin_probs[1] * 100) * (risk_score/100), 1), 
            "4-8 let": round((bin_probs[2] * 100) * (risk_score/100), 1), 
            ">8 let": round((bin_probs[3] * 100) * (risk_score/100), 1)
        }
        
        # Update time bins
        # time_keys = list(time_bins.keys())
        # for i, prob in enumerate(bin_probs):
        #     time_bins[time_keys[i]] = round(prob * 100, 1)
        
        # Prepare data for display
        data = {
            'Věková kategorie': vek,
            'ER status': 'Pozitivní' if er_status == 1 else 'Negativní',
            'Velikost nádoru (T)': tumour_size,
            'Grading': grading + 1,  # Convert back to 1-3 scale for display
            'Lymfatické uzliny': 'Pozitivní' if lym == 1 else 'Negativní',
            'Stadium': stadium,
            'Délka plánu (měsíce)': round(pl_delka * 365, 1),
            'Mamografický screening': 'Ano' if pl_mamo == 1 else 'Ne',
            'Operace (O)': o_count,
            'Radioterapie (R)': r_count,
            'Transplantace (T)': t_count,
            'Chemoterapie (C)': c_count,
            'Hormonální terapie (H)': h_count,
            'Imunoterapie (I)': i_count
        }
        
        return render_template('result.html', risk_score=risk_score, data=data, time_bins=time_bins)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)