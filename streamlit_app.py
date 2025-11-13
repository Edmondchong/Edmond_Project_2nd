import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------------------------
# LOAD MODELS & ARTIFACTS
# ---------------------------------------------
st.set_page_config(page_title="SECOM Sensor Drift & Anomaly Dashboard", layout="wide")

imputer = joblib.load("secom_imputer.joblib")
scaler = joblib.load("secom_scaler.joblib")
iso = joblib.load("secom_iso.joblib")

pca_clf = joblib.load("secom_pca_clf.joblib")
clf = joblib.load("secom_rf_clf.joblib")

ae_meta = joblib.load("secom_ae_meta.joblib")
input_dim = ae_meta["input_dim"]

# Tabular Autoencoder class
class TabularAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AE weights
ae = TabularAE(input_dim=input_dim).to(device)
ae.load_state_dict(torch.load("secom_ae_best.pth", map_location=device))
ae.eval()

# LSTM AE metadata
lstm_meta = joblib.load("secom_lstm_meta.joblib")
window_size = lstm_meta["window_size"]

# LSTM Autoencoder class
class LSTM_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h_last, _) = self.encoder_lstm(x)
        z = self.fc_latent(h_last[-1])
        z_seq = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder_lstm(z_seq)
        return self.fc_output(dec_out)

lstm_ae = LSTM_AE(input_dim=input_dim).to(device)
lstm_ae.load_state_dict(torch.load("secom_lstm_ae.pth", map_location=device))
lstm_ae.eval()

# ---------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------
def compute_tabular_ae_score(model, X_scaled):
    X_t = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        recon = model(X_t)
        mse = torch.mean((X_t - recon)**2, dim=1)
    return mse.cpu().numpy()

def compute_lstm_ae_score(model, X_scaled):
    # build window
    if X_scaled.shape[0] < window_size:
        return np.array([0])  # not enough samples
    seq = X_scaled[-window_size:]
    seq = torch.from_numpy(seq).float().unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(seq)
        mse = torch.mean((seq - recon)**2)
    return np.array([float(mse)])

def classify_health(iso, ae, lstm):
    if iso > 0.8 or ae > 0.8 or lstm > 1.2:
        return "🔴 CRITICAL"
    elif iso > 0.5 or ae > 0.5 or lstm > 0.9:
        return "🟡 WARNING"
    else:
        return "🟢 NORMAL"

# ---------------------------------------------
# UI
# ---------------------------------------------
st.title("🔧 SECOM Sensor Drift & Anomaly Detection Dashboard")
st.caption("Smart Manufacturing AI — Powered by IsolationForest, Autoencoder, LSTM Autoencoder, PCA & RandomForest")

uploaded_file = st.file_uploader("Upload a SECOM sensor CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

    st.write("### Uploaded Data (first 20 rows)")
    st.dataframe(df.head(20))

    # Preprocess
    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)
    # Isolation Forest score
    iso_score = -iso.decision_function(X_scaled)
    iso_mean = float(np.mean(iso_score))

    # AE score
    ae_score = compute_tabular_ae_score(ae, X_scaled)
    ae_mean = float(np.mean(ae_score))

    # LSTM AE score
    lstm_score = compute_lstm_ae_score(lstm_ae, X_scaled)
    lstm_mean = float(np.mean(lstm_score))

    # Classifier prediction
    X_pca = pca_clf.transform(X_scaled)
    pred_fail = clf.predict_proba(X_pca)[:, 1]
    fail_mean = float(np.mean(pred_fail))

    # Overall health
    health = classify_health(iso_mean, ae_mean, lstm_mean)

    # -----------------------------------------
    # METRICS
    # -----------------------------------------
    st.subheader("📊 Health Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IsolationForest Score", f"{iso_mean:.4f}")
    col2.metric("AE Reconstruction Error", f"{ae_mean:.4f}")
    col3.metric("LSTM AE Error", f"{lstm_mean:.4f}")
    col4.metric("Fail Probability", f"{fail_mean*100:.2f}%")

    st.subheader("⚠️ Overall Machine State")
    st.markdown(f"## {health}")

    # -----------------------------------------
    # PLOTS
    # -----------------------------------------
    st.subheader("🔍 Histogram of Anomaly Scores")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.hist(iso_score, bins=40, alpha=0.6, label="IsolationForest")
    ax.hist(ae_score, bins=40, alpha=0.6, label="Autoencoder")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔮 Fail Probability Distribution")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.hist(pred_fail, bins=40)
    st.pyplot(fig2)

else:
    st.info("Upload a CSV file to begin analysis.")
