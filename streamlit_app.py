import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from datetime import datetime
import tempfile
import os

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="SECOM Sensor Drift & Anomaly Dashboard", layout="wide")

# --------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------
imputer = joblib.load("secom_imputer.joblib")
scaler = joblib.load("secom_scaler.joblib")
iso = joblib.load("secom_iso.joblib")
pca_clf = joblib.load("secom_pca_clf.joblib")
clf = joblib.load("secom_rf_clf.joblib")

ae_meta = joblib.load("secom_ae_meta.joblib")
input_dim = ae_meta["input_dim"]

# -------- Tabular Autoencoder --------
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

ae = TabularAE(input_dim=input_dim).to(device)
ae.load_state_dict(torch.load("secom_ae_best.pth", map_location=device))
ae.eval()

# -------- LSTM Autoencoder --------
lstm_meta = joblib.load("secom_lstm_meta.joblib")
window_size = lstm_meta["window_size"]

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

# --------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------
def compute_tabular_ae_score(model, X_scaled):
    X_t = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        recon = model(X_t)
        mse = torch.mean((X_t - recon)**2, dim=1)
    return mse.cpu().numpy()

def compute_tabular_ae_recon_and_error(model, X_scaled):
    X_t = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        recon = model(X_t)
        err = (X_t - recon)**2
    return recon.cpu().numpy(), err.cpu().numpy()

def compute_lstm_ae_score(model, X_scaled):
    if X_scaled.shape[0] < window_size:
        return np.array([0.0])
    seq = X_scaled[-window_size:]
    seq = torch.from_numpy(seq).float().unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(seq)
        mse = torch.mean((seq - recon)**2)
    return np.array([float(mse)])

def classify_health(iso_mean, ae_mean, lstm_mean):
    if iso_mean > 0.8 or ae_mean > 0.8 or lstm_mean > 1.2:
        return "🔴 CRITICAL"
    elif iso_mean > 0.5 or ae_mean > 0.5 or lstm_mean > 0.9:
        return "🟡 WARNING"
    else:
        return "🟢 NORMAL"

def compute_sensor_feature_importance(pca_model, rf_model):
    pc_importance = rf_model.feature_importances_
    components = pca_model.components_
    return np.abs(components.T @ pc_importance)

# --------------------------------------------------------
# PDF REPORT GENERATOR
# --------------------------------------------------------
def generate_pdf(iso_mean, ae_mean, lstm_mean, fail_mean, health):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = tmp_file.name

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, 750, "SECOM Equipment Health Diagnostic Report")
    c.setFont("Helvetica", 12)
    c.drawString(30, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.line(30, 720, 580, 720)

    c.drawString(30, 700, f"IsolationForest Score: {iso_mean:.4f}")
    c.drawString(30, 680, f"AE Reconstruction Error: {ae_mean:.4f}")
    c.drawString(30, 660, f"LSTM AE Error: {lstm_mean:.4f}")
    c.drawString(30, 640, f"Fail Probability: {fail_mean*100:.2f}%")
    c.drawString(30, 620, f"Overall Health State: {health}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, 580, "AI Health Interpretation:")
    c.setFont("Helvetica", 11)

    if "CRITICAL" in health:
        msg = "Severe anomaly detected. Immediate engineering inspection required."
    elif "WARNING" in health:
        msg = "Moderate drift detected. Recommend maintenance scheduling."
    else:
        msg = "Machine is stable. No abnormal behavior observed."

    c.drawString(30, 560, msg)

    c.showPage()
    c.save()
    return pdf_path

# --------------------------------------------------------
# UI START
# --------------------------------------------------------
st.title("🔧 SECOM Sensor Drift & Anomaly Detection Dashboard")
uploaded_file = st.file_uploader("Upload SECOM CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)
    X_pca = pca_clf.transform(X_scaled)

    # Scores
    iso_score = -iso.decision_function(X_scaled)
    ae_score = compute_tabular_ae_score(ae, X_scaled)
    lstm_score = compute_lstm_ae_score(lstm_ae, X_scaled)

    iso_mean = float(np.mean(iso_score))
    ae_mean = float(np.mean(ae_score))
    lstm_mean = float(np.mean(lstm_score))

    pred_fail = clf.predict_proba(X_pca)[:, 1]
    fail_mean = float(np.mean(pred_fail))

    health = classify_health(iso_mean, ae_mean, lstm_mean)

    # AE error matrix
    _, ae_err_matrix = compute_tabular_ae_recon_and_error(ae, X_scaled)

    # Sensor importance
    sensor_importance = compute_sensor_feature_importance(pca_clf, clf)

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "🧪 Anomaly Scores",
        "📉 Drift",
        "🎯 PCA Visualization",
        "📡 SPC Charts",
        "⭐ Feature Importance",
        "🔎 SHAP Analysis"
    ])

    # ######################################################
    #  SHAP TAB (SAFE VERSION — PCA SHAP ONLY)
    # ######################################################
    with tab7:
        st.subheader("🔎 SHAP — PCA Component Explainability")

        explainer = shap.TreeExplainer(clf)

        # Use a small subset for SHAP speed
        X_shap = X_pca[:300]
        shap_values = explainer.shap_values(X_shap)[1]  

        pc_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        max_components = len(pc_names)

        if max_components <= 1:
            st.warning("Not enough PCA components for SHAP.")
        else:
            top_n_pc = st.slider(
                "Top N PCA components",
                min_value=1,
                max_value=max_components,
                value=min(5, max_components)
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_values[:, :top_n_pc], X_shap[:, :top_n_pc],
                              feature_names=pc_names[:top_n_pc], show=False)
            st.pyplot(fig)

    # ######################################################
    # PDF EXPORT
    # ######################################################
    st.subheader("📄 Export Full Diagnostic Report")

    if st.button("Generate PDF Report"):
        pdf_path = generate_pdf(iso_mean, ae_mean, lstm_mean, fail_mean, health)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Report", f, file_name="SECOM_Report.pdf")
        os.remove(pdf_path)

else:
    st.info("Upload CSV to begin analysis.")
