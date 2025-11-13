import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D PCA

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="SECOM Sensor Drift & Anomaly Detection Dashboard",
    layout="centered"        # <<< FIXED HERE
)

# ---------------------------------------------
# LOAD MODELS & ARTIFACTS
# ---------------------------------------------
imputer = joblib.load("secom_imputer.joblib")
scaler = joblib.load("secom_scaler.joblib")
iso = joblib.load("secom_iso.joblib")

pca_clf = joblib.load("secom_pca_clf.joblib")
clf = joblib.load("secom_rf_clf.joblib")

ae_meta = joblib.load("secom_ae_meta.joblib")
input_dim = ae_meta["input_dim"]

# ---- Tabular Autoencoder ----
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

# ---- LSTM Autoencoder ----
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

# ---------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------
def compute_tabular_ae_score(model, X_scaled: np.ndarray) -> np.ndarray:
    X_t = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        recon = model(X_t)
        mse = torch.mean((X_t - recon)**2, dim=1)
    return mse.cpu().numpy()

def compute_tabular_ae_recon_and_error(model, X_scaled: np.ndarray):
    X_t = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        recon = model(X_t)
        err = (X_t - recon)**2
    return recon.cpu().numpy(), err.cpu().numpy()

def compute_lstm_ae_score(model, X_scaled: np.ndarray) -> np.ndarray:
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
    sensor_importance = np.abs(components.T @ pc_importance)
    return sensor_importance

# ---------------------------------------------
# UI
# ---------------------------------------------
st.title("🔧 SECOM Sensor Drift & Anomaly Detection Dashboard")
st.caption("Smart Manufacturing AI — IsolationForest · Autoencoder · LSTM-AE · PCA · RandomForest")

uploaded_file = st.file_uploader("Upload a SECOM sensor CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)

    iso_score = -iso.decision_function(X_scaled)
    iso_mean = float(np.mean(iso_score))

    ae_score = compute_tabular_ae_score(ae, X_scaled)
    ae_mean = float(np.mean(ae_score))

    lstm_score = compute_lstm_ae_score(lstm_ae, X_scaled)
    lstm_mean = float(np.mean(lstm_score))

    X_pca = pca_clf.transform(X_scaled)
    pred_fail = clf.predict_proba(X_pca)[:, 1]
    fail_mean = float(np.mean(pred_fail))

    health = classify_health(iso_mean, ae_mean, lstm_mean)

    _, ae_err_matrix = compute_tabular_ae_recon_and_error(ae, X_scaled)
    sensor_importance = compute_sensor_feature_importance(pca_clf, clf)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Overview", "🧪 Anomaly Scores", "📉 Drift Analysis",
         "🎯 PCA Visualization", "📡 Sensor Trends / SPC", "⭐ Feature Importance"]
    )

    # TAB 1
    with tab1:
        st.subheader("📊 Machine Health Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("IsolationForest Score", f"{iso_mean:.4f}")
        col2.metric("AE Reconstruction Error", f"{ae_mean:.4f}")
        col3.metric("LSTM AE Error", f"{lstm_mean:.4f}")
        col4.metric("Fail Probability", f"{fail_mean*100:.2f}%")

        st.markdown(f"## {health}")
        st.dataframe(df.head(20))

    # TAB 2
    with tab2:
        st.subheader("🧪 Histogram of Anomaly Scores")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(iso_score, bins=40, alpha=0.6, label="IsolationForest")
        ax.hist(ae_score, bins=40, alpha=0.6, label="Autoencoder")
        ax.legend()
        st.pyplot(fig, use_container_width=False)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(pred_fail, bins=40)
        st.pyplot(fig2, use_container_width=False)

        fig_hm, ax_hm = plt.subplots(figsize=(10, 4))
        max_samples = st.slider("Number of samples to show (rows)",
                                10, min(200, ae_err_matrix.shape[0]), 50)
        im = ax_hm.imshow(ae_err_matrix[:max_samples], aspect='auto')
        st.pyplot(fig_hm, use_container_width=False)

    # TAB 3
    with tab3:
        st.subheader("📉 Drift Over Time (PC1)")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(X_pca[:, 0], marker='.')
        st.pyplot(fig3, use_container_width=False)

        st.write(f"LSTM AE Drift Score: **{lstm_mean:.6f}**")

    # TAB 4
    with tab4:
        st.subheader("🎯 PCA Advanced Visualization")

        fig4, ax4 = plt.subplots(figsize=(7, 6))
        sc = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=pred_fail)
        st.pyplot(fig4, use_container_width=False)

    # TAB 5
    with tab5:
        st.subheader("📡 Sensor Trend & SPC Chart")
        sensor_names = df.columns.tolist()
        selected_sensor = st.selectbox("Choose a sensor", sensor_names)

        series = df[selected_sensor].astype(float).values
        mean_val = np.nanmean(series)
        std_val = np.nanstd(series)
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val

        fig_sens, ax_sens = plt.subplots(figsize=(12, 4))
        ax_sens.plot(series)
        ax_sens.axhline(mean_val, color='green', linestyle='--')
        ax_sens.axhline(ucl, color='red', linestyle='--')
        ax_sens.axhline(lcl, color='red', linestyle='--')
        st.pyplot(fig_sens, use_container_width=False)

    # TAB 6
    with tab6:
        st.subheader("⭐ Feature Importance")
        fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
        imp_df = pd.DataFrame({
            "sensor": [f"sensor_{i}" for i in range(sensor_importance.shape[0])],
            "importance": sensor_importance
        }).sort_values("importance", ascending=False)
        top_n = st.slider("Show top N sensors", 5, 50, 15)
        ax_imp.bar(imp_df.head(top_n)["sensor"],
                   imp_df.head(top_n)["importance"])
        st.pyplot(fig_imp, use_container_width=False)

else:
    st.info("Upload a CSV file to begin analysis.")
