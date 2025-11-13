import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

import plotly.express as px
import plotly.graph_objects as go

# SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="SECOM Sensor Drift & Anomaly Detection Dashboard",
    layout="wide"
)


# ---------------------------------------------
# LOAD MODELS
# ---------------------------------------------
imputer = joblib.load("secom_imputer.joblib")
scaler = joblib.load("secom_scaler.joblib")
iso = joblib.load("secom_iso.joblib")

pca_clf = joblib.load("secom_pca_clf.joblib")
clf = joblib.load("secom_rf_clf.joblib")

ae_meta = joblib.load("secom_ae_meta.joblib")
input_dim = ae_meta["input_dim"]


# ---------------------------------------------
# AUTOENCODER CLASSES
# ---------------------------------------------
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

# Load Tabular AE
ae = TabularAE(input_dim=input_dim).to(device)
ae.load_state_dict(torch.load("secom_ae_best.pth", map_location=device))
ae.eval()


# LSTM-AE metadata
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
        z_rep = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder_lstm(z_rep)
        return self.fc_output(dec_out)


# Load LSTM-AE
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
        err = torch.mean((X_t - recon)**2, dim=1)
    return err.cpu().numpy()


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
    pc_imp = rf_model.feature_importances_
    comp = pca_model.components_
    return np.abs(comp.T @ pc_imp)


# SHAP FIXED VERSION ✔️
def compute_sensor_shap_importance(pca_model, shap_importance_pca):
    components = pca_model.components_      # shape (n_components, n_features)
    n_pca = components.shape[0]             # use only first N SHAP values
    shap_trimmed = shap_importance_pca[:n_pca]
    return np.abs(components.T @ shap_trimmed)


# ---------------------------------------------
# UI
# ---------------------------------------------
st.title("🔧 SECOM Sensor Drift & Anomaly Detection Dashboard")
st.caption("Smart Manufacturing AI — IsolationForest • AE • LSTM-AE • PCA • RF • SHAP • SPC • Clustering")

uploaded_file = st.file_uploader("Upload SECOM CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

    # Preprocess
    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)

    # Scores
    iso_score = -iso.decision_function(X_scaled)
    iso_mean = float(np.mean(iso_score))

    ae_score = compute_tabular_ae_score(ae, X_scaled)
    ae_mean = float(np.mean(ae_score))

    lstm_score = compute_lstm_ae_score(lstm_ae, X_scaled)
    lstm_mean = float(np.mean(lstm_score))

    X_pca = pca_clf.transform(X_scaled)
    pred_fail = clf.predict_proba(X_pca)[:, 1]
    fail_mean = float(np.mean(pred_fail))

    # AE error matrix
    _, ae_err_matrix = compute_tabular_ae_recon_and_error(ae, X_scaled)

    # Sensor importance (RF)
    sensor_importance = compute_sensor_feature_importance(pca_clf, clf)

    # Health
    health = classify_health(iso_mean, ae_mean, lstm_mean)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "🧪 Anomaly Scores",
        "📉 Drift",
        "🎯 PCA Visualization",
        "📡 SPC Charts",
        "⭐ Feature Importance",
        "🧩 Clusters",
        "🔍 SHAP Explainability",
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("📊 Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IsoForest", f"{iso_mean:.4f}")
        c2.metric("AE Err", f"{ae_mean:.4f}")
        c3.metric("LSTM Err", f"{lstm_mean:.4f}")
        c4.metric("Fail Prob", f"{fail_mean*100:.2f}%")
        st.markdown(f"## {health}")
        st.dataframe(df.head(20))

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("🧪 IsoForest + AE Score Distribution")

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=iso_score, nbinsx=40, opacity=0.6, name="IsoForest"))
        fig1.add_trace(go.Histogram(x=ae_score, nbinsx=40, opacity=0.6, name="AE Error"))
        fig1.update_layout(barmode="overlay", height=300)
        st.plotly_chart(fig1, use_container_width=True, key="iso_ae")

        st.subheader("Fail Probability Distribution")
        fig2 = px.histogram(pred_fail, nbins=40, height=300)
        st.plotly_chart(fig2, use_container_width=True, key="fail_hist")

        st.subheader("AE Error Heatmap (Sensors × Samples)")
        fig3 = px.imshow(ae_err_matrix[:50], aspect="auto", height=300)
        st.plotly_chart(fig3, use_container_width=True, key="ae_heatmap")

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("📉 PC1 Drift")
        fig_pc = px.line(y=X_pca[:, 0], height=300)
        st.plotly_chart(fig_pc, use_container_width=True, key="pc1")

    # ---------------- TAB 4 ----------------
    with tab4:
        st.subheader("🎯 PCA 2D")
        fig_2d = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                            color=pred_fail, color_continuous_scale="RdBu", height=300)
        st.plotly_chart(fig_2d, use_container_width=True, key="pca2d")

        st.subheader("🌐 PCA 3D")
        fig_3d = px.scatter_3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            color=ae_score, color_continuous_scale="Inferno", height=350
        )
        st.plotly_chart(fig_3d, use_container_width=True, key="pca3d")

        st.subheader("Interactive PCA Viewer")
        pcx = st.slider("PC X", 1, X_pca.shape[1], 1)
        pcy = st.slider("PC Y", 1, X_pca.shape[1], 2)
        fig_int = px.scatter(
            x=X_pca[:, pcx-1], y=X_pca[:, pcy-1],
            color=pred_fail, height=300
        )
        st.plotly_chart(fig_int, use_container_width=True, key="pca_int")

    # ---------------- TAB 5 ----------------
    with tab5:
        st.subheader("📡 Sensor SPC Chart")
        sensor_name = st.selectbox("Select sensor", df.columns)

        series = df[sensor_name].astype(float).values
        mean_val = np.nanmean(series)
        std_val = np.nanstd(series)
        ucl = mean_val + 3*std_val
        lcl = mean_val - 3*std_val

        fig_spc = go.Figure()
        fig_spc.add_trace(go.Scatter(y=series, mode="lines+markers"))
        fig_spc.add_hline(y=mean_val, line_color="green")
        fig_spc.add_hline(y=ucl, line_color="red")
        fig_spc.add_hline(y=lcl, line_color="red")
        fig_spc.update_layout(height=300)
        st.plotly_chart(fig_spc, use_container_width=True, key="spc")

    # ---------------- TAB 6 ----------------
    with tab6:
        st.subheader("⭐ Sensor Importance (RF + PCA)")
        imp_df = pd.DataFrame({
            "sensor": df.columns,
            "importance": sensor_importance
        }).sort_values("importance", ascending=False)

        st.dataframe(imp_df.head(20))

        fig_imp = px.bar(imp_df.head(20), x="sensor", y="importance", height=300)
        st.plotly_chart(fig_imp, use_container_width=True, key="feat_imp")

    # ---------------- TAB 7 ----------------
    with tab7:
        st.subheader("🧩 KMeans Clusters (Root Cause)")
        from sklearn.cluster import KMeans

        k = st.slider("Clusters", 2, 8, 3)
        km = KMeans(n_clusters=k, n_init="auto").fit(X_pca)
        labels = km.labels_

        fig_clus = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=labels.astype(str), height=300
        )
        st.plotly_chart(fig_clus, use_container_width=True, key="clusters")

    # ---------------- TAB 8 ----------------
    with tab8:
        st.subheader("🔍 SHAP Explainability")

        if not SHAP_AVAILABLE:
            st.warning("Install shap in requirements.txt")
        else:
            st.write("Explaining RF model (Fail probability)")

            # SHAP background sample
            bg = X_pca[:200]
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(bg)[1]
            shap_imp_pca = np.mean(np.abs(shap_vals), axis=0)

            # FIX: trim SHAP to PCA components
            components = pca_clf.components_
            n_pca = components.shape[0]
            shap_imp_pca = shap_imp_pca[:n_pca]

            sensor_shap = compute_sensor_shap_importance(pca_clf, shap_imp_pca)

            shap_df = pd.DataFrame({
                "sensor": df.columns,
                "importance": sensor_shap
            }).sort_values("importance", ascending=False)

            st.dataframe(shap_df.head(20))

            fig_shap = px.bar(shap_df.head(20), x="sensor", y="importance", height=300)
            st.plotly_chart(fig_shap, use_container_width=True, key="shap_bar")


else:
    st.info("📥 Upload a SECOM CSV file to begin.")
