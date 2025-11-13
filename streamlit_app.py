import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

import plotly.express as px
import plotly.graph_objects as go

# Optional SHAP (Upgrade 4)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="SECOM Sensor Drift & Anomaly Detection Dashboard",
    layout="wide"
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

def compute_sensor_shap_importance(pca_model, shap_importance_pca):
    components = pca_model.components_
    sensor_shap = np.abs(components.T @ shap_importance_pca)
    return sensor_shap


# ---------------------------------------------
# UI START
# ---------------------------------------------
st.title("🔧 SECOM Sensor Drift & Anomaly Detection Dashboard")
st.caption("Smart Manufacturing AI — IsolationForest · Autoencoder · LSTM-AE · PCA · RF · Clustering · SHAP · SPC")

uploaded_file = st.file_uploader("Upload a SECOM sensor CSV file", type=["csv"])

if uploaded_file:

    # ------------------ LOAD & PREPROCESS ------------------
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)

    # --- Scores ---
    iso_score = -iso.decision_function(X_scaled)
    iso_mean = float(np.mean(iso_score))

    ae_score = compute_tabular_ae_score(ae, X_scaled)
    ae_mean = float(np.mean(ae_score))

    lstm_score = compute_lstm_ae_score(lstm_ae, X_scaled)
    lstm_mean = float(np.mean(lstm_score))

    # PCA + classifier
    X_pca = pca_clf.transform(X_scaled)
    pred_fail = clf.predict_proba(X_pca)[:, 1]
    fail_mean = float(np.mean(pred_fail))

    # AE error matrix
    _, ae_err_matrix = compute_tabular_ae_recon_and_error(ae, X_scaled)

    # sensor-level importance
    sensor_importance = compute_sensor_feature_importance(pca_clf, clf)

    # health
    health = classify_health(iso_mean, ae_mean, lstm_mean)

    # ---------------------------------------------
    # TABS
    # ---------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "🧪 Anomaly Scores",
        "📉 Drift Analysis",
        "🎯 PCA Visualization",
        "📡 Sensor Trends / SPC",
        "⭐ Feature Importance",
        "🧩 Root Cause Clusters",
        "🔍 SHAP Explainability",
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("📊 Machine Health Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("IsolationForest Score", f"{iso_mean:.4f}")
        col2.metric("AE Reconstruction Error", f"{ae_mean:.4f}")
        col3.metric("LSTM AE Error", f"{lstm_mean:.4f}")
        col4.metric("Fail Probability", f"{fail_mean*100:.2f}%")

        st.markdown(f"## {health}")
        st.dataframe(df.head(20), use_container_width=True)

    # ----------------------------------------------------------------------
    # ---------------- TAB 2: ANOMALY SCORES -------------------------------
    # ----------------------------------------------------------------------
    with tab2:
        st.subheader("🧪 Histogram of Anomaly Scores")

        fig_iso_ae = go.Figure()
        fig_iso_ae.add_trace(go.Histogram(x=iso_score, nbinsx=40, name="IsolationForest", opacity=0.6))
        fig_iso_ae.add_trace(go.Histogram(x=ae_score, nbinsx=40, name="Autoencoder", opacity=0.6))
        fig_iso_ae.update_layout(barmode="overlay", height=300)
        st.plotly_chart(fig_iso_ae, use_container_width=True, key="iso_ae_chart")

        st.subheader("🔮 Fail Probability Distribution")
        fig_fail = px.histogram(x=pred_fail, nbins=40, height=300)
        st.plotly_chart(fig_fail, use_container_width=True, key="fail_chart")

        st.subheader("🔥 AE Reconstruction Error Heatmap (Samples × Sensors)")
        max_samples = st.slider("Rows to show", 10, min(200, ae_err_matrix.shape[0]), 50)
        fig_hm = px.imshow(ae_err_matrix[:max_samples], aspect="auto", height=300)
        st.plotly_chart(fig_hm, use_container_width=True, key="ae_heatmap")

    # ----------------------------------------------------------------------
    # ---------------- TAB 3: DRIFT ANALYSIS -------------------------------
    # ----------------------------------------------------------------------
    with tab3:
        st.subheader("📉 PC1 Drift Over Time")

        fig_pc1 = px.line(y=X_pca[:, 0], height=300)
        st.plotly_chart(fig_pc1, use_container_width=True, key="pc1_drift")

        st.subheader("📉 LSTM AE Drift Score")
        st.write(f"Latest LSTM AE Score (window={window_size}): **{lstm_mean:.6f}**")

    # ----------------------------------------------------------------------
    # ---------------- TAB 4: PCA VISUALIZATION ----------------------------
    # ----------------------------------------------------------------------
    with tab4:
        st.subheader("🎯 PCA 2D Colored by Fail Probability")

        X_pca_2d = X_pca[:, :2]
        fig_pca2d = px.scatter(
            x=X_pca_2d[:, 0],
            y=X_pca_2d[:, 1],
            color=pred_fail,
            color_continuous_scale="RdBu",
            height=350
        )
        st.plotly_chart(fig_pca2d, use_container_width=True, key="pca2d")

        st.subheader("🌐 PCA 3D Scatter (AE Error)")
        X_pca_3d = X_pca[:, :3]
        fig_pca3d = px.scatter_3d(
            x=X_pca_3d[:, 0],
            y=X_pca_3d[:, 1],
            z=X_pca_3d[:, 2],
            color=ae_score,
            color_continuous_scale="Inferno",
            height=400
        )
        st.plotly_chart(fig_pca3d, use_container_width=True, key="pca3d")

        st.subheader("🎛 Interactive PCA Viewer")
        pc_x = st.slider("X Component", 1, min(10, X_pca.shape[1]), 1)
        pc_y = st.slider("Y Component", 1, min(10, X_pca.shape[1]), 2)

        fig_pca_int = px.scatter(
            x=X_pca[:, pc_x-1],
            y=X_pca[:, pc_y-1],
            color=pred_fail,
            color_continuous_scale="RdBu",
            height=350
        )
        st.plotly_chart(fig_pca_int, use_container_width=True, key="pca_interactive")

        st.subheader("📉 PCA Drift Trajectory (PC1 vs PC2)")
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=X_pca_2d[:, 0],
            y=X_pca_2d[:, 1],
            mode='lines+markers',
            marker=dict(size=4)
        ))
        fig_traj.update_layout(height=350)
        st.plotly_chart(fig_traj, use_container_width=True, key="pca_traj")

        st.subheader("🔥 AE Error Heatmap on PCA (PC1 vs PC2)")
        fig_pca_heat = px.scatter(
            x=X_pca_2d[:, 0],
            y=X_pca_2d[:, 1],
            color=ae_score,
            color_continuous_scale="Inferno",
            height=350
        )
        st.plotly_chart(fig_pca_heat, use_container_width=True, key="pca_heat")

    # ----------------------------------------------------------------------
    # ---------------- TAB 5: SPC CHART ----------------------------
    # ----------------------------------------------------------------------
    with tab5:
        st.subheader("📡 Sensor Trends + SPC (3σ Control Chart)")

        sensor_list = df.columns.tolist()
        selected_sensor = st.selectbox("Choose sensor", sensor_list)

        series = df[selected_sensor].astype(float).values
        mean_val = np.nanmean(series)
        std_val  = np.nanstd(series)
        ucl = mean_val + 3*std_val
        lcl = mean_val - 3*std_val

        fig_spc = go.Figure()
        fig_spc.add_trace(go.Scatter(y=series, mode="lines+markers", name=selected_sensor))
        fig_spc.add_hline(y=mean_val, line_dash="dash", line_color="green")
        fig_spc.add_hline(y=ucl, line_dash="dash", line_color="red")
        fig_spc.add_hline(y=lcl, line_dash="dash", line_color="red")
        fig_spc.update_layout(height=300)

        st.plotly_chart(fig_spc, use_container_width=True, key="spc_chart")

        out_idx = np.where((series > ucl) | (series < lcl))[0]
        st.write(f"Out-of-control count: {len(out_idx)}")
        if len(out_idx) > 0:
            st.write(out_idx[:50])

    # ----------------------------------------------------------------------
    # ---------------- TAB 6: FEATURE IMPORTANCE ---------------------------
    # ----------------------------------------------------------------------
    with tab6:
        st.subheader("⭐ Sensor Feature Importance (RF + PCA)")

        imp_df = pd.DataFrame({
            "sensor": [f"sensor_{i}" for i in range(sensor_importance.shape[0])],
            "importance": sensor_importance
        }).sort_values("importance", ascending=False)

        top_n = st.slider("Top N Sensors", 5, 50, 15)
        top_imp = imp_df.head(top_n)

        st.dataframe(top_imp.reset_index(drop=True), use_container_width=True)

        fig_imp = px.bar(top_imp, x="sensor", y="importance", height=300)
        fig_imp.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_imp, use_container_width=True, key="feature_imp")

    # ----------------------------------------------------------------------
    # ---------------- TAB 7: ROOT CAUSE CLUSTERS --------------------------
    # ----------------------------------------------------------------------
    with tab7:
        st.subheader("🧩 Root Cause Clusters (KMeans)")

        from sklearn.cluster import KMeans

        n_clusters = st.slider("Number of clusters", 2, 8, 3)
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        cluster_labels = kmeans.fit_predict(X_pca)

        fig_cluster = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=cluster_labels.astype(str),
            height=350
        )
        st.plotly_chart(fig_cluster, use_container_width=True, key="cluster_scatter")

        st.write("### Cluster Summary (AE Error + Fail Prob)")
        cluster_summary = []
        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_summary.append({
                "cluster": c,
                "count": mask.sum(),
                "mean_ae_error": float(ae_score[mask].mean()) if mask.sum() else np.nan,
                "mean_fail_prob": float(pred_fail[mask].mean()) if mask.sum() else np.nan,
            })
        st.dataframe(pd.DataFrame(cluster_summary), use_container_width=True)

        st.write("### 🔍 Sensor Root Cause (AE Error per Cluster)")

        selected_cluster = st.selectbox("Select cluster", list(range(n_clusters)))
        mask = cluster_labels == selected_cluster

        if mask.sum() == 0:
            st.write("This cluster has 0 samples.")
        else:
            cluster_err = ae_err_matrix[mask].mean(axis=0)
            top_k = st.slider("Top K sensors", 5, 30, 10)

            cluster_imp = pd.DataFrame({
                "sensor": [f"sensor_{i}" for i in range(cluster_err.shape[0])],
                "mean_ae_error": cluster_err
            }).sort_values("mean_ae_error", ascending=False).head(top_k)

            st.dataframe(cluster_imp.reset_index(drop=True), use_container_width=True)

            fig_cluster_imp = px.bar(
                cluster_imp,
                x="sensor",
                y="mean_ae_error",
                height=300
            )
            fig_cluster_imp.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cluster_imp, use_container_width=True, key="cluster_sensor_imp")

    # ----------------------------------------------------------------------
    # ---------------- TAB 8: SHAP EXPLAINABILITY --------------------------
    # ----------------------------------------------------------------------
    with tab8:
        st.subheader("🔍 SHAP Explainability")

        if not SHAP_AVAILABLE:
            st.warning("SHAP not installed. Add `shap` to requirements.txt.")
        else:
            st.write("Explaining RandomForest classifier on PCA features (Fail probability).")

            max_bg = min(300, X_pca.shape[0])
            bg_idx = np.random.choice(X_pca.shape[0], max_bg, replace=False)
            X_bg = X_pca[bg_idx]

            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_bg)[1]  # class 1 = fail
            shap_importance_pca = np.mean(np.abs(shap_values), axis=0)

            # Sensor-level SHAP
            sensor_shap = compute_sensor_shap_importance(pca_clf, shap_importance_pca)

            shap_df = pd.DataFrame({
                "sensor": [f"sensor_{i}" for i in range(len(sensor_shap))],
                "shap_importance": sensor_shap
            }).sort_values("shap_importance", ascending=False)

            top_n_shap = st.slider("Top N sensors (SHAP)", 5, 50, 15)
            top_shap_df = shap_df.head(top_n_shap)

            st.dataframe(top_shap_df.reset_index(drop=True), use_container_width=True)

            fig_shap = px.bar(
                top_shap_df,
                x="sensor",
                y="shap_importance",
                height=300
            )
            fig_shap.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_shap, use_container_width=True, key="shap_bar")

            # Sample-level
            st.write("### 🔬 Single-sample SHAP")
            sample_idx = st.slider("Select sample index", 0, X_pca.shape[0]-1, 0)

            shap_sample = explainer.shap_values(X_pca[sample_idx:sample_idx+1])[1][0]
            pca_comp_df = pd.DataFrame({
                "PC": [f"PC{i+1}" for i in range(len(shap_sample))],
                "abs_shap": np.abs(shap_sample),
                "shap": shap_sample
            }).sort_values("abs_shap", ascending=False).head(10)

            st.dataframe(pca_comp_df, use_container_width=True)

else:
    st.info("Upload a CSV file to begin analysis.")
