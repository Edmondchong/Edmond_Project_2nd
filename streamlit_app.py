import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D PCA

# ---- PDF-related imports ----
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

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
    """AE anomaly score (per sample) = MSE over all sensors."""
    X_t = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        recon = model(X_t)
        mse = torch.mean((X_t - recon)**2, dim=1)
    return mse.cpu().numpy()

def compute_tabular_ae_recon_and_error(model, X_scaled: np.ndarray):
    """Return reconstruction and per-sensor squared error."""
    X_t = torch.from_numpy(X_scaled).float().to(device)
    with torch.no.grad():
        recon = model(X_t)
        err = (X_t - recon)**2
    return recon.cpu().numpy(), err.cpu().numpy()

def compute_lstm_ae_score(model, X_scaled: np.ndarray) -> np.ndarray:
    """Single LSTM AE score for the latest window."""
    if X_scaled.shape[0] < window_size:
        return np.array([0.0])
    seq = X_scaled[-window_size:]
    seq = torch.from_numpy(seq).float().unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(seq)
        mse = torch.mean((seq - recon)**2)
    return np.array([float(mse)])

def classify_health(iso_mean, ae_mean, lstm_mean):
    """Simple rule-based health classification."""
    if iso_mean > 0.8 or ae_mean > 0.8 or lstm_mean > 1.2:
        return "🔴 CRITICAL"
    elif iso_mean > 0.5 or ae_mean > 0.5 or lstm_mean > 0.9:
        return "🟡 WARNING"
    else:
        return "🟢 NORMAL"

def compute_sensor_feature_importance(pca_model, rf_model):
    """
    Approximate sensor-level importance:
    - rf feature_importances_ are over PCA components
    - pca.components_ maps components → original sensors
    We combine them to get importance per sensor.
    """
    pc_importance = rf_model.feature_importances_          # shape: (n_components,)
    components = pca_model.components_                     # shape: (n_components, n_features)
    # Weighted sum of absolute loadings
    sensor_importance = np.abs(components.T @ pc_importance)
    return sensor_importance  # (n_features,)

# ---- PDF helpers ----
def fig_to_img(fig):
    """Convert a Matplotlib figure to an in-memory PNG buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf

def generate_full_pdf(
    filename,
    iso_mean,
    ae_mean,
    lstm_mean,
    fail_mean,
    health,
    iso_score_fig,
    ae_heatmap_fig,
    pca_2d_fig,
    feature_imp_fig,
    spc_sensor,
    spc_mean,
    spc_ucl,
    spc_lcl,
    spc_outliers_list
):
    styles = getSampleStyleSheet()
    story = []

    # ---------------- HEADER -----------------
    story.append(Paragraph("<b>SECOM SENSOR DRIFT & ANOMALY DETECTION REPORT</b>", styles["Title"]))
    story.append(Spacer(1, 0.25 * inch))

    # ---------------- EXEC SUMMARY -----------------
    story.append(Paragraph("<b>1. Executive Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"System Overall Health Status: <b>{health}</b>", styles["Normal"]))
    story.append(Paragraph(f"Fail Probability (mean): <b>{fail_mean*100:.2f}%</b>", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- METRICS TABLE -----------------
    story.append(Paragraph("<b>2. Key Model Metrics</b>", styles["Heading2"]))

    table_data = [
        ["Metric", "Value"],
        ["IsolationForest Score (mean)", f"{iso_mean:.4f}"],
        ["Autoencoder Error (mean)", f"{ae_mean:.4f}"],
        ["LSTM AE Error (mean)", f"{lstm_mean:.4f}"],
        ["Fail Probability", f"{fail_mean*100:.2f}%"],
    ]

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    # ---------------- SPC SECTION -----------------
    story.append(Paragraph("<b>3. SPC — Statistical Process Control</b>", styles["Heading2"]))

    story.append(Paragraph(f"Selected Sensor: <b>{spc_sensor}</b>", styles["Normal"]))
    story.append(Paragraph(f"Mean: {spc_mean:.4f}", styles["Normal"]))
    story.append(Paragraph(f"UCL (+3σ): {spc_ucl:.4f}", styles["Normal"]))
    story.append(Paragraph(f"LCL (-3σ): {spc_lcl:.4f}", styles["Normal"]))
    story.append(Paragraph(f"Out-of-Control Points: <b>{len(spc_outliers_list)}</b>", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    # ---------------- HISTOGRAM IMAGE -----------------
    story.append(Paragraph("<b>4. Anomaly Score Distribution</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    iso_img = fig_to_img(iso_score_fig)
    story.append(RLImage(iso_img, width=5.5 * inch, height=3.2 * inch))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- AE HEATMAP IMAGE -----------------
    story.append(Paragraph("<b>5. Autoencoder Reconstruction Heatmap</b>", styles["Heading2"]))
    ae_img = fig_to_img(ae_heatmap_fig)
    story.append(RLImage(ae_img, width=5.5 * inch, height=3.2 * inch))
    story.append(PageBreak())

    # ---------------- PCA 2D PLOT -----------------
    story.append(Paragraph("<b>6. PCA Visualization (PC1 vs PC2)</b>", styles["Heading2"]))
    pca_img = fig_to_img(pca_2d_fig)
    story.append(RLImage(pca_img, width=5.5 * inch, height=3.2 * inch))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- FEATURE IMPORTANCE -----------------
    story.append(Paragraph("<b>7. Sensor Feature Importance</b>", styles["Heading2"]))
    imp_img = fig_to_img(feature_imp_fig)
    story.append(RLImage(imp_img, width=5.5 * inch, height=3.2 * inch))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- FOOTER -----------------
    story.append(Spacer(1, 0.5 * inch))
    story.append(
        Paragraph(
            "<i>This report was automatically generated using Smart Manufacturing AI Dashboard.</i>",
            styles["Italic"],
        )
    )

    doc = SimpleDocTemplate(filename, pagesize=A4)
    doc.build(story)

# ---------------------------------------------
# UI
# ---------------------------------------------
st.title("🔧 SECOM Sensor Drift & Anomaly Detection Dashboard")
st.caption("Smart Manufacturing AI — IsolationForest · Autoencoder · LSTM-AE · PCA · RandomForest")

uploaded_file = st.file_uploader("Upload a SECOM sensor CSV file", type=["csv"])

if uploaded_file:

    # ----------------- LOAD & PREPROCESS -----------------
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)

    # --- Scores from models ---
    iso_score = -iso.decision_function(X_scaled)
    iso_mean = float(np.mean(iso_score))

    ae_score = compute_tabular_ae_score(ae, X_scaled)
    ae_mean = float(np.mean(ae_score))

    lstm_score = compute_lstm_ae_score(lstm_ae, X_scaled)
    lstm_mean = float(np.mean(lstm_score))

    # Classifier output on PCA features
    X_pca = pca_clf.transform(X_scaled)
    pred_fail = clf.predict_proba(X_pca)[:, 1]
    fail_mean = float(np.mean(pred_fail))

    # Health label
    health = classify_health(iso_mean, ae_mean, lstm_mean)

    # Precompute AE per-sensor error for heatmap (used later)
    _, ae_err_matrix = compute_tabular_ae_recon_and_error(ae, X_scaled)  # shape: (N, F)

    # Sensor-level feature importance (RandomForest + PCA)
    sensor_importance = compute_sensor_feature_importance(pca_clf, clf)

    # --------- Precomputed figures for PDF (non-interactive defaults) ---------
    # 1) Histogram of anomaly scores
    fig_pdf_iso, ax_pdf_iso = plt.subplots(figsize=(10, 4))
    ax_pdf_iso.hist(iso_score, bins=40, alpha=0.6, label="IsolationForest")
    ax_pdf_iso.hist(ae_score, bins=40, alpha=0.6, label="Autoencoder")
    ax_pdf_iso.legend()
    ax_pdf_iso.set_xlabel("Score")
    ax_pdf_iso.set_ylabel("Count")

    # 2) AE heatmap (first up to 50 samples)
    max_samples_pdf = min(50, ae_err_matrix.shape[0])
    subset_err_pdf = ae_err_matrix[:max_samples_pdf, :]
    fig_pdf_hm, ax_pdf_hm = plt.subplots(figsize=(10, 4))
    im_pdf = ax_pdf_hm.imshow(subset_err_pdf, aspect='auto', interpolation='nearest')
    ax_pdf_hm.set_xlabel("Sensor Index")
    ax_pdf_hm.set_ylabel("Sample Index")
    ax_pdf_hm.set_title("AE Per-Sensor Squared Error (Top Rows)")

    # 3) PCA 2D scatter
    X_pca_2d_pdf = X_pca[:, :2]
    fig_pdf_pca2d, ax_pdf_pca2d = plt.subplots(figsize=(7, 6))
    sc_pdf = ax_pdf_pca2d.scatter(
        X_pca_2d_pdf[:, 0],
        X_pca_2d_pdf[:, 1],
        c=pred_fail,
        cmap="coolwarm",
        s=15
    )
    ax_pdf_pca2d.set_xlabel("PC1")
    ax_pdf_pca2d.set_ylabel("PC2")
    ax_pdf_pca2d.set_title("PCA 2D — Fail Probability")

    # 4) Feature importance (top 20)
    imp_df_pdf = pd.DataFrame({
        "sensor": [f"sensor_{i}" for i in range(sensor_importance.shape[0])],
        "importance": sensor_importance
    }).sort_values("importance", ascending=False)
    top_imp_pdf = imp_df_pdf.head(20)
    fig_pdf_imp, ax_pdf_imp = plt.subplots(figsize=(10, 4))
    ax_pdf_imp.bar(top_imp_pdf["sensor"], top_imp_pdf["importance"])
    ax_pdf_imp.set_xticklabels(top_imp_pdf["sensor"], rotation=45, ha="right")
    ax_pdf_imp.set_ylabel("Importance (approx.)")
    ax_pdf_imp.set_title("Top Sensor Importance (via PCA + RF)")

    # 5) SPC stats for report (use first sensor as default)
    spc_sensor = df.columns[0]
    series_pdf = df[spc_sensor].astype(float).values
    spc_mean = np.nanmean(series_pdf)
    spc_std = np.nanstd(series_pdf)
    spc_ucl = spc_mean + 3 * spc_std
    spc_lcl = spc_mean - 3 * spc_std
    spc_outliers_idx = np.where((series_pdf > spc_ucl) | (series_pdf < spc_lcl))[0]

    # ---------------------------------------------
    # TABS
    # ---------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Overview",
            "🧪 Anomaly Scores",
            "📉 Drift Analysis",
            "🎯 PCA Visualization",
            "📡 Sensor Trends / SPC",
            "⭐ Feature Importance",
        ]
    )

    # ------------------- TAB 1: OVERVIEW -------------------
    with tab1:
        st.subheader("📊 Machine Health Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("IsolationForest Score", f"{iso_mean:.4f}")
        col2.metric("AE Reconstruction Error", f"{ae_mean:.4f}")
        col3.metric("LSTM AE Error", f"{lstm_mean:.4f}")
        col4.metric("Fail Probability", f"{fail_mean*100:.2f}%")

        st.markdown("### ⚠️ Overall Machine State")
        st.markdown(f"## {health}")

        st.markdown("### Raw Uploaded Data (first 20 rows)")
        st.dataframe(df.head(20))

        # ---- PDF Download Button ----
        st.markdown("### 📄 Download Full PDF Report")
        if st.button("Generate Full PDF Report"):
            pdf_path = "SECOM_Full_Report.pdf"

            generate_full_pdf(
                filename=pdf_path,
                iso_mean=iso_mean,
                ae_mean=ae_mean,
                lstm_mean=lstm_mean,
                fail_mean=fail_mean,
                health=health,
                iso_score_fig=fig_pdf_iso,
                ae_heatmap_fig=fig_pdf_hm,
                pca_2d_fig=fig_pdf_pca2d,
                feature_imp_fig=fig_pdf_imp,
                spc_sensor=spc_sensor,
                spc_mean=spc_mean,
                spc_ucl=spc_ucl,
                spc_lcl=spc_lcl,
                spc_outliers_list=spc_outliers_idx.tolist()
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=f,
                    file_name="SECOM_Full_Report.pdf",
                    mime="application/pdf"
                )

    # ---------------- TAB 2: ANOMALY SCORES -----------------
    with tab2:
        st.subheader("🧪 Histogram of Anomaly Scores")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(iso_score, bins=40, alpha=0.6, label="IsolationForest")
        ax.hist(ae_score, bins=40, alpha=0.6, label="Autoencoder")
        ax.legend()
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("🔮 Fail Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(pred_fail, bins=40)
        ax2.set_xlabel("Fail Probability")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        st.subheader("🔥 Autoencoder Reconstruction Error Heatmap (Samples x Sensors)")
        max_samples = st.slider("Number of samples to show (rows)", 10, min(200, ae_err_matrix.shape[0]), 50)
        subset_err = ae_err_matrix[:max_samples, :]
        fig_hm, ax_hm = plt.subplots(figsize=(10, 4))
        im = ax_hm.imshow(subset_err, aspect='auto', interpolation='nearest')
        ax_hm.set_xlabel("Sensor Index")
        ax_hm.set_ylabel("Sample Index")
        ax_hm.set_title("AE Per-Sensor Squared Error")
        fig_hm.colorbar(im, ax=ax_hm, label="Error")
        st.pyplot(fig_hm)

    # --------------- TAB 3: DRIFT ANALYSIS ------------------
    with tab3:
        st.subheader("📉 Drift Over Time (PC1)")

        pc1 = X_pca[:, 0]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(pc1, marker='.', linestyle='-')
        ax3.set_xlabel("Sample Index")
        ax3.set_ylabel("PC1")
        ax3.set_title("PC1 Drift Over Time")
        ax3.grid(True)
        st.pyplot(fig3)

        st.subheader("📉 LSTM AE Drift (Single Window Score)")
        st.write(f"LSTM AE Drift Score (latest window, size={window_size}): **{lstm_mean:.6f}**")

    # --------------- TAB 4: PCA VISUALIZATION (ADVANCED) ---
    with tab4:
        st.subheader("🎯 PCA Advanced Visualization")

        # 1) PCA 2D colored by fail probability
        st.markdown("### 📌 PCA 2D Scatter (colored by Fail Probability)")
        X_pca_2d = X_pca[:, :2]
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        sc = ax4.scatter(
            X_pca_2d[:, 0],
            X_pca_2d[:, 1],
            c=pred_fail,
            cmap="coolwarm",
            s=15
        )
        plt.colorbar(sc, ax=ax4, label="Fail Probability")
        ax4.set_xlabel("PC1")
        ax4.set_ylabel("PC2")
        ax4.set_title("PCA 2D — Fail Probability")
        st.pyplot(fig4)

        # 2) PCA 3D colored by AE error
        st.markdown("### 🌐 PCA 3D Scatter (colored by AE Error)")
        X_pca_3d = X_pca[:, :3]
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        sc3 = ax_3d.scatter(
            X_pca_3d[:, 0],
            X_pca_3d[:, 1],
            X_pca_3d[:, 2],
            c=ae_score,
            cmap="inferno",
            s=20
        )
        fig_3d.colorbar(sc3, ax=ax_3d, shrink=0.6, label="AE Error")
        ax_3d.set_xlabel("PC1")
        ax_3d.set_ylabel("PC2")
        ax_3d.set_zlabel("PC3")
        ax_3d.set_title("PCA 3D — AE Error")
        st.pyplot(fig_3d)

        # 3) Interactive PCA components
        st.markdown("### 🎛 Interactive PCA Component Viewer")
        pc_x = st.slider("Choose X-Axis Component", 1, min(10, X_pca.shape[1]), 1)
        pc_y = st.slider("Choose Y-Axis Component", 1, min(10, X_pca.shape[1]), 2)

        fig_inter, ax_inter = plt.subplots(figsize=(7, 6))
        sc_int = ax_inter.scatter(
            X_pca[:, pc_x-1],
            X_pca[:, pc_y-1],
            c=pred_fail,
            cmap="coolwarm",
            s=15
        )
        ax_inter.set_xlabel(f"PC{pc_x}")
        ax_inter.set_ylabel(f"PC{pc_y}")
        ax_inter.set_title(f"PCA Projection PC{pc_x} vs PC{pc_y}")
        st.pyplot(fig_inter)

        # 4) PCA drift trajectory
        st.markdown("### 📉 PCA Drift Trajectory (PC1 vs PC2)")
        fig_traj, ax_traj = plt.subplots(figsize=(7, 6))
        ax_traj.plot(X_pca_2d[:, 0], X_pca_2d[:, 1], '-o', markersize=3)
        ax_traj.set_xlabel("PC1")
        ax_traj.set_ylabel("PC2")
        ax_traj.set_title("PCA Drift Path Over Time")
        st.pyplot(fig_traj)

        # 5) PCA anomaly heatmap (AE error)
        st.markdown("### 🔥 AE Error Heatmap on PCA (PC1 vs PC2)")
        fig_heat, ax_heat = plt.subplots(figsize=(7, 6))
        sc_heat = ax_heat.scatter(
            X_pca_2d[:, 0],
            X_pca_2d[:, 1],
            c=ae_score,
            cmap="inferno",
            s=20
        )
        plt.colorbar(sc_heat, ax=ax_heat, label="AE Error")
        ax_heat.set_xlabel("PC1")
        ax_heat.set_ylabel("PC2")
        ax_heat.set_title("AE Error Heatmap in PCA Space")
        st.pyplot(fig_heat)

    # --------- TAB 5: SENSOR TRENDS + SPC (3-SIGMA) ---------
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
        ax_sens.plot(series, marker='.', linestyle='-', label=selected_sensor)
        ax_sens.axhline(mean_val, color='green', linestyle='--', label="Mean")
        ax_sens.axhline(ucl, color='red', linestyle='--', label="UCL (Mean + 3σ)")
        ax_sens.axhline(lcl, color='red', linestyle='--', label="LCL (Mean - 3σ)")
        ax_sens.set_title(f"{selected_sensor} — SPC Chart")
        ax_sens.set_xlabel("Sample Index")
        ax_sens.set_ylabel("Sensor Value")
        ax_sens.legend()
        ax_sens.grid(True)
        st.pyplot(fig_sens)

        # Highlight out-of-control points
        out_of_control_idx = np.where((series > ucl) | (series < lcl))[0]
        st.write(f"Out-of-control points (beyond 3σ): {len(out_of_control_idx)}")
        if len(out_of_control_idx) > 0:
            st.write("Indices:", out_of_control_idx.tolist()[:50])

    # ------------- TAB 6: FEATURE IMPORTANCE ----------------
    with tab6:
        st.subheader("⭐ Feature Importance (Sensor-Level Approximation)")

        # Build DataFrame of importance
        imp_df = pd.DataFrame({
            "sensor": [f"sensor_{i}" for i in range(sensor_importance.shape[0])],
            "importance": sensor_importance
        }).sort_values("importance", ascending=False)

        top_n = st.slider("Show top N sensors", 5, 50, 15)
        top_imp = imp_df.head(top_n)

        st.write("### Top Important Sensors for PASS/FAIL Classification")
        st.dataframe(top_imp.reset_index(drop=True))

        fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
        ax_imp.bar(top_imp["sensor"], top_imp["importance"])
        ax_imp.set_xticklabels(top_imp["sensor"], rotation=45, ha="right")
        ax_imp.set_ylabel("Importance (approx.)")
        ax_imp.set_title("Top Sensor Importance (via PCA + RF)")
        st.pyplot(fig_imp)

else:
    st.info("Upload a CSV file to begin analysis.")
