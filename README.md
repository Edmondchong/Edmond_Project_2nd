
# Edmond Chong's 2nd Project

### 👉 Try Demo: 

🖼️ Use this for testing demo >>> [Files_to_test_demo](./Files_to_test_demo)  

### 🤝 Full project is private to prevent unauthorized copying, but Happy to share upon "Recruiter Request"  
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*** The Streamlit demo may go idle after a period of inactivity — click “Yes, get this app back up” to restart it. Please note that it may take a short while to reload.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📖 Project Overview: **🏭 Equipment Drift & Sensor Anomaly Detection for Semiconductor SMAI**. 

This project focuses on detecting equipment drift and sensor anomalies using high-dimensional SECOM sensor data — a real-world manufacturing dataset often used in semiconductor research.

It replicates common fab-level SMAI applications, including:

sensor drift detection

OOC/OOS monitoring

chamber stability analysis

equipment health prediction

SPC-based reliability monitoring

This system aligns with tasks performed in Micron, TI, GlobalFoundries, and SSMC SMAI engineering teams.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔬 Key Approaches

Multi-Model Drift & Anomaly Detection

Built a 4-model AI system using:

IsolationForest (statistical anomaly detection)

Autoencoder (Tabular AE)

LSTM-AE for time-series drift patterns

PCA Drift Analysis

Random Forest for PASS/FAIL classification

Statistical Process Control (SPC)

Implemented industrial SPC rules:

UCL / LCL (3-sigma) thresholds

Out-of-Control (OOC) signal detection

Out-of-Specification (OOS) deviation alerts

Early identification of equipment instability

Interpretability & Engineering Insights

PCA drift trajectory visualization

AE reconstruction error heatmaps

Sensor-importance rankings via PCA → RF mapping

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
⚙️ Tech Stack

Python

NumPy / Pandas / scikit-learn

PyTorch

Autoencoder (Tabular)

LSTM-AE

PCA

SPC (3-sigma UCL/LCL)

Streamlit

Docker

AWS EC2

HuggingFace Hub (Private Artifacts)

CI/CD (GitHub Actions)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
✨ Features

Built a multi-model equipment drift and anomaly detection pipeline.

Real-time SPC monitoring for OOC/OOS sensor behavior.

PCA-based drift trajectory visualization across time.

AE & LSTM-AE reconstruction heatmaps for sensor-level insights.

Full analytics dashboard: anomaly histograms, PCA maps, sensor-importance ranking.

Secure model loading from private HuggingFace repository.

Deployed as a cloud-based dashboard via Streamlit + Docker + AWS.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Notes for Recruiters

The GitHub repo contains only the demo app code to protect model IP and prevent unauthorized copying.

The full training pipeline (data preprocessing, feature engineering, PCA modeling, AE/LSTM-AE training, Random Forest classifier, experiments, and deployment scripts) is kept private, but I am happy to share it upon recruiter request.

This project demonstrates full-stack SMAI capabilities:
→ sensor data processing → drift modeling → SPC monitoring → anomaly detection → dashboard → deployment.


