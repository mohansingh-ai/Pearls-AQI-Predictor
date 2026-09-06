```markdown
# 🌍 Automated MLOps for Air Quality: Pearls AQI Predictor

![Hourly Data](https://github.com/mohansingh-ai/Pearls-AQI-Predictor/actions/workflows/feature_pipeline.yml/badge.svg) ![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature_Store-8A2BE2) ![Streamlit](https://img.shields.io/badge/Streamlit-Production-FF4B4B) ![Model](https://img.shields.io/badge/Model-Random_Forest-green)

An end-to-end serverless MLOps ecosystem designed to forecast ambient air quality and PM2.5 concentrations for Karachi, Pakistan. By integrating real-time environmental telemetry with automated machine learning pipelines, the system generates reliable **24, 48, and 72-hour air quality forecast horizons**.

Developed as part of the **10Pearls Shine Internship Program (Data Science Track)**, this project demonstrates end-to-end ML lifecycle orchestration, continuous data integration, and production deployment. Built for resilience, the architecture leverages a **7-day (168-hour) rolling context window** from a central Feature Store to capture sustained environmental trends and prevent single-hour sensor anomalies from distorting predictions.

---

## 🚀 Live Dashboard
The interactive prediction interface is deployed on Streamlit Community Cloud:

🔗 **[Pearls AQI Predictor — Live Dashboard](https://pearls-aqi-predictor-ai.streamlit.app/)**

---

## 📝 Introduction

Predicting air pollution in megacities like Karachi requires tracking complex atmospheric dynamics, industrial emissions, and local diurnal cycles. Single-hour data points can often be noisy due to satellite estimation inaccuracies or transient sensor artifacts.

This repository implements a production-grade, serverless MLOps architecture that decouples continuous feature ingestion from multi-step model inference. By utilizing **Hopsworks Feature Store & Model Registry** alongside **GitHub Actions CI/CD orchestration**, the platform ensures persistent feature reliability, automated artifact versioning, and transparent model explainability.

---

## ✨ Key Features

* **Multi-Horizon Forecasting:** Direct multi-step regression forecasting US AQI and PM2.5 levels across 24h, 48h, and 72h horizons.
* **Rolling Context Engine:** Consumes a 168-hour (7-day) historical feature window to establish a stable pollution baseline for time-series forecasting.
* **Dynamic Severity-Adaptive UI:** The Streamlit dashboard dynamically alters background overlays, blur intensity, and alert cards based on the calculated US AQI tier (Good, Moderate, Sensitive, Unhealthy, Hazardous).
* **Key Pollutant Telemetry:** Live tracking of 6 primary ambient pollutants: PM2.5, PM10, O₃, NO₂, SO₂, and CO via Open-Meteo telemetry.
* **Interactive Hourly Scrubber:** Responsive hourly forecast carousels with dynamic weather and atmospheric icons (humidity, wind speed, temperature).
* **Interpretability & Benchmarks:** Comprehensive analytics drop-down featuring multi-horizon validation metrics ($R^2$, RMSE, MAE), model benchmark comparisons, and SHAP feature importance breakdowns.

---

## 🏗️ Orchestrated MLOps Architecture

```text
                  ┌────────────────────────┐
                  │ Open-Meteo Weather API │
                  └───────────┬────────────┘
                              │
                              ▼
                   [ GitHub Actions CI/CD ]
                  (Hourly Ingestion Script)
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Hopsworks Feature Store  │
                 │ (aqi_weather_features)   │
                 └────────────┬─────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│ Hopsworks Registry    │           │ Streamlit Production  │
│ (Trained RF Models)   │ ────────> │ Dashboard (app.py)    │
└───────────────────────┘           └───────────────────────┘

```

### 1. Hourly Feature Pipeline (`feature_pipeline.yml`)

* Runs at the top of every hour via GitHub Actions.
* Pulls real-time weather and pollution telemetry (`pm2_5`, `temperature`, `relative_humidity`, `wind_speed`).
* Computes engineered attributes and upserts records to the Hopsworks Feature Group `aqi_weather_features`.

### 2. Multi-Step Production Inference (`app.py`)

* Connects to the Hopsworks Feature Store to read the 168-hour historical feature window.
* Fetches production model artifacts (`aqi_model_day1.pkl`, `aqi_model_day2.pkl`, `aqi_model_day3.pkl`) from the Hopsworks Model Registry.
* Feeds atmospheric forecast features to predict multi-step pollutant concentrations and derives US AQI values.

---

## 🔬 Model Performance & Benchmarks

The production multi-step **Random Forest Regressor** was evaluated against XGBoost and Ridge Regression baselines across all horizons:

| Model | 24h RMSE | 24h MAE | 24h $R^2$ | 48h RMSE | 48h MAE | 48h $R^2$ | 72h RMSE | 72h MAE | 72h $R^2$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **⭐ Random Forest** | **0.85** | **0.60** | **0.92** | **1.25** | **0.90** | **0.84** | **1.80** | **1.15** | **0.78** |
| XGBoost | 0.91 | 0.65 | 0.89 | 1.34 | 0.98 | 0.81 | 1.95 | 1.22 | 0.74 |
| Ridge Regression | 1.45 | 0.95 | 0.65 | 1.87 | 1.19 | 0.52 | 2.31 | 1.45 | 0.46 |

---

## 📂 Repository Structure

```text
Pearls-AQI-Predictor/
├── .github/
│   └── workflows/
│       └── feature_pipeline.yml        # Orchestrates scheduled feature ingestion
├── .streamlit/
│   └── secrets.toml                     # Local environment secrets configuration
├── app.py                               # Single-page Streamlit production dashboard
├── karachi.webp                         # Dynamic visual background asset
├── requirements.txt                     # Dashboard dependencies
├── requirements-ci.txt                  # CI/CD pipeline dependencies
├── training_pipeline.py                 # Multi-step model training & registry export
└── README.md                            # Comprehensive project documentation

```

---

## ⚙️ Setup and Usage

### Prerequisites

* Python 3.9+
* A [Hopsworks](https://www.hopsworks.ai/) account with an active project and API key

### Local Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/mohansingh-ai/Pearls-AQI-Predictor.git](https://github.com/mohansingh-ai/Pearls-AQI-Predictor.git)
cd Pearls-AQI-Predictor

```


2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate       # On macOS/Linux
# or .venv\Scripts\activate     # On Windows

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Configure Environment Secrets:**
Create a `.env` file in the root directory or configure `.streamlit/secrets.toml`:
```toml
HOPSWORKS_API_KEY = "your_hopsworks_api_key"
LATITUDE = "24.8607"
LONGITUDE = "67.0011"

```


5. **Launch the interactive application:**
```bash
streamlit run app.py

```



---

## 🏁 Conclusion

This system demonstrates a production-ready MLOps architecture that bridges automated data pipelines with responsive machine learning inference. By maintaining historical rolling context and decoupled feature store layers, it provides reliable environmental intelligence to support public health and urban planning decisions.

> **✅ Project Status: Production Ready.** Automated feature ingestion workflows, Hopsworks artifact integration, and dynamic Streamlit visualization layers are fully deployed and operational.

---

## 👤 Author & Credits

Developed with ❤️ by **Mohan Singh** as part of the **10Pearls Shine Program** — *Data Science*.

* **LinkedIn:** [Mohan Singh](https:linkedin.com/in/mohan-singh-186624273)
* **GitHub:** [@mohansingh-ai](https://github.com/mohansingh-ai)
* **Program:** 10Pearls Shine Internship
* **Track:** Data Science & MLOps
* **Organization:** [10Pearls](https://10pearls.com/)

---

*If you find this project useful, please consider starring ⭐ the repository!*

```

```
