# 🥛 Smart Dairy Demand Intelligence System

An AI-powered, interactive demand forecasting dashboard designed for the dairy industry. This system leverages advanced time-series forecasting models to predict dairy product demand, helping businesses optimize inventory management, streamline supply chains, and minimize product spoilage.

## 🚀 Live Dashboard Feature
The application is built using **Streamlit** with a custom, polished green-themed UI to provide an intuitive user experience for inventory and supply chain managers.

---

## ✨ Key Features

- **Interactive UI/UX:** Filter data by specific dairy products and choose a forecasting horizon dynamically (7 to 30 days) using interactive sidebar widgets.
- **Exploratory Data Analysis (EDA):** Visualizes overall sales trends, product-wise distribution, monthly seasonality patterns, and external business drivers (e.g., Temperature and Festival impacts).
- **Multi-Model Evaluation:** Automatically trains and compares three robust statistical and machine learning models:
  - **ARIMA** (Autoregressive Integrated Moving Average)
  - **SARIMA** (Seasonal ARIMA)
  - **Meta's Prophet** (optimized for daily and yearly seasonality)
- **Automated Performance Matrix:** Evaluates models side-by-side using real-time **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error) metrics.
- **Smart Business Alerting:** Includes an automated threshold flag that alerts users (`⚠️ High Demand Expected — Increase Inventory`) if predicted demand crosses critical limits (>500 units).
- **Data Export:** Built-in feature to download the generated forecast table as a `.csv` file with a single click.

---

## 🛠️ Tech Stack & Libraries Used

- **Framework:** Streamlit (Custom CSS injected)
- **Forecasting Models:** Meta Prophet, Statsmodels (ARIMA, SARIMA)
- **Machine Learning & Metrics:** Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Plotly Express, Seaborn, Matplotlib

---

## 📂 Project Structure

```text
├── app.py                            # Main Streamlit application file
├── SmartDairy_Final_Dataset_v2.csv   # Historical sales and environmental dataset
├── README.md                         # Project documentation
└── Project_report.docm               # Comprehensive project report & documentation
