import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Dairy Intelligence",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(160deg,#eef4ec 0%,#f5f0e8 50%,#e8f2ec 100%);
}

/* Sidebar */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg,#122b1c 0%,#0c1f14 55%,#071509 100%);
    border-right: 1px solid #2a5c40;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: #d7f3e4 !important;
}

/* Hide Tooltip */
div[role="tooltip"] {
    display: none !important;
}

/* Hide Toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide Fullscreen */
button[kind="header"] {
    display: none !important;
}

/* Hide Footer */
footer {
    visibility: hidden !important;
}

/* Hide Menu */
#MainMenu {
    visibility: hidden !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: #1a3d27 !important;
    border: 1px solid #3a7a55 !important;
    border-radius: 8px !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: white;
    border-radius: 15px;
    padding: 1rem;
    border: 1px solid #dbe8df;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

/* Titles */
h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    color: #122b1c !important;
}

/* Download Button */
[data-testid="stDownloadButton"] button {
    background: #2d6a4f !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: #52b788;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════

@st.cache_data
def load_data():
    return pd.read_csv("SmartDairy_Final_Dataset_v2.csv")

df = load_data()

df['Date'] = pd.to_datetime(df['Date'])

# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════

st.sidebar.markdown("""
<div style="text-align:center;padding-top:15px;">
    <div style="font-size:50px;">🥛</div>
    <h2 style="color:#7dd4a8;">Smart Dairy</h2>
    <p style="font-size:12px;color:#8ec8a8;">
        Intelligence System
    </p>
</div>
""", unsafe_allow_html=True)

selected_product = st.sidebar.selectbox(
    "Select Product",
    df['Product_Name'].unique()
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    7,
    30,
    30
)

show_data = st.sidebar.checkbox(
    "Show Raw Data"
)

# ══════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════

st.title("🥛 Smart Dairy Demand Intelligence")

st.markdown("""
<p style='color:#5a7a66;font-size:17px;'>
AI-Based Dairy Product Demand Forecasting Dashboard
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# FILTER DATA
# ══════════════════════════════════════════════════

filtered_df = df[
    df['Product_Name'] == selected_product
].copy()

filtered_df = filtered_df.sort_values('Date')

if show_data:
    st.subheader("📋 Raw Dataset")
    st.dataframe(
        filtered_df,
        use_container_width=True
    )

# ══════════════════════════════════════════════════
# KPI METRICS
# ══════════════════════════════════════════════════

st.subheader("📊 Dashboard Metrics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Sales",
    int(filtered_df['Quantity_Sold'].sum())
)

c2.metric(
    "Average Sales",
    round(filtered_df['Quantity_Sold'].mean(), 2)
)

c3.metric(
    "Maximum Sales",
    int(filtered_df['Quantity_Sold'].max())
)

# ══════════════════════════════════════════════════
# EDA SECTION
# ══════════════════════════════════════════════════

st.header("🔍 Exploratory Data Analysis")

# SALES TREND

st.subheader("📈 Sales Trend")

fig1, ax1 = plt.subplots(figsize=(12,4))

ax1.plot(
    filtered_df['Date'],
    filtered_df['Quantity_Sold'],
    color='#2d6a4f',
    linewidth=2
)

ax1.fill_between(
    filtered_df['Date'],
    filtered_df['Quantity_Sold'],
    alpha=0.2,
    color='#52b788'
)

plt.xticks(rotation=30)

st.pyplot(fig1)

# PRODUCT SALES

st.subheader("🥛 Product Wise Sales")

product_sales = df.groupby(
    'Product_Name'
)['Quantity_Sold'].sum().reset_index()

fig2, ax2 = plt.subplots(figsize=(12,4))

sns.barplot(
    data=product_sales,
    x='Product_Name',
    y='Quantity_Sold',
    palette='Greens',
    ax=ax2
)

plt.xticks(rotation=30)

st.pyplot(fig2)

# MONTHLY SEASONALITY

st.subheader("📅 Monthly Seasonality")

filtered_df['Month'] = filtered_df['Date'].dt.month

monthly = filtered_df.groupby(
    'Month'
)['Quantity_Sold'].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(10,4))

ax3.plot(
    monthly['Month'],
    monthly['Quantity_Sold'],
    marker='o',
    color='#2d6a4f'
)

st.pyplot(fig3)

# TEMPERATURE VS DEMAND

if 'Temperature' in filtered_df.columns:

    st.subheader("🌡 Temperature vs Demand")

    fig4, ax4 = plt.subplots(figsize=(8,4))

    ax4.scatter(
        filtered_df['Temperature'],
        filtered_df['Quantity_Sold'],
        color='#52b788'
    )

    st.pyplot(fig4)

# FESTIVAL IMPACT

if 'Festival_Indicator' in filtered_df.columns:

    st.subheader("🎉 Festival Impact on Sales")

    fig5, ax5 = plt.subplots(figsize=(7,4))

    sns.barplot(
        data=filtered_df,
        x='Festival_Indicator',
        y='Quantity_Sold',
        palette=['#d4a843','#2d6a4f'],
        ax=ax5
    )

    st.pyplot(fig5)

# ══════════════════════════════════════════════════
# FORECASTING SECTION
# ══════════════════════════════════════════════════

st.header("🤖 Demand Forecasting")

forecast_data = filtered_df[
    ['Date','Quantity_Sold']
].copy()

forecast_data.columns = ['ds','y']

split = int(len(forecast_data) * 0.8)

train = forecast_data.iloc[:split]

test = forecast_data.iloc[split:]

# ══════════════════════════════════════════════════
# ARIMA
# ══════════════════════════════════════════════════

try:

    arima_model = ARIMA(
        train['y'],
        order=(5,1,0)
    ).fit()

    arima_pred = arima_model.forecast(
        steps=len(test)
    )

    arima_mae = mean_absolute_error(
        test['y'],
        arima_pred
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            test['y'],
            arima_pred
        )
    )

except:

    arima_mae = 0
    arima_rmse = 0

# ══════════════════════════════════════════════════
# SARIMA
# ══════════════════════════════════════════════════

try:

    sarima_model = SARIMAX(
        train['y'],
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    ).fit(disp=False)

    sarima_pred = sarima_model.forecast(
        steps=len(test)
    )

    sarima_mae = mean_absolute_error(
        test['y'],
        sarima_pred
    )

    sarima_rmse = np.sqrt(
        mean_squared_error(
            test['y'],
            sarima_pred
        )
    )

except:

    sarima_mae = 0
    sarima_rmse = 0

# ══════════════════════════════════════════════════
# PROPHET
# ══════════════════════════════════════════════════

prophet_model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True
)

prophet_model.fit(train)

future = prophet_model.make_future_dataframe(
    periods=len(test)
)

forecast = prophet_model.predict(future)

forecast_test = forecast[
    ['ds','yhat']
].tail(len(test))

prophet_mae = mean_absolute_error(
    test['y'],
    forecast_test['yhat']
)

prophet_rmse = np.sqrt(
    mean_squared_error(
        test['y'],
        forecast_test['yhat']
    )
)

# ══════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════

st.subheader("📋 Model Performance")

performance = pd.DataFrame({

    'Model': ['ARIMA','SARIMA','Prophet'],

    'MAE': [
        round(arima_mae,2),
        round(sarima_mae,2),
        round(prophet_mae,2)
    ],

    'RMSE': [
        round(arima_rmse,2),
        round(sarima_rmse,2),
        round(prophet_rmse,2)
    ]
})

st.dataframe(
    performance,
    use_container_width=True
)

# ══════════════════════════════════════════════════
# FINAL FORECAST
# ══════════════════════════════════════════════════

final_model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True
)

final_model.fit(forecast_data)

# TODAY
today = pd.Timestamp.today().normalize()

# TOMORROW
tomorrow = today + pd.Timedelta(days=1)

# NEXT N DAYS
future_dates = pd.date_range(
    start=tomorrow,
    periods=forecast_days,
    freq='D'
)

future_df = pd.DataFrame({
    'ds': future_dates
})

# PREDICTION
final_forecast = final_model.predict(
    future_df
)

forecast_df = pd.DataFrame({

    'Date': future_dates,

    'Demand': final_forecast['yhat']
    .clip(lower=0)
    .round(2)
})

# ══════════════════════════════════════════════════
# FORECAST CHART
# ══════════════════════════════════════════════════

st.subheader("📉 Forecast Chart")

fig6, ax6 = plt.subplots(figsize=(14,5))

# ACTUAL

ax6.plot(
    filtered_df['Date'],
    filtered_df['Quantity_Sold'],
    label='Actual',
    color='#122b1c',
    linewidth=2
)

# FORECAST

ax6.plot(
    forecast_df['Date'],
    forecast_df['Demand'],
    label='Forecast',
    color='#d4a843',
    linestyle='--',
    linewidth=2
)

# CONFIDENCE BAND

ax6.fill_between(
    forecast_df['Date'],
    forecast_df['Demand'] * 0.9,
    forecast_df['Demand'] * 1.1,
    alpha=0.2,
    color='#d4a843'
)

ax6.legend()

plt.xticks(rotation=30)

st.pyplot(fig6)

# ══════════════════════════════════════════════════
# INTERACTIVE CHART
# ══════════════════════════════════════════════════

st.subheader("🔎 Interactive Forecast")

actual_df = filtered_df[
    ['Date','Quantity_Sold']
].copy()

actual_df.columns = ['Date','Demand']

actual_df['Type'] = 'Actual'

forecast_plot = forecast_df.copy()

forecast_plot['Type'] = 'Forecast'

combined = pd.concat([
    actual_df,
    forecast_plot
])

fig_px = px.line(
    combined,
    x='Date',
    y='Demand',
    color='Type',
    template='plotly_white',
    color_discrete_map={
        'Actual':'#122b1c',
        'Forecast':'#d4a843'
    }
)

fig_px.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#f6f2ea',
    font_family="DM Sans"
)

st.plotly_chart(
    fig_px,
    use_container_width=True
)

# ══════════════════════════════════════════════════
# FORECAST TABLE
# ══════════════════════════════════════════════════

st.subheader(
    f"🗓 Next {forecast_days} Days Prediction"
)

st.dataframe(
    forecast_df,
    use_container_width=True
)

# ══════════════════════════════════════════════════
# DOWNLOAD BUTTON
# ══════════════════════════════════════════════════

st.download_button(
    "⬇ Download Forecast CSV",
    forecast_df.to_csv(index=False).encode('utf-8'),
    "forecast.csv",
    "text/csv"
)

# ══════════════════════════════════════════════════
# ALERT SYSTEM
# ══════════════════════════════════════════════════

if forecast_df['Demand'].max() > 500:

    st.warning(
        "⚠️ High Demand Expected — Increase Inventory"
    )

else:

    st.success(
        "✅ Demand Stable"
    )

# ══════════════════════════════════════════════════
# BUSINESS INSIGHTS
# ══════════════════════════════════════════════════

st.header("💡 Business Insights")

highest = forecast_df.loc[
    forecast_df['Demand'].idxmax()
]

lowest = forecast_df.loc[
    forecast_df['Demand'].idxmin()
]

a, b = st.columns(2)

a.metric(
    "🔺 Peak Demand",
    round(highest['Demand'],2)
)

a.write(
    f"📅 {highest['Date'].strftime('%d-%m-%Y')}"
)

b.metric(
    "🔻 Lowest Demand",
    round(lowest['Demand'],2)
)

b.write(
    f"📅 {lowest['Date'].strftime('%d-%m-%Y')}"
)

# ══════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════

st.markdown("---")

st.markdown("""
<p style='text-align:center;color:#5a7a66;font-size:13px;'>
🥛 Smart Dairy Demand Intelligence · Powered by Streamlit & Prophet
</p>
""", unsafe_allow_html=True)