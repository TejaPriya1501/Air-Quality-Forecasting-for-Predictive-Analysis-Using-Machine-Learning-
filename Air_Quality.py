import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Air Quality Prediction", layout="wide")

# Function to encode local image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""

bg_image = get_base64_image("BG_Image.png")

# Styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 1rem;
    }}
    .status-good {{ color: green; font-weight: bold; }}
    .status-normal {{ color: orange; font-weight: bold; }}
    .status-hilarious {{ color: red; font-weight: bold; }}
    .title-style {{ font-size: 2.5rem; font-weight: 700; color: #003366; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title-style'>😶‍🌫️ AIR QUALITY FORECASTING PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='title-style'>Dashboard</h2>", unsafe_allow_html=True)

# Load and preprocess
data = pd.read_csv("AirQuality_Datasets_UCI.csv")
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
data = data.dropna(thresh=5)

for col in data.columns:
    if col not in ["Date", "Time"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna().reset_index(drop=True)

target_stats = {
    'CO(GT)': (2.20, 0.10, 11.90),
    'PT08.S1(CO)': (1040.00, 500.00, 2080.00),
    'NMHC(GT)': (160.00, 0.00, 1180.00),
    'C6H6(GT)': (12.00, 0.10, 64.20),
    'PT08.S2(NMHC)': (950.00, 200.00, 2700.00),
    'NOx(GT)': (240.00, 2.00, 1470.00),
    'PT08.S3(NOx)': (800.00, 100.00, 2700.00),
    'NO2(GT)': (120.00, 2.00, 350.00),
    'PT08.S4(NO2)': (740.00, 200.00, 1200.00),
    'PT08.S5(O3)': (880.00, 150.00, 2500.00),
}

with st.sidebar:
    target = st.selectbox("Select Target Pollutant", list(target_stats.keys()), index=7)
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LightGBM", "SVR"])
    avg, mn, mx = target_stats[target]
    st.markdown(f"**Average:** {avg}")
    st.markdown(f"**Min:** {mn}")
    st.markdown(f"**Max:** {mx}")

features = data.drop(columns=[c for c in ["Date", "Time", target] if c in data.columns]).columns
X = data[features]
y = data[target]

X_train = X.iloc[:-1]
y_train = y.iloc[:-1]
X_current = X.iloc[[-1]]

# --- Model training & prediction ---
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_choice == "XGBoost":
    model = XGBRegressor(n_estimators=100, random_state=42)
elif model_choice == "LightGBM":
    model = LGBMRegressor(n_estimators=100, random_state=42)
elif model_choice == "SVR":
    model = SVR(kernel='rbf')
else:
    st.error("Unsupported model selected.")
    st.stop()

model.fit(X_train, y_train)
current_prediction = model.predict(X_current)[0]
y_true_recent = y.tail(20)
y_pred_recent = model.predict(X.tail(20))
mse = mean_squared_error(y_true_recent, y_pred_recent)
r2 = r2_score(y_true_recent, y_pred_recent)

# --- Sidebar status ---
with st.sidebar:
    if current_prediction <= avg:
        status = "<span class='status-good'>Good</span>"
    elif current_prediction <= mx:
        status = "<span class='status-normal'>Normal</span>"
    else:
        status = "<span class='status-hilarious'>Hilarious</span>"
    st.markdown(f"### Status: {status}", unsafe_allow_html=True)

    about = st.selectbox("About", ["Info", "Credits"])
    if about == "Info":
        st.markdown("A Predictive Dashboard for Urban Air Quality Using Machine Learning — An Interactive Insight into Pollution Levels with Real-Time Classification.")
    else:
        st.markdown("Data Source: UCI ML Repository\n\nMaintained by: **Teja Priya.P**")

# --- Results ---
st.subheader("🔮 Predicted Current Value")
st.write(f"**Predicted {target}:** {current_prediction:.2f} using **{model_choice}**")

st.subheader("📊 Model Performance on Recent Data")
st.write(f"- Mean Squared Error (last 20): {mse:.2f}")
st.write(f"- R² Score (last 20): {r2:.2f}")

st.subheader("🔍 Recent Predictions Comparison")
comparison_df = pd.DataFrame({
    "Actual": y_true_recent.values,
    "Predicted": y_pred_recent
}).reset_index(drop=True)
st.dataframe(comparison_df)

# --- Plot ---
st.subheader("📈 Actual vs Predicted (Recent 20)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(y_true_recent.values, label="Actual", marker="o")
ax.plot(y_pred_recent, label="Predicted", marker="x")
ax.set_xlabel("Index")
ax.set_ylabel(target)
ax.set_title(f"{model_choice}: Actual vs Predicted for {target}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Download buttons ---
buf = io.BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.download_button("📅 Download Plot as PNG", data=buf, file_name=f"{target}_plot.png", mime="image/png")

# PDF report
def generate_pdf_report(target, model_choice, prediction, mse, r2):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, h - 80, "Air Quality Forecast Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, h - 130, f"Pollutant: {target}")
    c.drawString(100, h - 150, f"Model Used: {model_choice}")
    c.drawString(100, h - 170, f"Predicted Value: {prediction:.2f}")
    c.drawString(100, h - 190, f"MSE (Last 20): {mse:.2f}")
    c.drawString(100, h - 210, f"R² (Last 20): {r2:.2f}")
    c.drawString(100, h - 250, "Report Generated by Air Quality Dashboard")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

st.subheader("📄 Export Report as PDF")
pdf_buf = generate_pdf_report(target, model_choice, current_prediction, mse, r2)
st.download_button("📅 Download PDF Report", data=pdf_buf, file_name=f"{target}_report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("© 2025 | Predictive Dashboard for Urban Air Quality using Machine Learning")