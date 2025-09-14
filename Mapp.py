import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Page config

st.set_page_config(page_title="Water Resource Management - Week 4", layout="wide")
st.title("Water Resource Management — Week 4 (Final) — Deployment App")

st.markdown("""
Upload your dataset (CSV). Expected: a date/time column or a 'Year' column and one or more numeric columns (e.g., water_usage, rainfall).

This app will:
- Preview data
- Clean (parse dates, drop duplicates)
- Show summary and visualizations
- Compute rolling averages and a simple forecast using lag-features + LinearRegression
- Let you download the processed dataset
""")


# File uploader

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# STOP app if no file uploaded
if uploaded_file is None:
    st.info("Please upload a CSV file first.")
    st.stop()  # ensures df is never used when undefined

# READ CSV safely
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# CHECK if df is empty
if df.empty:
    st.error("CSV loaded but it is empty!")
    st.stop()

# Column selection

st.sidebar.header("Column selection")
cols = list(df.columns)

date_col = st.sidebar.selectbox(
    "Select date/time column (or Year)", 
    options=[None] + cols, 
    index=0
)

numeric_cols = st.sidebar.multiselect(
    "Numeric columns to analyze", 
    options=cols, 
    default=[c for c in cols if df[c].dtype.kind in 'fi'][:1] if len(cols) > 0 else []
)


# Parse date if selected

if date_col and date_col in df.columns:
    if date_col.lower() != "year":
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col)
            st.success(f"Parsed {date_col} as datetime.")
        except Exception as e:
            st.warning(f"Could not parse {date_col} as datetime: {e}")
    else:
        df[date_col] = df[date_col].astype(int)
        df = df.sort_values(date_col)
        st.info(f"Using {date_col} as integer (not datetime).")


# Cleaning

st.subheader("Cleaning & Summary")
before = len(df)
df = df.drop_duplicates()
after = len(df)
st.write(f"Rows before dedup: {before}, after dedup: {after}")
st.write("Missing values (per column):")
st.write(df.isna().sum())


# Summary statistics

st.subheader("Summary Statistics")
if numeric_cols:
    st.write(df[numeric_cols].describe().T)
else:
    st.info("No numeric columns selected.")

# Visualizations

st.subheader("Visualizations")
viz_col = None
if numeric_cols:
    viz_col = st.selectbox("Pick a numeric column to visualize", options=numeric_cols)

if date_col and viz_col and date_col in df.columns and viz_col in df.columns:
    fig, ax = plt.subplots()
    ax.plot(df[date_col], df[viz_col])
    ax.set_xlabel(date_col)
    ax.set_ylabel(viz_col)
    ax.set_title(f"Time series: {viz_col}")
    st.pyplot(fig)

    # Rolling window
    window = st.slider("Rolling window (days/rows)", min_value=1, max_value=30, value=7)
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        roll = df.set_index(date_col)[viz_col].rolling(window=window).mean()
    else:
        roll = df[viz_col].rolling(window=window).mean()

    fig2, ax2 = plt.subplots()
    ax2.plot(roll.index, roll.values)
    ax2.set_title(f"Rolling mean ({window}) — {viz_col}")
    st.pyplot(fig2)
else:
    st.info("Select both a date column and a numeric column to see time-series plots.")

# Forecasting

st.subheader("Simple Forecast (Lag-based Linear Regression)")
if date_col and viz_col and date_col in df.columns and viz_col in df.columns:
    horizon = st.number_input("Forecast horizon (number of periods)", min_value=1, max_value=365, value=7)
    lags = st.slider("Number of lag features", min_value=1, max_value=10, value=3)

    # Prepare model DataFrame
    df_model = df[[date_col, viz_col]].dropna().copy().reset_index(drop=True)

    # Create lag features
    for lag in range(1, lags + 1):
        df_model[f'lag_{lag}'] = df_model[viz_col].shift(lag)
    df_model = df_model.dropna().reset_index(drop=True)

    if len(df_model) < 10:
        st.warning("Not enough rows after creating lags for reliable forecast. Need at least 10.")
    else:
        # Train-test split
        split = int(len(df_model) * 0.8)
        train, test = df_model.iloc[:split], df_model.iloc[split:]

        X_train = train[[f"lag_{i}" for i in range(1, lags+1)]].values
        y_train = train[viz_col].values
        X_test = test[[f"lag_{i}" for i in range(1, lags+1)]].values
        y_test = test[viz_col].values

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Evaluation
        st.write("Test MAE:", mean_absolute_error(y_test, preds))
        st.write("Test RMSE:", mean_squared_error(y_test, preds, squared=False))

        # Iterative forecast
        last_values = df_model[[f"lag_{i}" for i in range(1, lags+1)]].iloc[-1].values.tolist()
        forecast = []
        for i in range(horizon):
            x = np.array(last_values[-lags:]).reshape(1, -1)
            yhat = model.predict(x)[0]
            forecast.append(yhat)
            last_values.append(yhat)

        # Forecast index
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            last_date = df[date_col].max()
            freq = pd.infer_freq(df[date_col])
            if freq is None:
                diffs = df[date_col].diff().dropna()
                if len(diffs):
                    median = diffs.median()
                    idx = pd.date_range(start=last_date + median, periods=horizon, freq=median)
                else:
                    idx = pd.RangeIndex(start=len(df), stop=len(df) + horizon)
            else:
                idx = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                                    periods=horizon, freq=freq)
            forecast_index = idx
        else:
            if date_col.lower() == "year":
                last_year = df[date_col].max()
                forecast_index = list(range(last_year + 1, last_year + horizon + 1))
            else:
                forecast_index = list(range(len(df), len(df) + horizon))

        fc_series = pd.Series(data=forecast, index=forecast_index, name=f"{viz_col}_forecast")
        st.line_chart(fc_series)


# Download processed data

st.subheader("Download Processed Data with Lag Features")
if st.button("Download CSV"):
    df_model_download = df_model.copy()
    csv = df_model_download.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name="processed_data.csv", mime='text/csv')
