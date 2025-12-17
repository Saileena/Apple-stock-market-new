
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Apple Stock Prediction", layout="wide")

st.title("📈 Apple Stock Price Prediction App")

# Load dataset
data = pd.read_csv("Stock Market.csv")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.sort_values("Date")
data.set_index("Date", inplace=True)

# Load trained model
model = pickle.load(open("xgb_model.pkl", "rb"))

# Feature engineering
df = data.copy()
df['ma7'] = df['Close'].rolling(7).mean()
df['ma14'] = df['Close'].rolling(14).mean()
df['ma21'] = df['Close'].rolling(21).mean()
df['volatility'] = df['Close'].rolling(7).std()
df = df.dropna()

# Sidebar date selection
st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

filtered = df.loc[start_date:end_date]

st.subheader("📊 Historical Close Price")
st.line_chart(filtered['Close'])

# Prediction function
def predict_next_30_days(model, df):
    last = df.iloc[-1]
    ma7, ma14, ma21, vol = last[['ma7','ma14','ma21','volatility']]
    close = last['Close']

    preds = []
    for _ in range(30):
        X = np.array([[ma7, ma14, ma21, vol]])
        next_close = model.predict(X)[0]
        preds.append(next_close)

        vol = (next_close - close) / close
        close = next_close
        ma7 = (ma7 * 6 + next_close) / 7
        ma14 = (ma14 * 13 + next_close) / 14
        ma21 = (ma21 * 20 + next_close) / 21

    return preds

future_prices = predict_next_30_days(model, df)
future_dates = pd.date_range(df.index[-1], periods=30)

future_df = pd.DataFrame(
    {"Predicted Close": future_prices},
    index=future_dates
)

st.subheader("🔮 Next 30 Days Prediction")
st.line_chart(future_df)
st.dataframe(future_df)

st.success("Prediction generated successfully 🚀")
