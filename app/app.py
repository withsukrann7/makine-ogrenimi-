import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet

st.title("Zaman Serisi Tahmini")
ticker= st.selectbox("Varlık Seç",["BTC-USD","GOOG","USDTRY=X"])
gun = st.slider("Kaç gün ilerisi?",30,720,365)
st.write(ticker,gun)

@st.cache_data
def veri_getir(ticker):
    df = yf.download(ticker, "2015-01-01", str(date.today()))
    df=df[["Close"]]
    df=df.reset_index()
    df.columns=["ds","y"]
    return df
df = veri_getir(ticker)
st.write(df.tail())


@st.cache_resource
def tahmin_uret(ticker, gun):
    df = veri_getir(ticker)
    model = Prophet()
    model.fit(df)
    gelecek = model.make_future_dataframe(gun)
    tahmin = model.predict(gelecek)
    return model, tahmin

model, tahmin = tahmin_uret(ticker, gun)
fig = model.plot(tahmin)
st.pyplot(fig)

fig2 = model.plot_components(tahmin)
st.pyplot(fig2)
