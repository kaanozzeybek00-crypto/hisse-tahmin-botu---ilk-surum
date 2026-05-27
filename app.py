# app.py
# Çalıştırmak için: streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Hisse Tahmin Botu",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Hisse Senedi Tahmin Botu")
st.caption("XGBoost modeli ile yarınki fiyat hareketi tahmini")

# ─────────────────────────────────────────
# MODELİ YÜKLE (bir kere yüklenir, cache'lenir)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load("model.pkl")
    ozellikler = joblib.load("ozellikler.pkl")
    return model, ozellikler

try:
    model, OZELLIKLER = load_model()
except FileNotFoundError:
    st.error("❌ model.pkl bulunamadı. Önce 'python train.py' çalıştır.")
    st.stop()

# ─────────────────────────────────────────
# VERİ HAZIRLAMA FONKSİYONU
# ─────────────────────────────────────────
def ozellikleri_hesapla(hisse: str) -> pd.DataFrame:
    ticker = yf.download(hisse, start="2020-01-01", progress=False)
    df = ticker.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if len(df) < 60:
        return None, "Yeterli veri yok (en az 60 gün gerekli)"

    df["MA_7"]  = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()

    df["Degisim(%)"] = df["Close"].pct_change()
    df["Varyans_30"] = df["Close"].rolling(30).var()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]           = ema12 - ema26
    df["Signal"]         = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal"]

    delta    = df["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    rs       = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = (100 - (100 / (1 + rs))).round(1)

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)

    df["Lag_Return_1"]    = df["Degisim(%)"].shift(1)
    df["Lag_Return_2"]    = df["Degisim(%)"].shift(2)
    df["Lag_Return_3"]    = df["Degisim(%)"].shift(3)
    df["Weekly_Momentum"] = df["Close"].pct_change(5)

    df["Price_MA7_Diff"] = (df["Close"] - df["MA_7"])  / df["MA_7"]
    df["MA_7_14_Diff"]   = (df["MA_7"]  - df["MA_14"]) / df["MA_14"]

    df["Avg_Volume_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"]  = df["Volume"] / df["Avg_Volume_20"]
    df["Volume_Change"] = df["Volume"].pct_change()

    df = df.round(2)
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df, None

# ─────────────────────────────────────────
# KULLANICI ARAYÜZÜ
# ─────────────────────────────────────────
hisse = st.text_input(
    "Hisse kodu girin",
    placeholder="ör: THYAO.IS, AAPL, TSLA",
    help="BIST hisseleri için .IS uzantısı ekle (THYAO.IS gibi)"
)

if st.button("Tahmin Yap", type="primary") and hisse:

    with st.spinner("Veri çekiliyor ve analiz ediliyor..."):
        df, hata = ozellikleri_hesapla(hisse.strip().upper())

    if hata:
        st.error(f"❌ {hata}")
    else:
        guncel_fiyat = float(df["Close"].iloc[-1])
        son_tarih    = df.index[-1].strftime("%d %B %Y")
        son_veri     = df[OZELLIKLER].tail(1)

        tahmin       = model.predict(son_veri)[0]
        olasilik     = model.predict_proba(son_veri)[0]

        # ── Sonuç kartı ──
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Hisse", hisse.upper())
            st.metric("Son Kapanış", f"{guncel_fiyat:.2f}")

        with col2:
            st.metric("Son Veri Tarihi", son_tarih)
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

        st.divider()

        if tahmin == 1:
            guven = olasilik[1] * 100
            st.success(f"🚀 TAHMİN: YÜKSELİŞ BEKLENİYOR")
            st.metric("Güven Oranı", f"%{guven:.1f}")
        else:
            guven = olasilik[0] * 100
            st.error(f"📉 TAHMİN: DÜŞÜŞ / YATAY BEKLENİYOR")
            st.metric("Güven Oranı", f"%{guven:.1f}")

        st.caption("⚠️ Bu bir yatırım tavsiyesi değildir. Sadece eğitim amaçlıdır.")

        # ── Fiyat grafiği ──
        st.divider()
        st.subheader("Son 90 Günlük Kapanış Fiyatı")
        st.line_chart(df["Close"].tail(90))
