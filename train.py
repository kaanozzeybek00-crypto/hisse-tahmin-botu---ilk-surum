# train.py
# Bu dosyayı bir kere çalıştırıyorsun → model.pkl ve scaler.pkl oluşuyor
# Sonra app.py bu dosyaları yükleyerek tahmin yapıyor

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ─────────────────────────────────────────
# 1. VERİ ÇEK
# ─────────────────────────────────────────
HISSE = "THYAO.IS"   # ← buraya kendi test ettiğin hisseyi yaz

print(f"[1/5] {HISSE} verisi çekiliyor...")
ticker = yf.download(HISSE, start="2020-01-01", end="2026-05-01", progress=False)
df = ticker.copy()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ─────────────────────────────────────────
# 2. TEKNİK İNDİKATÖRLER
# ─────────────────────────────────────────
print("[2/5] Özellikler hesaplanıyor...")

df["MA_7"]  = df["Close"].rolling(window=7).mean()
df["MA_14"] = df["Close"].rolling(window=14).mean()
df["MA_21"] = df["Close"].rolling(window=21).mean()

df["Degisim(%)"] = df["Close"].pct_change()
df["Varyans_30"] = df["Close"].rolling(window=30).var()

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]           = ema12 - ema26
df["Signal"]         = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Histogram"] = df["MACD"] - df["Signal"]

delta    = df["Close"].diff()
gain     = delta.where(delta > 0, 0)
loss     = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs       = avg_gain / avg_loss
df["RSI"] = (100 - (100 / (1 + rs))).round(1)

df["Lag_1"] = df["Close"].shift(1)
df["Lag_2"] = df["Close"].shift(2)
df["Lag_3"] = df["Close"].shift(3)

df["Lag_Return_1"]   = df["Degisim(%)"].shift(1)
df["Lag_Return_2"]   = df["Degisim(%)"].shift(2)
df["Lag_Return_3"]   = df["Degisim(%)"].shift(3)
df["Weekly_Momentum"] = df["Close"].pct_change(5)

df["Price_MA7_Diff"] = (df["Close"] - df["MA_7"])  / df["MA_7"]
df["MA_7_14_Diff"]   = (df["MA_7"]  - df["MA_14"]) / df["MA_14"]

df["Avg_Volume_20"] = df["Volume"].rolling(window=20).mean()
df["Volume_Ratio"]  = df["Volume"] / df["Avg_Volume_20"]
df["Volume_Change"] = df["Volume"].pct_change()

df = df.round(2)
df.dropna(inplace=True)

# ─────────────────────────────────────────
# 3. HEDEF DEĞİŞKEN
# ─────────────────────────────────────────
esik = 0.005
df = df.iloc[:-1]
df["Hedef"] = (df["Close"].shift(-1) > df["Close"] * (1 + esik)).astype(int)
df.dropna(inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ─────────────────────────────────────────
# 4. MODEL EĞİT
# ─────────────────────────────────────────
print("[3/5] Model eğitiliyor...")

OZELLIKLER = [
    "MA_7", "MA_14", "MA_21",
    "Price_MA7_Diff", "MA_7_14_Diff", "Weekly_Momentum",
    "RSI", "MACD", "Signal", "MACD_Histogram",
    "Varyans_30", "Avg_Volume_20", "Volume_Ratio", "Volume_Change",
    "Lag_1", "Lag_2", "Lag_3",
    "Lag_Return_1", "Lag_Return_2", "Lag_Return_3"
]

x = df[OZELLIKLER]
y = df["Hedef"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)

model_xgb = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
model_xgb.fit(x_train, y_train)

acc = accuracy_score(y_test, model_xgb.predict(x_test))
print(f"    XGBoost Doğruluk: %{acc*100:.2f}")

# ─────────────────────────────────────────
# 5. KAYDET
# ─────────────────────────────────────────
print("[4/5] Dosyalar kaydediliyor...")

joblib.dump(model_xgb, "model.pkl")
joblib.dump(OZELLIKLER, "ozellikler.pkl")

print("[5/5] Tamamlandı!")
print("  → model.pkl      ✓")
print("  → ozellikler.pkl ✓")
print(f"\nModel {HISSE} hissesi üzerinde eğitildi.")
print("Artık 'streamlit run app.py' ile uygulamayı başlatabilirsin.")
