# 📈 Hisse Senedi Tahmin Botu

XGBoost modeli ile hisse senedi fiyat hareketini tahmin eden web uygulaması.

## Ne Yapıyor?
Hisse kodu girince (ör: THYAO.IS, AAPL) yarınki fiyatın yükseliş mi düşüş mü olacağını tahmin eder.

## Kullanılan Teknolojiler
- Python, XGBoost, Scikit-learn
- yfinance (veri çekme)
- Streamlit (web arayüzü)

## Nasıl Çalıştırılır?
pip install -r requirements.txt
python train.py
streamlit run app.py

## Özellikler
- RSI, MACD, Hareketli Ortalamalar
- Lag features, Volume analizi
- %80 train / %20 test, shuffle=False (zaman serisi)
