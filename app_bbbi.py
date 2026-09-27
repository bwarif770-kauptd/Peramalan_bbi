import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Konfigurasi agar optimal di layar sentuh tablet
st.set_page_config(
    page_title="BBI Fish Forecasting", page_icon="🐟", layout="centered"
)

st.title("🐟 Peramalan Stok & Bibit Ikan BBI")
st.write(
    "Sistem Prediksi Kebutuhan & Penjualan Bibit Ikan Berdasarkan Pola Musim"
)

# Sidebar Pengaturan untuk Tablet
st.sidebar.image(
    "https://img.icons8.com/color/96/fish.png", width=80
)  # Logo ikan ilustrasi
st.sidebar.header("⚙️ Konfigurasi")

# Pilihan Input Data
data_option = st.sidebar.radio(
    "Sumber Data:", ["Gunakan Data Contoh (Dummy)", "Upload File CSV"]
)

if data_option == "Gunakan Data Contoh (Dummy)":
  np.random.seed(100)
  date_range = pd.date_range(start="2023-01-01", end="2025-12-31", freq="M")
  # Simulasi tren naik dengan pola musiman tahunan (s=12)
  trend = np.linspace(2000, 5000, len(date_range))
  seasonality = 800 * np.sin(np.linspace(0, 3 * 2 * np.pi, len(date_range)))
  sales = np.clip(trend + seasonality + np.random.normal(0, 100, len(date_range)), 500, None).astype(int)
  
  df = pd.DataFrame({"Penjualan": sales}, index=date_range)
else:
  uploaded_file = st.sidebar.file_uploader("Upload CSV (.csv)", type=["csv"])
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Tanggal"], index_col="Tanggal")
  else:
    st.info("👆 Silakan pilih atau upload data terlebih dahulu di menu samping.")
    st.stop()

# Tampilkan Grafik Data Historis
st.subheader("📈 Grafik Data Historis")
st.line_chart(df)

# Pengaturan Parameter Musim
st.subheader("🎛️ Pengaturan Parameter Peramalan")
col1, col2 = st.columns(2)

with col1:
  forecast_steps = st.slider("Horizon Prediksi (Bulan)", 1, 12, 6)
with col2:
  seasonal_periods = st.number_input("Periode Musim (Bulan)", value=12)

# Tombol Eksekusi ramalan (Ramah Touchscreen)
if st.button("🚀 Jalankan Peramalan Musim", use_container_width=True):
  try:
    # Model Holt-Winters dengan komponen musiman
    model = ExponentialSmoothing(
        df.iloc[:, 0],
        trend="add",
        seasonal="add",
        seasonal_periods=int(seasonal_periods),
    ).fit()

    forecast = model.forecast(forecast_steps)
    future_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="M",
    )
    
    df_forecast = pd.DataFrame({"Prediksi Permintaan": forecast.values}, index=future_dates)

    st.success("Peramalan Berhasil Dibuat!")
    
    # Tampilkan Hasil Prediksi
    st.subheader("📋 Tabel Hasil Prediksi & Kebutuhan Stok")
    st.dataframe(df_forecast.style.format("{:.0f} ekor"), use_container_width=True)

    # Visualisasi Gabungan
    st.subheader("📊 Proyeksi Grafik ke Depan")
    df_combined = pd.concat([df, df_forecast.rename(columns={"Prediksi Permintaan": df.columns[0]})])
    st.line_chart(df_combined)

    # Catatan BBI
    st.warning(
        "💡 **Catatan Operasional BBI:** Perhatikan kenaikan kurva pada bulan-bulan"
        " tertentu untuk mengantisipasi lonjakan permintaan dari kelompok tani"
        " ikan."
    )

  except Exception as e:
    st.error(f"Gagal memproses data: {e}")