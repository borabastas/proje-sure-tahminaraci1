import streamlit as st
import os
import json
import joblib
import pandas as pd

st.set_page_config(page_title="İnşaat Proje Süresi Tahmin Uygulaması", layout="centered")

# Başlık ve Açıklama
st.title("🏗️ İnşaat Proje Süresi Tahmin Modeli OKAN SEYFİ DELİMAZI 2022232030")
st.write("Projenizin parametrelerini girerek tahmini tamamlanma süresini hesaplayın.")

# Modelleri ve Meta Veriyi Yükle
OUT_DIR = "models"
meta_path = os.path.join(OUT_DIR, "meta.json")
model_basit_path = os.path.join(OUT_DIR, "sure_modeli_basit.pkl")
model_gelismis_path = os.path.join(OUT_DIR, "sure_modeli_gelismis.pkl")

try:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_basit = joblib.load(model_basit_path)
    model_gelismis = joblib.load(model_gelismis_path)
except Exception as e:
    st.error(f"Model dosyaları yüklenirken hata oluştu: {e}")
    st.stop()

# Sekmeler
tab1, tab2 = st.tabs(["📊 Basit Model ile Tahmin", "🚀 Gelişmiş Model ile Tahmin"])

with tab1:
    st.header("Basit Model")
    st.info(f"Bu model şu metriklerle eğitilmiştir: MAE: {meta['basit_model']['mae']:.2f} ay, R²: {meta['basit_model']['r2']:.4f}")
    
    alan_m2 = st.number_input(
        "İnşaat Alanı (m²)",
        min_value=10.0,
        max_value=10000.0,
        value=1000.0,
        key="basit_alan"
    )
    
    kat_sayisi = st.slider(
        "Kat Sayısı",
        min_value=1,
        max_value=50,
        value=5,
        key="basit_kat"
    )
    
    # Ekstrapolasyon Uyarısı
    if alan_m2 < meta["alan_m2"]["min"] or alan_m2 > meta["alan_m2"]["max"]:
        st.warning(f"⚠️ Girdiğiniz alan değeri ({alan_m2} m²), eğitim verisi sınırlarının dışındadır ({meta['alan_m2']['min']} - {meta['alan_m2']['max']} m²). Tahmin hassasiyeti düşebilir.")
        
    if kat_sayisi < meta["kat_sayisi"]["min"] or kat_sayisi > meta["kat_sayisi"]["max"]:
        st.warning(f"⚠️ Girdiğiniz kat sayısı ({kat_sayisi}), eğitim verisi sınırlarının dışındadır ({meta['kat_sayisi']['min']} - {meta['kat_sayisi']['max']}). Tahmin hassasiyeti düşebilir.")

    if st.button("Süre Hesapla (Basit)", key="btn_basit"):
        X_input = pd.DataFrame([[alan_m2, kat_sayisi]], columns=meta["basit_model"]["features"])
        tahmin = model_basit.predict(X_input)[0]
        st.success(f"⏱️ Tahmini Proje Süresi: **{tahmin:.2f} Ay**")

with tab2:
    st.header("Gelişmiş Model")
    st.info(f"Bu model şu metriklerle eğitilmiştir: MAE: {meta['gelismis_model']['mae']:.2f} ay, R²: {meta['gelismis_model']['r2']:.4f}")
    
    alan_m2_g = st.number_input(
        "İnşaat Alanı (m²)",
        min_value=10.0,
        max_value=10000.0,
        value=1000.0,
        key="gel_alan"
    )
    
    kat_sayisi_g = st.slider(
        "Kat Sayısı",
        min_value=1,
        max_value=50,
        value=5,
        key="gel_kat"
    )
    
    insaat_yili = st.number_input(
        "İnşaat Yılı",
        min_value=1900,
        max_value=2030,
        value=2024
    )
    
    zemin_sinifi = st.selectbox(
        "Zemin Sınıfı",
        options=meta["zemin_siniflari"]
    )
    
    # Ekstrapolasyon Uyarıları
    if alan_m2_g < meta["alan_m2"]["min"] or alan_m2_g > meta["alan_m2"]["max"]:
        st.warning(f"⚠️ Girdiğiniz alan değeri ({alan_m2_g} m²), eğitim verisi sınırlarının dışındadır ({meta['alan_m2']['min']} - {meta['alan_m2']['max']} m²).")
        
    if kat_sayisi_g < meta["kat_sayisi"]["min"] or kat_sayisi_g > meta["kat_sayisi"]["max"]:
        st.warning(f"⚠️ Girdiğiniz kat sayısı ({kat_sayisi_g}), eğitim verisi sınırlarının dışındadır ({meta['kat_sayisi']['min']} - {meta['kat_sayisi']['max']}).")
        
    if insaat_yili < meta["insaat_yili"]["min"] or insaat_yili > meta["insaat_yili"]["max"]:
        st.warning(f"⚠️ Girdiğiniz inşaat yılı ({insaat_yili}), eğitim verisi sınırlarının dışındadır ({meta['insaat_yili']['min']} - {meta['insaat_yili']['max']}).")

    if st.button("Süre Hesapla (Gelişmiş)", key="btn_gel"):
        # One-hot encoding hazırlığı
        input_data = {
            "alan_m2": alan_m2_g,
            "kat_sayisi": kat_sayisi_g,
            "insaat_yili": insaat_yili,
            "zemin_sinifi_B": 1 if zemin_sinifi == "B" else 0,
            "zemin_sinifi_C": 1 if zemin_sinifi == "C" else 0,
            "zemin_sinifi_D": 1 if zemin_sinifi == "D" else 0
        }
        X_input = pd.DataFrame([input_data], columns=meta["gelismis_model"]["features"])
        tahmin = model_gelismis.predict(X_input)[0]
        st.success(f"⏱️ Tahmini Proje Süresi: **{tahmin:.2f} Ay**")
