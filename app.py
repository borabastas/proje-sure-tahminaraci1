# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
import joblib
import pandas as pd

# Sayfa Yapılandırması
st.set_page_config(
    page_tiaye="İnşaat Süresi Tahmin Sistemi",
    page_icon="🏗️",
    layout="centered"
)

# Model ve Metaveri Yolları
MODEL_DIR = "models"
META_PATH = os.path.join(MODEL_DIR, "meta.json")
BASIT_MODEL_PATH = os.path.join(MODEL_DIR, "sure_modeli_basit.pkl")
GELISMIS_MODEL_PATH = os.path.join(MODEL_DIR, "sure_modeli_gelismis.pkl")

@st.cache_resource
def verileri_ve_modelleri_yukle():
    if not os.path.exists(META_PATH):
        return None, None, None
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    basit_model = joblib.load(BASIT_MODEL_PATH)
    gelismis_model = joblib.load(GELISMIS_MODEL_PATH)
    return meta, basit_model, gelismis_model

meta, basit_model, gelismis_model = verileri_ve_modelleri_yukle()

if meta is None:
    st.error("Hata: Model veya metaveri dosyaları bulunamadı! Lütfen önce train_models.py çalıştırın.")
else:
    st.tiaye("🏗️ İnşaat Projesi Süre Tahmin Sistemi")
    st.write("Bu uygulama, makine öğrenmesi modellerini kullanarak inşaat projelerinin tahmini tamamlanma sürelerini hesaplar.")
    
    # Yan Panel
    st.sidebar.header("⚙️ Model Seçimi")
    model_tipi = st.sidebar.radio(
        "Kullanılacak Model Türü:",
        options=["Basit Model", "Gelişmiş Model"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Başarı Metrikleri")
    if model_tipi == "Basit Model":
        st.sidebar.metric("R² Skoru (Test)", f"{meta['basit_model']['r2']:.4f}")
        st.sidebar.metric("Ortalama Muayak Hata (MAE)", f"{meta['basit_model']['mae']:.2f} Ay")
        st.sidebar.write("**Kullanılan Özellikler:**", ", ".join(meta['basit_model']['features']))
    else:
        st.sidebar.metric("R² Skoru (Test)", f"{meta['gelismis_model']['r2']:.4f}")
        st.sidebar.metric("R² Skoru (Eğitim)", f"{meta['gelismis_model']['train_r2']:.4f}")
        st.sidebar.metric("Ortalama Muayak Hata (MAE)", f"{meta['gelismis_model']['mae']:.2f} Ay")
        st.sidebar.write("**Kullanılan Özellikler:**", ", ".join(meta['gelismis_model']['features']))

    # Ana Panel
    st.header("📋 Proje Detaylarını Giriniz")
    col1, col2 = st.columns(2)
    
    with col1:
        alan_m2 = st.number_input(
            "İnşaat Alanı (m²):",
            min_value=50,
            max_value=10000,
            value=1000,
            step=50
        )
        kat_sayisi = st.number_input(
            "Kat Sayısı:",
            min_value=1,
            max_value=30,
            value=5,
            step=1
        )
        
    with col2:
        insaat_yili = st.number_input(
            "İnşaat Başlangıç Yılı:",
            min_value=2010,
            max_value=2030,
            value=2024,
            step=1
        )
        zemin_sinifi = st.selectbox(
            "Zemin Sınıfı:",
            options=meta["zemin_siniflari"]
        )
        
    # Ekstrapolasyon Kontrolü
    ekstrapolasyon_uyarisi = []
    if alan_m2 < meta["alan_m2"]["min"] or alan_m2 > meta["alan_m2"]["max"]:
        ekstrapolasyon_uyarisi.append(f"⚠️ İnşaat Alanı ({alan_m2} m²), eğitim verisi sınırlarının ({meta['alan_m2']['min']} - {meta['alan_m2']['max']} m²) dışındadır.")
    if kat_sayisi < meta["kat_sayisi"]["min"] or kat_sayisi > meta["kat_sayisi"]["max"]:
        ekstrapolasyon_uyarisi.append(f"⚠️ Kat Sayısı ({kat_sayisi}), eğitim verisi sınırlarının ({meta['kat_sayisi']['min']} - {meta['kat_sayisi']['max']}) dışındadır.")
    if model_tipi == "Gelişmiş Model":
        if insaat_yili < meta["insaat_yili"]["min"] or insaat_yili > meta["insaat_yili"]["max"]:
            ekstrapolasyon_uyarisi.append(f"⚠️ İnşaat Yılı ({insaat_yili}), eğitim verisi sınırlarının ({meta['insaat_yili']['min']} - {meta['insaat_yili']['max']}) dışındadır.")

    st.markdown("---")
    if st.button("🚀 İnşaat Süresini Tahmin Et", use_container_width=True):
        for uyari in ekstrapolasyon_uyarisi:
            st.warning(uyari)
            
        if model_tipi == "Basit Model":
            girdi_df = pd.DataFrame([{
                'alan_m2': alan_m2,
                'kat_sayisi': kat_sayisi
            }])
            tahmin = basit_model.predict(girdi_df)[0]
        else:
            girdi_dict = {
                'alan_m2': alan_m2,
                'kat_sayisi': kat_sayisi,
                'insaat_yili': insaat_yili,
                'zemin_sinifi_B': 1 if zemin_sinifi == 'B' else 0,
                'zemin_sinifi_C': 1 if zemin_sinifi == 'C' else 0,
                'zemin_sinifi_D': 1 if zemin_sinifi == 'D' else 0
            }
            girdi_df = pd.DataFrame([girdi_dict])
            tahmin = gelismis_model.predict(girdi_df)[0]
            
        st.success(f"### ⏱️ Tahmini Tamamlanma Süresi: **{tahmin:.1f} Ay**")
        
        if model_tipi == "Basit Model":
            st.info(
                f"**Basit Model Formülü:**

"
                f"Süre (Ay) = {basit_model.intercept_:.4f} + "
                f"({meta['basit_model']['alan_katsayisi']:.4f} * Alan) + "
                f"({meta['basit_model']['kat_katsayisi']:.4f} * Kat Sayısı)"
            )
