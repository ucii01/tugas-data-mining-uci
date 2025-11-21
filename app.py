import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Aplikasi Cerdas Analisis Sentimen",
    layout="wide",
    initial_sidebar_state="expanded"
)

REQUIRED_FILES = [
    'ensemble_model.pkl', 
    'tfidf_vectorizer.pkl', 
    'label_encoder.pkl',
    'model_stats.pkl'
]

file_check = [os.path.exists(f) for f in REQUIRED_FILES]
if not all(file_check):
    st.error("🚨 **File Model Tidak Ditemukan!**")
    st.markdown("""
    Mohon pastikan file-file berikut telah diunduh dari Google Colab dan diletakkan di folder yang sama dengan `app.py`:
    1. `ensemble_model.pkl`
    2. `tfidf_vectorizer.pkl`
    3. `label_encoder.pkl`
    4. `model_stats.pkl`
    """)
    st.stop()

@st.cache_resource
def load_components():
    try:
        ensemble_model = joblib.load('ensemble_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        le = joblib.load('label_encoder.pkl')
        stats = joblib.load('model_stats.pkl')
        
        label_mapping = {i: label for i, label in enumerate(stats['label_classes'])}
        
        return ensemble_model, tfidf_vectorizer, le, stats, label_mapping
    except Exception as e:
        st.error(f"Gagal memuat komponen model: {e}")
        st.stop()

ensemble_model, tfidf_vectorizer, le, stats, label_mapping = load_components()

def predict_sentiment(text):
    
    input_vector = tfidf_vectorizer.transform([text])
    
    prediction_encoded = ensemble_model.predict(input_vector)[0]
    
    prediction_proba = ensemble_model.predict_proba(input_vector)
    confidence = np.max(prediction_proba) * 100
    
    sentiment_result = label_mapping.get(prediction_encoded, "Unknown")
    
    proba_data = {label_mapping[i]: prediction_proba[0][i] for i in range(len(label_mapping))}

    return sentiment_result, confidence, proba_data


st.title("🇮🇩 Aplikasi Cerdas Sentimen: Dynamic Duo")
st.markdown("""
Aplikasi ini memprediksi sentimen (_Positive_, _Neutral_, atau _Negative_) dari teks Bahasa Indonesia menggunakan Model **Ensemble (VotingClassifier)** yang menggabungkan:
1.  **Multinomial Naive Bayes**
2.  **Random Forest Classifier**

Target Akurasi Super **($>90\%$ )** telah dicapai melalui preprocessing $\text{TF-IDF}$ yang optimal.
""")
st.write("---")


st.sidebar.title("📊 Model Performance (Akurasi Uji)")

ensemble_accuracy = stats['ensemble_accuracy'] * 100
mnb_accuracy = stats['mnb_accuracy'] * 100
rfc_accuracy = stats['rfc_accuracy'] * 100

st.sidebar.markdown(f"**Akurasi Model Ensemble:**")

if ensemble_accuracy >= 90.0:
    st.sidebar.success(f"## {ensemble_accuracy:.2f}% (SUPER! 🏆)")
else:
    st.sidebar.warning(f"## {ensemble_accuracy:.2f}%")

st.sidebar.markdown("""---""")
st.sidebar.markdown(f"**Perbandingan Akurasi Model Dasar:**")
st.sidebar.info(f"**1. Naive Bayes (MNB):** {mnb_accuracy:.2f}%")
st.sidebar.info(f"**2. Random Forest (RFC):** {rfc_accuracy:.2f}%")
st.sidebar.caption("Gabungan (Ensemble) kedua model ini terbukti efektif dalam meningkatkan akurasi klasifikasi sentimen.")


st.header("📝 Form Input Sentimen")
user_input = st.text_area(
    "Masukkan Teks (Tweet) Bahasa Indonesia Anda di sini:",
    "Pengiriman cepat dan produknya berfungsi sangat baik, saya merekomendasikan toko ini!",
    height=150
)

if st.button("Prediksi Sentimen", type="primary"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks untuk diprediksi.")
    else:
        
        sentiment, confidence, proba_data = predict_sentiment(user_input)
        
        
        st.write("---")
        st.header("✅ Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentimen Terdeteksi:")
            if sentiment == 'positive':
                st.balloons()
                st.success(f"**{sentiment.upper()}** (Optimis)")
            elif sentiment == 'negative':
                st.error(f"**{sentiment.upper()}** (Kritik/Keluhan)")
            else:
                st.warning(f"**{sentiment.upper()}** (Netral/Informasi)")
                
        with col2:
            st.subheader("Tingkat Kepercayaan Model:")
            st.metric(label="Persentase", value=f"{confidence:.2f}%")

        st.subheader("Visualisasi Probabilitas Kelas:")
        
        proba_df = pd.DataFrame(
            proba_data.values(), 
            index=proba_data.keys(), 
            columns=['Probabilitas']
        ).sort_values(by='Probabilitas', ascending=False)
        
        st.bar_chart(proba_df)

        with st.expander("Detail Input dan Analisis"):
            st.text_area("Teks yang Diprediksi:", user_input, height=100, disabled=True)
            st.dataframe(proba_df.T, use_container_width=True)
