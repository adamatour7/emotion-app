# import streamlit as st
# import joblib
# from PIL import Image
# import pandas as pd
# import numpy as np

# # Charger le modèle et le vectorizer
# model = joblib.load('svm_emotion_model.joblib')
# vectorizer = joblib.load('tfidf_vectorizer.joblib')

# # Configuration de la page
# st.set_page_config(page_title="Analyse des Émotions", page_icon="😊", layout="centered")

# # Ajout d'un fond coloré
# st.markdown("""
#     <style>
#         body {
#             background-color: #f0f2f6;
#         }
#         .stTextInput > div > div > input {
#             font-size: 18px;
#             padding: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Chargement d'une image pour le design
# image = Image.open("1600026195_3c030bfa7d8875e47bfc8c0ce4e2d65e-800x800.png")  # Ajoutez une image dans votre dossier
# st.image(image, use_column_width=True)

# # Titre principal
# st.title("🔍 Analyse des Émotions dans le Texte")
# st.markdown("Entrez une phrase et découvrez l'émotion dominante !")

# # Champ de texte utilisateur
# user_input = st.text_input("✍️ Entrez votre texte ici :", "")

# # Prédiction du modèle
# if st.button("🔍 Analyser"):
#     if user_input:
#         user_input_clean = vectorizer.transform([user_input])
#         prediction = model.predict(user_input_clean)[0]
        
#         # Affichage du résultat avec un design agréable
#         st.success(f"🌟 Émotion détectée : **{prediction.upper()}**")
#     else:
#         st.warning("⛔ Veuillez entrer un texte avant d'analyser.")

# # Pied de page
# st.markdown("""
#     ---
#     🧑‍💻 *Développé avec ❤️ par Adama Toure*
# """)




import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# Charger le modèle et le vectorizer
model = joblib.load('svm_emotion_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Config page
st.set_page_config(page_title="Analyse des Émotions", page_icon="😊", layout="wide")

# Emojis des émotions
emotion_emojis = {
    'joy': '😊',
    'sadness': '😢',
    'fear': '😨',
    'surprise': '😲',
    'neutral': '😐',
    'disgust': '🤢',
    'shame': '😳'
}

# Fonction pour charger une animation Lottie


# CSS personnalisé
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }

        .title {
            font-size: 48px;
            text-align: center;
            font-weight: bold;
            padding: 10px;
            animation: colorChange 6s infinite;
        }

        @keyframes colorChange {
            0%   { color: #1f77b4; }
            25%  { color: #2ca02c; }
            50%  { color: #ff7f0e; }
            75%  { color: #d62728; }
            100% { color: #1f77b4; }
        }

        .emotion {
            font-size: 32px;
            color: #2ca02c;
            text-align: center;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }

        .stButton > button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #105f95;
            transform: scale(1.05);
        }

        .footer {
            font-size: 14px;
            text-align: center;
            color: #888;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)


# Menu
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Aller à :", ["🎯 Analyse d'Émotion", "📜 Historique", "📊 Tableau de bord"])

# Page d’analyse
if page == "🎯 Analyse d'Émotion":
    st.markdown('<div class="title">🔍 Analyse des Émotions dans le Texte</div>', unsafe_allow_html=True)
    st.markdown("## Entrez une phrase et découvrez l’émotion dominante !")
    

    user_input = st.text_input("✍️ Entrez votre texte ici :", "")

    if st.button("🔍 Analyser"):
        if user_input:
            user_input_clean = vectorizer.transform([user_input])
            prediction = model.predict(user_input_clean)[0]

            emoji = emotion_emojis.get(prediction.lower(), "❓")
            st.markdown(f'<div class="emotion">🌟 Émotion détectée : {emoji} <strong>{prediction.upper()}</strong></div>', unsafe_allow_html=True)

            # Graphique
            st.subheader("📊 Visualisation")
            emotions = ['joy', 'sadness', 'fear', 'surprise', 'neutral', 'disgust', 'shame']
            emotion_prob = {e: 0 for e in emotions}
            emotion_prob[prediction.lower()] = 1
            st.bar_chart(pd.DataFrame.from_dict(emotion_prob, orient='index', columns=["Probabilité"]))

            # Enregistrement
            data = {
                "texte": [user_input],
                "emotion_predite": [prediction],
                "date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            df_new = pd.DataFrame(data)
            file_path = "historique_predictions.csv"

            if os.path.exists(file_path):
                df_new.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df_new.to_csv(file_path, mode='w', header=True, index=False)
        else:
            st.warning("⛔ Veuillez entrer un texte avant de lancer l'analyse.")

# Historique
elif page == "📜 Historique":
    st.markdown('<div class="title">📜 Historique des Prédictions</div>', unsafe_allow_html=True)
    file_path = "historique_predictions.csv"

    if os.path.exists(file_path):
        df_history = pd.read_csv(file_path)
        st.dataframe(df_history[::-1], use_container_width=True)
    else:
        st.info("Aucun historique disponible pour le moment.")

# Tableau de bord
elif page == "📊 Tableau de bord":
    st.markdown('<div class="title">📊 Tableau de bord des Prédictions</div>', unsafe_allow_html=True)
    file_path = "historique_predictions.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        st.markdown(f"### Nombre total de prédictions : **{len(df)}**")

        # Distribution des émotions
        emotion_counts = df['emotion_predite'].value_counts()
        emotion_percents = df['emotion_predite'].value_counts(normalize=True).mul(100).round(2)

        # Tableau résumé
        summary_df = pd.DataFrame({
            'Nombre': emotion_counts,
            'Pourcentage (%)': emotion_percents
        })

        st.markdown("### Répartition des émotions")
        st.dataframe(summary_df)

        # Camembert avec matplotlib
        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
        ax.axis('equal')
        st.pyplot(fig)

        # Timeline (par date si possible)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            timeline = df.groupby(df['date'].dt.date).size()
            st.markdown("### Nombre de prédictions par jour")
            st.line_chart(timeline)
        else:
            st.info("La colonne 'date' est absente, la timeline ne peut pas être affichée.")
    else:
        st.info("Aucun historique disponible pour le moment.")

# Pied de page
st.markdown('<div class="footer">🧑‍💻 Développé avec ❤️ par <strong>Adama Toure</strong></div>', unsafe_allow_html=True)
