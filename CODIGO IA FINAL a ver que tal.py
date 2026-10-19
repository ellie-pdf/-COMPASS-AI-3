import pandas as pd

import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

# Leer datos

df: pd.DataFrame = pd.read_excel("base IA.xlsx")


# Variables

X_text = df["query_text"]

y = df["categoria"]

# Vectorizar texto

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X_text)

# Entrenar modelo

modelo = LogisticRegression(max_iter=1000)

modelo.fit(X, y)

# Interfaz



st.title("🧭 Compass AI")

st.subheader("Guiding query investigations faster")

texto = st.text_area("Paste query")

if st.button("Analyze"):

    X_new = vectorizer.transform([texto])
    categoria = modelo.predict(X_new)[0]
    st.write("Category:")
    st.write(categoria)

