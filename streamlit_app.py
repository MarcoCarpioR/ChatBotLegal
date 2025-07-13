import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# Configura tu clave de API de Gemini desde el entorno
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Cargar el índice FAISS y los textos asociados
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("data/faiss_index.index")
    chunks = pd.read_csv("data/text_chunks.csv")
    return index, chunks

# Cargar modelo de embeddings
@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Función para obtener respuesta del modelo Gemini
def generar_respuesta(pregunta, contexto):
    prompt = f"""Eres un asistente legal experto en normativas peruanas para habilitación de consultorios dentales. Usa únicamente el contexto proporcionado. Sé claro, preciso y profesional.\n
    Pregunta: {pregunta}\n
    Contexto:\n{contexto}\n
    Respuesta:"""

    modelo = genai.GenerativeModel("gemini-1.5-flash")
    respuesta = modelo.generate_content(prompt)
    return respuesta.text.strip()

# Interfaz
st.title("🦷 ChatClínica Legal - Asistente para Consultorios Dentales")
st.markdown("Consulta normativa legal sobre la habilitación de consultorios odontológicos en Perú (Arequipa).")

pregunta_usuario = st.text_input("🔍 Escribe tu pregunta legal:")

if pregunta_usuario:
    with st.spinner("Buscando en normas legales..."):
        index, chunks = cargar_index_y_datos()
        modelo = cargar_modelo_embeddings()

        # Obtener embedding de la pregunta
        embedding_pregunta = modelo.encode([pregunta_usuario])
        _, indices = index.search(np.array(embedding_pregunta).astype("float32"), k=5)

        # Recuperar los textos más similares
        contexto = "\n".join(chunks.iloc[indices[0]]["texto"].tolist())

        # Generar respuesta
        respuesta = generar_respuesta(pregunta_usuario, contexto)

    st.success("✅ Respuesta:")
    st.write(respuesta)

    with st.expander("📄 Fragmentos normativos utilizados"):
        st.text(contexto)
