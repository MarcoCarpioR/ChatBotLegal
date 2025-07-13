import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configurar clave de API de Gemini desde secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Cargar índice FAISS y metadatos
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)  # Lista de strings
    return index, metadata

# Cargar modelo de embeddings
@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Generar respuesta con Gemini
def generar_respuesta(pregunta, contexto):
    prompt = f"""Eres un asistente legal experto en normativas peruanas para habilitación de consultorios dentales. Usa únicamente el contexto proporcionado. Sé claro, preciso y profesional.\n
    Pregunta: {pregunta}\n
    Contexto:\n{contexto}\n
    Respuesta:"""
    
    modelo = genai.GenerativeModel("gemini-1.5-flash")
    respuesta = modelo.generate_content(prompt)
    return respuesta.text.strip()

# Interfaz de usuario
st.title("🦷 ChatClínica Legal - Asistente para Consultorios Dentales")
st.markdown("Consulta normativa legal sobre la habilitación de consultorios odontológicos en Perú (Arequipa).")

pregunta_usuario = st.text_input("🔍 Escribe tu pregunta legal:")

if pregunta_usuario:
    with st.spinner("Buscando en normas legales..."):
        index, metadata = cargar_index_y_datos()
        modelo = cargar_modelo_embeddings()

        # Obtener embedding de la pregunta
        embedding = modelo.encode([pregunta_usuario]).astype("float32")

        # Buscar los 5 fragmentos más relevantes
        _, indices = index.search(embedding, k=5)
        contexto = "\n".join([metadata[i] for i in indices[0]])

        # Generar respuesta
        respuesta = generar_respuesta(pregunta_usuario, contexto)

    st.success("✅ Respuesta:")
    st.write(respuesta)

    with st.expander("📄 Fragmentos normativos utilizados"):
        st.text(contexto)
