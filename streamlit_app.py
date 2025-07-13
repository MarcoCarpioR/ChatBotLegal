import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configurar Gemini con clave API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Cargar índice FAISS y metadatos
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["textos"], data["fuentes"]

# Cargar modelo de embeddings local
@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Embedding de la pregunta
def embed_text(modelo, texto):
    embedding = modelo.encode([texto])
    return np.array(embedding, dtype="float32").reshape(1, -1)

# Buscar fragmentos relevantes
def buscar_contexto(pregunta, modelo, index, textos, fuentes, k=1):
    vector = embed_text(modelo, pregunta)
    distancias, indices = index.search(vector, k)
    resultados = [(textos[i], fuentes[i]) for i in indices[0]]
    return resultados

# Interfaz de usuario
st.title("🦷 ChatClínica Legal")
st.markdown("Asistente legal para habilitación de consultorios odontológicos en Perú.")

pregunta = st.text_input("🔍 Escribe tu pregunta legal:")

if pregunta:
    with st.spinner("Buscando normativa relevante..."):
        index, textos, fuentes = cargar_index_y_datos()
        modelo_embed = cargar_modelo_embeddings()

        fragmentos = buscar_contexto(pregunta, modelo_embed, index, textos, fuentes, k=1)
        contexto, fuente = fragmentos[0]

        prompt = f"""
Eres un asistente legal especializado en consultorios odontológicos en Perú.
Responde con claridad y precisión legal en base al siguiente fragmento normativo:

--- CONTEXTO ---
{contexto}
--- FIN DEL CONTEXTO ---

Pregunta: {pregunta}
Respuesta:
"""

        modelo_chat = genai.GenerativeModel("gemini-2.5-pro")
        respuesta = modelo_chat.generate_content(prompt).text.strip()

    st.success("✅ Respuesta:")
    st.write(respuesta)
    st.info(f"📁 Fuente: {fuente}")

    with st.expander("📄 Fragmento normativo usado"):
        st.code(contexto)
