import streamlit as st
import faiss
import pickle
import numpy as np
import google.generativeai as genai

# Configurar Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Cargar FAISS y metadatos
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["textos"], data["fuentes"]

# Modelo generativo y de embedding (usando Gemini)
@st.cache_resource
def cargar_modelos():
    modelo_embed = genai.EmbeddingModel("models/embedding-001")
    modelo_chat = genai.GenerativeModel("gemini-2.5-pro")
    return modelo_embed, modelo_chat

# Embedding de texto
def embed_text(modelo, texto):
    respuesta = modelo.embed_content(content=texto, task_type="RETRIEVAL_QUERY")
    vector = respuesta["embedding"]
    return np.array(vector, dtype="float32").reshape(1, -1)

# Buscar fragmentos más similares
def buscar_contexto(pregunta, modelo_embed, index, textos, fuentes, k=1):
    vector = embed_text(modelo_embed, pregunta)
    distancias, indices = index.search(vector, k)
    resultados = [(textos[i], fuentes[i]) for i in indices[0]]
    return resultados

# Interfaz de usuario
st.title("🦷 ChatClínica Legal")
st.markdown("Consulta sobre habilitación de consultorios odontológicos en Perú.")

pregunta = st.text_input("🔍 Escribe tu pregunta legal:")

if pregunta:
    with st.spinner("Buscando normativa relevante..."):
        index, textos, fuentes = cargar_index_y_datos()
        modelo_embed, modelo_chat = cargar_modelos()

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

        respuesta = modelo_chat.generate_content(prompt).text.strip()

    st.success("✅ Respuesta:")
    st.write(respuesta)
    st.info(f"📁 Fuente: {fuente}")

    with st.expander("📄 Fragmento normativo usado"):
        st.code(contexto)
