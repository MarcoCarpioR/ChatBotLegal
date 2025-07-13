import streamlit as st
import faiss
import numpy as np
import pickle
import google.generativeai as genai

# Configurar API Key desde secretos de Streamlit
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Cargar índice FAISS y metadatos
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Obtener embedding desde Gemini
@st.cache_resource
def cargar_modelo_embeddings():
    modelo = genai.EmbeddingModel("models/embedding-001")
    return modelo

# Generar respuesta con Gemini Pro
def generar_respuesta(pregunta, contexto):
    prompt = f"""Eres un asistente legal experto en normativa peruana para la habilitación de consultorios odontológicos.
Responde con precisión legal usando exclusivamente el contexto provisto.

Pregunta: {pregunta}

Contexto normativo:
{contexto}

Respuesta:"""
    modelo = genai.GenerativeModel("gemini-1.5-flash")
    respuesta = modelo.generate_content(prompt)
    return respuesta.text.strip()

# Interfaz de usuario
st.title("🦷 ChatClínica Legal")
st.markdown("Asistente legal para normativas de habilitación de consultorios odontológicos en Perú.")

pregunta_usuario = st.text_input("🔍 Escribe tu pregunta legal:")

if pregunta_usuario:
    with st.spinner("Buscando información normativa..."):
        index, metadata = cargar_index_y_datos()
        modelo_embed = cargar_modelo_embeddings()

        # Obtener embedding desde Gemini
        embedding_response = modelo_embed.embed_content(
            pregunta_usuario,
            task_type="retrieval_query",
            title="Consulta legal"
        )
        embedding = np.array(embedding_response.embedding, dtype="float32").reshape(1, -1)

        # Búsqueda en FAISS
        distancias, indices = index.search(embedding, k=5)

        # Obtener contexto de los top 5 chunks
        contexto = "\n".join([metadata[i] for i in indices[0]])

        # Generar respuesta con contexto
        respuesta = generar_respuesta(pregunta_usuario, contexto)

    st.success("✅ Respuesta:")
    st.write(respuesta)

    with st.expander("📄 Fragmentos normativos utilizados"):
        st.text(contexto)
