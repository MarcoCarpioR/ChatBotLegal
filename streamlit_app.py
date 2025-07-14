import streamlit as st
import faiss
import pickle
import numpy as np
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel

# Inicializar Vertex AI
aiplatform.init(project="chatclinica", location="us-central1")

# Cargar índice FAISS y metadatos
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["textos"], data["fuentes"]

# Cargar modelos Gemini
@st.cache_resource
def cargar_modelos():
    modelo_embed = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    modelo_chat = GenerativeModel("gemini-2.5-pro")
    return modelo_embed, modelo_chat

# Obtener embedding
def embed_text(modelo, texto):
    embedding = modelo.get_embeddings([texto])[0].values
    return np.array(embedding, dtype="float32").reshape(1, -1)

# Buscar contexto más relevante
def buscar_contexto(pregunta, modelo_embed, index, textos, fuentes, k=1):
    vector = embed_text(modelo_embed, pregunta)
    distancias, indices = index.search(vector, k)
    resultados = [(textos[i], fuentes[i]) for i in indices[0]]
    return resultados

# Interfaz de Streamlit
st.set_page_config(page_title="ChatClínica Legal", page_icon="⚖️")
st.title("🧠 Chatbot Legal para Consultorios Odontológicos")

pregunta = st.text_input("Escribe tu pregunta legal:", "")

if pregunta:
    index, textos, fuentes = cargar_index_y_datos()
    modelo_embed, modelo_chat = cargar_modelos()

    fragmentos = buscar_contexto(pregunta, modelo_embed, index, textos, fuentes)
    contexto, fuente = fragmentos[0]

    prompt = f"""
Eres un asistente legal para consultorios odontológicos en Perú.
Responde la siguiente consulta en base a la normativa entregada.

--- CONTEXTO ---
{contexto}
--- FIN DEL CONTEXTO ---

Pregunta: {pregunta}
Respuesta:
"""
    with st.spinner("Pensando..."):
        respuesta = modelo_chat.generate_content(prompt).text.strip()

    st.markdown("### 🤖 Respuesta:")
    st.write(respuesta)

    st.markdown(f"📁 **Fuente**: {fuente}")
