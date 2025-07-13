import streamlit as st
import faiss
import pickle
import numpy as np
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# Inicializar Vertex AI solo una vez
aiplatform.init(
    project=st.secrets["GCP_PROJECT"],
    location=st.secrets["GCP_REGION"]
)

# Cargar FAISS y metadatos
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["textos"], data["fuentes"]

# Cargar modelos Gemini desde Vertex AI
@st.cache_resource
def cargar_modelos():
    embed_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    chat_model = GenerativeModel("gemini-2.5-pro")
    return embed_model, chat_model

# Obtener embedding de una pregunta
def embed_text(modelo, texto):
    embedding_response = modelo.get_embeddings([texto])
    vector = embedding_response[0].values
    return np.array(vector, dtype="float32").reshape(1, -1)

# Buscar fragmentos más relevantes en el índice FAISS
def buscar_contexto(pregunta, modelo_embed, index, textos, fuentes, k=1):
    vector = embed_text(modelo_embed, pregunta)
    distancias, indices = index.search(vector, k)
    resultados = [(textos[i], fuentes[i]) for i in indices[0]]
    return resultados

# Interfaz Streamlit
st.title("🦷 ChatClínica Legal")
st.markdown("Consulta normativa legal para habilitación de consultorios odontológicos en Perú.")

pregunta = st.text_input("🔍 Escribe tu pregunta legal:")

if pregunta:
    with st.spinner("Buscando normativa relevante..."):
        index, textos, fuentes = cargar_index_y_datos()
        modelo_embed, modelo_chat = cargar_modelos()

        fragmentos = buscar_contexto(pregunta, modelo_embed, index, textos, fuentes, k=1)
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

        respuesta = modelo_chat.generate_content(prompt).text.strip()

    st.success("✅ Respuesta:")
    st.write(respuesta)
    st.info(f"📁 Fuente: {fuente}")

    with st.expander("📄 Fragmento normativo usado"):
        st.code(contexto)
