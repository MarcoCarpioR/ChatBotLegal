import streamlit as st
import faiss
import pickle
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import numpy as np

# Configuración de Vertex AI
aiplatform.init(project="chatclinica", location="us-central1")

# Cargar índice y metadatos
index = faiss.read_index("index_normas.faiss")
with open("metadata.pkl", "rb") as f:
    data = pickle.load(f)
textos = data["textos"]
fuentes = data["fuentes"]

# Inicializar modelos
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
modelo_chat = GenerativeModel("gemini-2.5-pro")

# Función de recuperación y respuesta
def responder_chatbot(pregunta):
    pregunta_emb = embedding_model.get_embeddings([pregunta])[0].values
    pregunta_np = np.array(pregunta_emb, dtype=np.float32).reshape(1, -1)
    _, indices = index.search(pregunta_np, k=5)
    contexto = "\n".join([textos[i] for i in indices[0]])
    prompt = f"""Contexto legal:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"""
    respuesta = modelo_chat.generate_content(prompt)
    return respuesta.text

# --- Streamlit UI ---
st.set_page_config(page_title="ChatClínica", page_icon="🦷")
st.title("🦷 Chatbot Legal para Consultorios Dentales - Arequipa")

st.markdown("""
Este asistente responde preguntas sobre habilitación de consultorios odontológicos en base a normas del MINSA, COP y municipios.
""")

pregunta = st.text_input("Ingresa tu pregunta legal aquí", placeholder="Ej. ¿Cuántos metros necesita un consultorio dental?")
if st.button("Consultar"):
    if pregunta.strip() != "":
        with st.spinner("Buscando en las normas..."):
            respuesta = responder_chatbot(pregunta)
        st.success("✅ Respuesta del Chatbot:")
        st.write(respuesta)
    else:
        st.warning("Por favor, ingresa una pregunta válida.")
