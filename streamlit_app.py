import os
import json
import pickle
import streamlit as st
import numpy as np
import faiss

from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# ▒▒▒▒▒▒ CONFIGURACIÓN: Autenticación con Vertex AI ▒▒▒▒▒▒
@st.cache_resource
def cargar_modelos():
    # Escribir las credenciales del service account en /tmp
    sa_info = {
        "type": "service_account",
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"],
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"]["client_id"],
        "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
        "token_uri": st.secrets["gcp_service_account"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
        "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
    }

    with open("/tmp/credentials.json", "w") as f:
        json.dump(sa_info, f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

    # Inicializar Vertex AI
    aiplatform.init(
        project=st.secrets["GCP_PROJECT"],
        location=st.secrets["GCP_REGION"]
    )

    # Cargar modelos de Vertex AI
    modelo_embed = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    modelo_chat = GenerativeModel("gemini-2.5-pro")
    return modelo_embed, modelo_chat


# ▒▒▒▒▒▒ CARGA DE DATOS FAISS ▒▒▒▒▒▒
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["textos"], data["fuentes"]


# ▒▒▒▒▒▒ EMBEDDING DE LA PREGUNTA ▒▒▒▒▒▒
def embed_text(modelo, texto):
    embedding = modelo.get_embeddings([texto])[0].values
    return np.array(embedding, dtype="float32").reshape(1, -1)


# ▒▒▒▒▒▒ BÚSQUEDA DE FRAGMENTOS ▒▒▒▒▒▒
def buscar_contexto(pregunta, modelo_embed, index, textos, fuentes, k=1):
    vector = embed_text(modelo_embed, pregunta)
    distancias, indices = index.search(vector, k)
    resultados = [(textos[i], fuentes[i]) for i in indices[0]]
    return resultados


# ▒▒▒▒▒▒ INTERFAZ DE USUARIO ▒▒▒▒▒▒
st.set_page_config(page_title="Chatbot Legal para Consultorios Dentales", page_icon="🦷")
st.title("🧠 Chatbot Legal - Consultorios Odontológicos (Perú)")
st.markdown("Haz una pregunta basada en las normativas de salud dental vigentes.")

pregunta = st.text_input("👤 Tu consulta legal:")

if pregunta:
    with st.spinner("Buscando respuesta..."):
        # Carga los modelos y datos
        index, textos, fuentes = cargar_index_y_datos()
        modelo_embed, modelo_chat = cargar_modelos()

        fragmentos = buscar_contexto(pregunta, modelo_embed, index, textos, fuentes)
        contexto, fuente = fragmentos[0]

        # Prompt para Gemini
        prompt = f"""
Eres un asistente legal para consultorios odontológicos en Perú.
Responde la siguiente consulta en base a la normativa entregada.

--- CONTEXTO ---
{contexto}
--- FIN DEL CONTEXTO ---

Pregunta: {pregunta}
Respuesta:
        """

        respuesta = modelo_chat.generate_content(prompt)

        # Mostrar respuesta
        st.success("🤖 Gemini dice:")
        st.write(respuesta.text.strip())
        st.caption(f"📄 Fuente: {fuente}")
