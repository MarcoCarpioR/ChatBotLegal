import streamlit as st
import os
import json
import faiss
import pickle
import numpy as np

# ────────────────────────
# 1. CARGAR CREDENCIALES GCP
# ────────────────────────
with open("gcp_key.json", "w") as f:
    json.dump(json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"]), f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

# ────────────────────────
# 2. IMPORTAR MODELOS DE VERTEX AI
# ────────────────────────
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel

aiplatform.init(
    project=st.secrets["GCP_PROJECT"],
    location=st.secrets["GCP_REGION"]
)

# ────────────────────────
# 3. CARGAR FAISS + METADATOS
# ────────────────────────
@st.cache_resource
def cargar_index_y_datos():
    index = faiss.read_index("index_normas.faiss")
    with open("metadata.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["textos"], data["fuentes"]

# ────────────────────────
# 4. CARGAR MODELOS GEMINI
# ────────────────────────
@st.cache_resource
def cargar_modelos():
    embed_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    chat_model = GenerativeModel("gemini-2.5-pro")
    return embed_model, chat_model

# ────────────────────────
# 5. FUNCIONES DE CONSULTA
# ────────────────────────
def embed_text(modelo, texto):
    vector = modelo.get_embeddings([texto])[0].values
    return np.array(vector, dtype="float32").reshape(1, -1)

def buscar_contexto(pregunta, modelo_embed, index, textos, fuentes, k=1):
    vector = embed_text(modelo_embed, pregunta)
    distancias, indices = index.search(vector, k)
    resultados = [(textos[i], fuentes[i]) for i in indices[0]]
    return resultados

# ────────────────────────
# 6. UI STREAMLIT
# ────────────────────────
st.title("🦷 ChatClínica Legal")
st.markdown("Consulta normativa para habilitación de consultorios odontológicos en Perú.")

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
