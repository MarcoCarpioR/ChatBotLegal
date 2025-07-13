import faiss
import pickle
import numpy as np
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# Inicializar Vertex AI
aiplatform.init(project="chatclinica", location="us-central1")

# Cargar FAISS y metadatos
index = faiss.read_index("index_normas.faiss")
with open("metadata.pkl", "rb") as f:
    data = pickle.load(f)

textos = data["textos"]
fuentes = data["fuentes"]

# Embedding model
from vertexai.preview.language_models import TextEmbeddingModel
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

# Modelo generativo (chat) real
model = GenerativeModel("gemini-2.5-pro")

def embed_text(texto):
    embedding = embedding_model.get_embeddings([texto])[0].values
    return np.array(embedding, dtype="float32")

def buscar_fragmento(pregunta, top_k=1):
    vector = embed_text(pregunta)
    vector = np.expand_dims(vector, axis=0)
    distancias, indices = index.search(vector, top_k)
    resultados = []
    for idx in indices[0]:
        fragmento = textos[idx]
        fuente = fuentes[idx]
        resultados.append((fragmento, fuente))
    return resultados

print("🧠 Chatbot Legal con Gemini 2.5 Pro activo. Escribe tu pregunta o 'salir' para terminar.\n")

while True:
    pregunta = input("👤 Tú: ")
    if pregunta.lower() in ["salir", "exit", "q"]:
        break

    fragmentos = buscar_fragmento(pregunta, top_k=1)
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

    response = model.generate_content(prompt)
    print(f"\n🤖 Gemini:\n{response.text.strip()}\n📁 Fuente: {fuente}\n{'-'*60}")
