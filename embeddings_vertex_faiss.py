import os
import faiss
import numpy as np
import pickle
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

# Inicializar Vertex AI
aiplatform.init(project="chatclinica", location="us-central1")  # ← Cambia esto

# Cargar modelo de embeddings actualizado
model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

# Rutas de entrada y salida
RUTA_TEXTOS = "textos_extraidos"  # Carpeta donde están tus .txt
ARCHIVO_FAISS = "index_normas.faiss"
ARCHIVO_METADATA = "metadata.pkl"

# Fragmentar texto largo por contexto semántico
def fragmentar_texto(texto, max_longitud=1000):
    oraciones = texto.split(". ")
    fragmentos, actual = [], ""
    for oracion in oraciones:
        if len(actual) + len(oracion) < max_longitud:
            actual += oracion + ". "
        else:
            fragmentos.append(actual.strip())
            actual = oracion + ". "
    if actual:
        fragmentos.append(actual.strip())
    return fragmentos

# Cargar textos y fragmentar
textos = []
fuentes = []

for archivo in os.listdir(RUTA_TEXTOS):
    if archivo.endswith(".txt"):
        ruta = os.path.join(RUTA_TEXTOS, archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
            fragmentos = fragmentar_texto(contenido)
            for fragmento in fragmentos:
                if len(fragmento.strip()) > 50:
                    textos.append(fragmento.strip())
                    fuentes.append(archivo)

print(f"📄 Total de fragmentos listos para embeddings: {len(textos)}")

# Obtener embeddings uno por uno (porque Gemini solo permite batch=1)
embeddings = []
for i, texto in enumerate(textos):
    print(f"🔎 Generando embedding {i+1}/{len(textos)}...")
    try:
        response = model.get_embeddings([texto])
        embeddings.append(response[0].values)
    except Exception as e:
        print(f"❌ Error en fragmento {i+1}: {e}")
        embeddings.append([0.0] * 768)  # Relleno si falla

# Convertir a array NumPy para FAISS
embeddings_np = np.array(embeddings).astype("float32")

# Crear y guardar índice FAISS
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

faiss.write_index(index, ARCHIVO_FAISS)

# Guardar textos y fuentes
with open(ARCHIVO_METADATA, "wb") as f:
    pickle.dump({"textos": textos, "fuentes": fuentes}, f)

print("✅ Todo listo: Embeddings creados y guardados.")
