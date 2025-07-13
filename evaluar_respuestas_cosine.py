# evaluar_respuestas_cosine.py
# Evalúa la similitud por coseno entre respuestas esperadas y generadas por el chatbot

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cargar modelo preentrenado de embeddings (recomendado para textos largos)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar archivo CSV con las respuestas
archivo_csv = "conversaciones_chatclinica.csv"
df = pd.read_csv(archivo_csv)

# Eliminar filas con errores o respuestas vacías
df = df[df["respuesta_generada_por_chatclinica"] != "[ERROR]"]
df = df.dropna(subset=["respuesta_generada_por_chatclinica", "respuesta_esperada"])

# Calcular embeddings de todas las respuestas
generadas_embeddings = model.encode(df["respuesta_generada_por_chatclinica"].tolist(), convert_to_tensor=True)
esperadas_embeddings = model.encode(df["respuesta_esperada"].tolist(), convert_to_tensor=True)

# Calcular similitudes por coseno
similaridades = util.cos_sim(generadas_embeddings, esperadas_embeddings)

# Extraer diagonal (comparación 1:1 entre pares)
df["similaridad_coseno"] = [float(similaridades[i][i]) for i in range(len(df))]

# Guardar nuevo CSV con resultado
df.to_csv("evaluacion_similitud_coseno.csv", index=False, encoding="utf-8-sig")

print("✅ Evaluación completada. Resultados guardados en 'evaluacion_similitud_coseno.csv'")
	