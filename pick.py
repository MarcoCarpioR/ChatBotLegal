import os
import pickle

carpeta = "textos_extraidos"
archivos = [f for f in os.listdir(carpeta) if f.endswith(".txt")]

chunks = []
for archivo in archivos:
    with open(os.path.join(carpeta, archivo), "r", encoding="utf-8") as f:
        contenido = f.read()
        for i in range(0, len(contenido), 300):
            chunk = contenido[i:i+400]
            if len(chunk) > 100:
                chunks.append(chunk.strip())

with open("textos_normas.pkl", "wb") as f:
    pickle.dump(chunks, f)
