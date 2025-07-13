# evaluar_respuestas_metricas.py
# Calcula métricas BLEU, ROUGE-L, BERTScore y Coseno entre respuestas generadas y respuestas esperadas

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import bert_score

# Cargar archivo con respuestas
csv_path = "respuestas_chatbots.csv"
df = pd.read_csv(csv_path)

# Mostrar columnas detectadas para verificar nombres
print("📋 Columnas detectadas:", df.columns.tolist())

# Columnas esperadas: pregunta_usuario, respuesta_generada_por_chatclinica, respuesta_esperada
cols = ["respuesta_generada_por_chatclinica"]

# Inicializar modelos
sbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

# Resultados
resultados = []

print("🔍 Calculando métricas...\n")

for i, row in df.iterrows():
    entrada = {
        "pregunta": row["pregunta_usuario"],
        "respuesta_esperada": row["respuesta_esperada"] if "respuesta_esperada" in row else ""
    }
    ref = row["respuesta_esperada"]

    for modelo in cols:
        hip = row[modelo]

        # BLEU
        bleu = sentence_bleu([ref.split()], hip.split(), smoothing_function=smoothie)

        # ROUGE-L
        try:
            score = scorer.score(ref, hip)
            rouge_l = score["rougeL"].fmeasure
        except:
            rouge_l = 0.0

        # Coseno SBERT
        try:
            emb_ref = sbert.encode(ref, convert_to_tensor=True)
            emb_hip = sbert.encode(hip, convert_to_tensor=True)
            sim_coseno = float(util.pytorch_cos_sim(emb_ref, emb_hip).item())
        except:
            sim_coseno = 0.0

        # BERTScore
        try:
            P, R, F1 = bert_score.score([hip], [ref], lang="es", verbose=False)
            bert_f1 = F1[0].item()
        except:
            bert_f1 = 0.0

        entrada.update({
            f"{modelo}_bleu": bleu,
            f"{modelo}_rouge_l": rouge_l,
            f"{modelo}_coseno": sim_coseno,
            f"{modelo}_bertscore_f1": bert_f1
        })

    resultados.append(entrada)

# Exportar
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("evaluacion_metricas_chatbots.csv", index=False, encoding="utf-8-sig")

print("✅ Evaluación completada. Archivo generado: evaluacion_metricas_chatbots.csv")
