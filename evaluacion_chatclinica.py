# evaluacion_chatclinica.py
# Evaluación automática desde logs de conversación para ChatClinica

import csv
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# ----------- Funciones de evaluación -----------
def tokenize(text):
    return re.findall(r'\w+', text.lower())

def f1_precision_recall(gold, pred):
    gold_tokens = set(tokenize(gold))
    pred_tokens = set(tokenize(pred))
    true_positives = len(gold_tokens & pred_tokens)
    precision = true_positives / len(pred_tokens) if pred_tokens else 0
    recall = true_positives / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def bleu_score(gold, pred):
    reference = [tokenize(gold)]
    candidate = tokenize(pred)
    return sentence_bleu(reference, candidate)

def rouge_scores(gold, pred):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(gold, pred)

# ----------- Leer logs desde CSV -----------
input_csv = "conversaciones_chatclinica.csv"  # Debe tener: pregunta, generada, esperada
resultados = []

with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pregunta = row.get("pregunta_usuario", "")
        generada = row.get("respuesta_generada_por_chatclinica", "")
        esperada = row.get("respuesta_esperada", generada)  # si no hay esperada, usa generada

        p, r, f1 = f1_precision_recall(esperada, generada)
        bleu = bleu_score(esperada, generada)
        rouge = rouge_scores(esperada, generada)

        resultados.append({
            "pregunta": pregunta,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "bleu": round(bleu, 3),
            "rouge1": round(rouge['rouge1'].fmeasure, 3),
            "rougeL": round(rouge['rougeL'].fmeasure, 3)
        })

# ----------- Guardar resultados -----------
with open("metricas_chatclinica.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["pregunta", "precision", "recall", "f1", "bleu", "rouge1", "rougeL"])
    writer.writeheader()
    writer.writerows(resultados)

print("✅ Evaluación automática completada desde logs. Revisa 'metricas_chatclinica.csv'")
