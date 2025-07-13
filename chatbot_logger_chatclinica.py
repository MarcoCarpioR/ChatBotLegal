# chatbot_logger_chatclinica.py
# Script mejorado para simular preguntas clave al chatbot ChatClinica y guardar respuestas en CSV correctamente alineadas

from vertexai.generative_models import GenerativeModel
import csv
import os
import time


gemini_model = GenerativeModel("gemini-2.5-pro")

# Preguntas clave sobre consultorios dentales (evaluación de calidad)
preguntas = [
    # 🏢 Infraestructura general
    #"responde todo en una sola oracion de 15 palabras o menos"
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas ¿De qué tamaño debe ser el consultorio dental?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Es obligatorio tener un baño en el consultorio?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Cuántas unidades dentales puedo tener?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Debo tener un área de esterilización?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿La sala de espera es obligatoria?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué requisitos debe cumplir la sala de procedimientos?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué metraje mínimo se exige por unidad dental?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué condiciones debe tener el área de esterilización?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Cuántos consultorios puede tener un centro odontológico privado?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Cuál es el área mínima para un consultorio de ortodoncia?",

    # 🔌 Equipamiento y mobiliario
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué equipamiento es obligatorio en un consultorio dental?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Debo tener autoclave o esterilizador específico?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Es necesario contar con lavamanos en cada consultorio?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué mobiliario debe estar presente en la sala de espera?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué características debe tener la iluminación del consultorio?",

    # 🧑‍⚕️ Requisitos legales y normativos
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué licencia municipal necesito para abrir un consultorio dental?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Necesito autorización del Ministerio de Salud?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Cuáles son los requisitos del Colegio Odontológico del Perú?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué normas técnicas debo cumplir para operar legalmente?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Es obligatorio tener un libro de reclamaciones?",

    # ⚠️ Salubridad y seguridad
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Es necesario tener un protocolo de bioseguridad?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué materiales son obligatorios para la gestión de residuos?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Debo contar con señalización de seguridad interna?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿El consultorio debe tener ventilación natural?",

    # 👥 Personal y atención al paciente
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Puede atender un técnico dental en lugar de un odontólogo?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Cuántos odontólogos pueden trabajar en un solo consultorio?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Es obligatorio llevar historia clínica de cada paciente?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Puedo contratar personal administrativo sin título en salud?",

    # 🛠️ Otros aspectos relevantes
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Puedo operar un consultorio dental en mi domicilio?",
    "en una linea, sin saltos de linea, 30 palabras o menos y sin comas¿Qué pasa si no cumplo con las normas de infraestructura?"

]

# Archivo CSV de salida
csv_path = "conversaciones_chatclinica.csv"
file_exists = os.path.isfile(csv_path)

with open(csv_path, "w", newline='', encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["pregunta_usuario", "respuesta_generada_por_chatclinica", "respuesta_esperada"])

    for i, pregunta in enumerate(preguntas):
        print(f"\U0001F9E0 Pregunta {i+1}/{len(preguntas)}: {pregunta}")
        try:
            respuesta = gemini_model.generate_content(pregunta)
            texto_respuesta = respuesta.text.strip()
            print(f"\U0001F916 Respuesta: {texto_respuesta[:200]}...\n")

            # Guardar en CSV
            writer.writerow([pregunta, texto_respuesta, ""])

            time.sleep(1)  # Espaciado para evitar sobrecargar API
        except Exception as e:
            print(f"❌ Error al generar respuesta: {e}\n")
            writer.writerow([pregunta, "[ERROR]", ""])

print("✅ Finalizó la simulación y registro de respuestas en 'conversaciones_chatclinica.csv'")
