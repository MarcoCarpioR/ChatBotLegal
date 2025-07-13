# chatbot_logic.py (VERSIÓN FINAL Y COMPLETA CON LA CORRECCIÓN DE ACCESO A METADATOS)

import os
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from typing import List, Dict

# --- CONFIGURACIÓN DE GCP ---
# ¡IMPORTANTE! Reemplaza con tus propios valores
PROJECT_ID = "323193639525" # Reemplaza con el ID de tu proyecto de GCP
REGION = "us-central1"
VECTOR_SEARCH_ENDPOINT_ID = "2532889961326182400"
EMBEDDING_MODEL_NAME = "text-embedding-004"
GEMINI_MODEL_NAME = "gemini-pro"
INDEX_DISPLAY_NAME_DEPLOYED = "chatbot_legal_index_deployed"

# Inicializa la API de Google Cloud Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

try:
    # Inicializa los modelos de embeddings y Gemini
    embedding_model = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    gemini_model = ChatVertexAI(model_name=GEMINI_MODEL_NAME)
    print("Modelos de Embeddings y Gemini inicializados correctamente.")
except Exception as e:
    print(f"Error al inicializar modelos. Asegúrate de tener las APIs habilitadas y permisos correctos: {e}")
    exit()

try:
    # Conecta al endpoint de Vector Search desplegado
    vector_search_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/indexEndpoints/{VECTOR_SEARCH_ENDPOINT_ID}"
    )
    print(f"Conectado al endpoint de Vector Search: {vector_search_endpoint.resource_name}")
except Exception as e:
    print(f"Error al conectar al endpoint de Vector Search. Verifica el ID y los permisos: {e}")
    exit()

def ask_chatbot(user_query: str) -> str:
    print(f"\nPregunta del usuario: {user_query}")

    try:
        print("Generando embedding para la pregunta...")
        query_embedding = embedding_model.embed_query(user_query)

        print("Consultando Vector Search para documentos relevantes...")
        # Realiza la consulta a Vector Search
        # ¡IMPORTANTE! return_full_datapoint=True para obtener los metadatos completos
        response = vector_search_endpoint.find_neighbors(
            deployed_index_id=INDEX_DISPLAY_NAME_DEPLOYED,
            queries=[query_embedding],
            num_neighbors=5, # Recupera los 5 fragmentos más relevantes
            return_full_datapoint=True # Solicita el datapoint completo con metadatos
        )

        context_chunks_content: List[str] = []
        neighbors_list = []

        # Adapta la respuesta para obtener la lista de vecinos
        if response:
            if hasattr(response[0], 'neighbors'):
                neighbors_list = response[0].neighbors
            elif isinstance(response[0], list): # En caso de que la respuesta sea una lista directa
                neighbors_list = response[0]
            else:
                print(f"Advertencia: Formato de respuesta inesperado de find_neighbors: {type(response[0])}")

        if neighbors_list:
            print(f"Se encontraron {len(neighbors_list)} vecinos relevantes.")
            for neighbor in neighbors_list:
                found_text_content = False
                
                # --- CORRECCIÓN CLAVE AQUÍ ---
                # Accede a los metadatos a través de 'datapoint.rest_metadata_fields', que AHORA SABEMOS ES UN DICCIONARIO
                if hasattr(neighbor, 'datapoint') and hasattr(neighbor.datapoint, 'rest_metadata_field'):
                    # Si 'rest_metadata_fields' es un diccionario, accedemos directamente por la clave
                    rest_metadata_dict = neighbor.datapoint.rest_metadata_field
                    if 'text_content' in rest_metadata_dict and isinstance(rest_metadata_dict['text_content'], str):
                        context_chunks_content.append(rest_metadata_dict['text_content'])
                        found_text_content = True
                        print(f"DEBUG: 'text_content' encontrado en datapoint.rest_metadata_fields para ID: {neighbor.id}")
                    else:
                        print(f"Advertencia: 'text_content' no encontrado o no es string en el diccionario de rest_metadata_fields para ID: {neighbor.id}. Contenido del diccionario: {rest_metadata_dict}")
                
                if not found_text_content:
                    print(f"Advertencia: 'text_content' no pudo ser extraído para el ID de vecino: {neighbor.id}.")

        else:
            print("No se encontraron documentos relevantes en Vector Search.")

        # Une los fragmentos de contexto para pasarlos al modelo de lenguaje
        context = "\n\n".join(context_chunks_content)

        if not context:
            return "Lo siento, no pude encontrar información relevante en mis documentos para responder a tu pregunta. Por favor, intenta reformularla."

        # Construye el prompt para el modelo Gemini
        prompt = f"""Eres un asistente legal experto en la habilitación de consultorios dentales en Arequipa, Perú, basado en la normativa vigente.
        Utiliza el siguiente contexto proporcionado para responder a la pregunta del usuario.
        Si la pregunta no puede ser respondida con la información del contexto, indica que no tienes esa información en tus documentos y sugieres buscar asesoría legal profesional.
        No inventes información.

        Contexto:
        {context}

        Pregunta del usuario: {user_query}

        Respuesta:
        """

        print("Generando respuesta con Gemini...")
        ai_response = gemini_model.predict(prompt)
        return ai_response.text

    except Exception as e:
        print(f"Ocurrió un error inesperado durante el procesamiento: {e}")
        return "Lo siento, hubo un problema técnico al intentar responder tu pregunta. Por favor, inténtalo de nuevo más tarde."

# Punto de entrada principal del script
if __name__ == "__main__":
    print("Iniciando la prueba del chatbot. Escribe 'salir' para terminar.")
    while True:
        user_input = input("\nTu pregunta: ")
        if user_input.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        response = ask_chatbot(user_input)
        print(f"\nRespuesta del Chatbot:\n{response}")