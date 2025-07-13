import os
import json
from google.cloud import aiplatform, storage
from langchain_google_vertexai import VertexAIEmbeddings
import time
from procesar_documentos import process_documents # Importa la función de tu script anterior

# --- CONFIGURACIÓN DE GCP ---
PROJECT_ID = "chatbotlegalarequipa"  # ¡REEMPLAZA CON LA ID DE TU PROYECTO!
REGION = "us-central1"           # ¡REEMPLAZA CON UNA REGIÓN ADECUADA!
INDEX_DISPLAY_NAME = "chatbot_legal_index"
EMBEDDING_MODEL_NAME = "text-embedding-004" # O "text-embedding-gecko@003" si tienes problemas con 004
GCS_BUCKET_NAME = f"{PROJECT_ID}-vector-search-data"
GCS_DATA_PATH = f"gs://{GCS_BUCKET_NAME}/vector_search_data/"
LOCAL_EMBEDDINGS_FILE = "embeddings_data.json" # Cambiado de .jsonl a .json

# Inicializa el cliente de Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

def generate_embeddings(texts):
    """Genera embeddings para una lista de textos usando Vertex AI."""
    try:
        embeddings_model = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"Generando embeddings con el modelo: {EMBEDDING_MODEL_NAME}...")
        vectors = embeddings_model.embed_documents(texts)
        print(f"Embeddings generados para {len(vectors)} textos.")
        return vectors
    except Exception as e:
        print(f"Error al generar embeddings: {e}")
        print("Asegúrate de que la Vertex AI API esté habilitada y de que tu proyecto tenga los permisos correctos.")
        print("También, verifica que el modelo de embeddings esté disponible en tu región.")
        return []

def upload_embeddings_to_gcs(processed_chunks):
    """
    Genera embeddings, los guarda en un archivo JSONL y lo sube a Cloud Storage.
    """
    if not processed_chunks:
        print("No hay chunks para generar embeddings y subir.")
        return None

    print(f"Generando embeddings para {len(processed_chunks)} chunks...")
    texts_to_embed = [chunk['text_content'] for chunk in processed_chunks]
    embeddings = generate_embeddings(texts_to_embed)

    if not embeddings:
        print("No se pudieron generar los embeddings. Abortando carga a GCS.")
        return None

    print(f"Creando archivo local '{LOCAL_EMBEDDINGS_FILE}' para la carga...")
    with open(LOCAL_EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(processed_chunks):
            if i < len(embeddings):
                data_item = {
                    "id": chunk['chunk_id'],
                    "embedding": embeddings[i],
                    "rest_metadata_field": {
                        "text_content": chunk['text_content'],
                    }
                }
                f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
    print(f"Archivo '{LOCAL_EMBEDDINGS_FILE}' creado.")

    storage_client = storage.Client(project=PROJECT_ID)
    try:
        bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
        print(f"Bucket '{GCS_BUCKET_NAME}' ya existe.")
    except Exception:
        print(f"Creando bucket '{GCS_BUCKET_NAME}' para Vertex AI Vector Search...")
        bucket = storage_client.create_bucket(GCS_BUCKET_NAME, location=REGION)
        print(f"Bucket '{GCS_BUCKET_NAME}' creado.")

    destination_blob_name = f"vector_search_data/{LOCAL_EMBEDDINGS_FILE}"
    blob = bucket.blob(destination_blob_name)
    print(f"Subiendo '{LOCAL_EMBEDDINGS_FILE}' a gs://{GCS_BUCKET_NAME}/{destination_blob_name}...")
    blob.upload_from_filename(LOCAL_EMBEDDINGS_FILE)
    print("Archivo subido a Cloud Storage.")

    return GCS_DATA_PATH

def create_or_update_vector_search_index(gcs_data_uri):
    """Crea o actualiza un índice y un endpoint en Vertex AI Vector Search."""
    if not gcs_data_uri:
        print("No se proporcionó URI de datos de GCS para el índice.")
        return None

    # Intentar buscar un índice existente
    existing_indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{INDEX_DISPLAY_NAME}"')
    index = None
    if existing_indexes:
        index = existing_indexes[0]
        print(f"Usando índice existente: {index.resource_name}")
        print("Actualizando el índice con los nuevos datos. Esto puede tardar tiempo (minutos a horas)...")
        update_operation = index.update_embeddings(
            contents_delta_uri=gcs_data_uri,
            is_complete_overwrite=True
        )
        update_operation.wait()
        print("Carga/actualización de embeddings al índice completada.")
    else:
        print(f"Creando nuevo índice: {INDEX_DISPLAY_NAME}...")
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=INDEX_DISPLAY_NAME,
            contents_delta_uri=gcs_data_uri,
            dimensions=768,
            approximate_neighbors_count=10,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=7,
            labels={"project": "chatbot-legal"}
        )
        print(f"Índice '{INDEX_DISPLAY_NAME}' creado. ID: {index.name}")
        print("La creación del índice puede tardar un tiempo (30-60 minutos o más), puedes monitorearlo en la consola de Vertex AI.")
        index.wait()
        print("Creación del índice completada.")

    # --- INICIO DE LA LÓGICA DE DESPLIEGUE DEL ENDPOINT ---
    index_endpoint = None # Inicializa para asegurar su estado

    print("Iniciando la lógica de despliegue del endpoint...") # Mensaje de depuración

    try:
        # Busca si ya hay un endpoint desplegado para este índice
        existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{INDEX_DISPLAY_NAME}_endpoint"'
        )

        if existing_endpoints:
            index_endpoint = existing_endpoints[0]
            print(f"Usando endpoint existente: {index_endpoint.resource_name}")

            # Verifica si el índice actual ya está desplegado en este endpoint
            is_index_deployed_on_endpoint = False
            for deployed_idx in index_endpoint.deployed_indexes:
                if deployed_idx.index  == index.resource_name:
                    is_index_deployed_on_endpoint = True
                    print(f"El índice '{index.resource_name}' ya está desplegado en este endpoint.")
                    break

            if not is_index_deployed_on_endpoint:
                print(f"Desplegando índice actualizado en el endpoint existente. Esto puede tardar...")
                index_endpoint.deploy_index(
                    index=index,
                    deployed_index_id=f"{INDEX_DISPLAY_NAME}_deployed",
                    machine_type="e2-standard-16",
                    min_replica_count=1,
                    max_replica_count=1,
                ).wait()
                print(f"Índice '{index.resource_name}' desplegado en endpoint existente.")
            # Si is_index_deployed_on_endpoint es True, index_endpoint ya está asignado desde existing_endpoints[0]

        else:
            print(f"Creando y desplegando endpoint para el índice. Esto puede tardar (10-20 minutos)...")
            # Primero crea el objeto endpoint
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=f"{INDEX_DISPLAY_NAME}_endpoint",
                public_endpoint_enabled=True
                #network=None
            )
            print(f"Endpoint '{index_endpoint.resource_name}' creado. Desplegando índice...")

            # Luego, despliega el índice en este nuevo endpoint
            index_endpoint.deploy_index(
                index=index,
                deployed_index_id=f"{INDEX_DISPLAY_NAME}_deployed",
                machine_type="e2-standard-16",
                min_replica_count=1,
                max_replica_count=1
            ).wait()
            print(f"Endpoint desplegado. Nombre: {index_endpoint.resource_name}")

    except Exception as e:
        print(f"ERROR: Fallo inesperado durante la operación del endpoint: {e}")
        # import traceback
        # traceback.print_exc() # Descomenta para un traceback más detallado si es necesario
        return None # Devuelve None si ocurre un error en esta sección

    # --- FIN DE LA LÓGICA DE DESPLIEGUE DEL ENDPOINT ---

    if index_endpoint:
        print("Proceso de indexación completado. El índice está listo para ser consultado.")
        return index_endpoint
    else:
        # Esta rama solo se alcanzaría si index_endpoint es None después del try-except
        # y si ninguna excepción fue capturada (lo cual sería muy inusual).
        print("ERROR FATAL: El endpoint no fue creado ni recuperado por una razón desconocida después de la lógica de despliegue.")
        return None

if __name__ == "__main__":
    print("Iniciando el proceso de indexación...")
    print("Primero, ejecutando el procesamiento de documentos...")
    all_processed_chunks = process_documents()

    if all_processed_chunks:
        print(f"Se obtuvieron {len(all_processed_chunks)} chunks para indexar.")

        gcs_uri = upload_embeddings_to_gcs(all_processed_chunks)

        if gcs_uri:
            vector_search_endpoint = create_or_update_vector_search_index(gcs_uri)
            if vector_search_endpoint:
                print(f"\n¡El chatbot está casi listo! El endpoint para consultas es: {vector_search_endpoint.resource_name}")
                print("Ahora puedes pasar al siguiente paso para conectar Gemini y hacer preguntas.")
            else:
                # Este mensaje se imprime si create_or_update_vector_search_index retorna None
                print("No se pudo crear o desplegar el endpoint de Vector Search.")
        else:
            print("No se pudo subir la data de embeddings a GCS. Revisa los errores.")
    else:
        print("No se pudieron obtener los chunks. Revisa el script 'procesar_documentos.py' y la carpeta de documentos.")