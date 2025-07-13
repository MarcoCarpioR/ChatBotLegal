import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directorio donde están tus PDFs
DOCUMENTS_DIR = "documentos_legales"
# Archivo de salida para el texto procesado (opcional, para depuración)
OUTPUT_FILE = "documentos_procesados.txt"

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or "" # Añadir or "" para manejar páginas vacías
    except Exception as e:
        print(f"Error al leer {pdf_path}: {e}")
    return text

def process_documents():
    processed_chunks = []

    print(f"Buscando documentos en: {DOCUMENTS_DIR}")
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"Error: La carpeta '{DOCUMENTS_DIR}' no existe.")
        return []

    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No se encontraron archivos PDF en '{DOCUMENTS_DIR}'.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    for filename in pdf_files:
        pdf_path = os.path.join(DOCUMENTS_DIR, filename)
        print(f"Procesando: {filename}")
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(f"Advertencia: No se extrajo texto de {filename}.")
            continue

        # Divide este documento en chunks individualmente
        chunks = text_splitter.create_documents([text])
        print(f"{len(chunks)} chunks creados para {filename}")

        for i, chunk_doc in enumerate(chunks):
            processed_chunks.append({
                "chunk_id": f"{filename}_chunk_{i}",  # ← ID único por documento
                "text_content": chunk_doc.page_content,
                "metadata": chunk_doc.metadata
            })

    print(f"\nTotal de chunks procesados: {len(processed_chunks)}")
    return processed_chunks


if __name__ == "__main__":
    print("Iniciando procesamiento de documentos...")
    processed_data = process_documents()
    if processed_data:
        print("\nProcesamiento completado. La data está lista para ser usada en embeddings y Vector Search.")
        # Opcional: guardar los chunks en un archivo JSON para inspección
        # import json
        # try:
        #     with open("chunks_list.json", "w", encoding="utf-8") as f:
        #         json.dump(processed_data, f, ensure_ascii=False, indent=4)
        #     print("Chunks guardados en 'chunks_list.json' para revisión.")
        # except Exception as e:
        #     print(f"Error al guardar 'chunks_list.json': {e}")
    else:
        print("El procesamiento de documentos no generó chunks. Revisa los mensajes anteriores para ver si hubo errores o advertencias.")