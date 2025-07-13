import pytesseract
from pdf2image import convert_from_path
import os

# Ruta al ejecutable de Tesseract en tu sistema
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Ruta al binario de Poppler (ajusta si es diferente)
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"

# Carpetas de entrada y salida
RUTA_ENTRADA = "pdfs_originales"    # Carpeta con tus PDFs escaneados
RUTA_SALIDA = "textos_extraidos"    # Carpeta donde se guardarán los textos extraídos

# Crear carpeta de salida si no existe
os.makedirs(RUTA_SALIDA, exist_ok=True)

def ocr_pdf(pdf_path, output_txt_path, lang="spa"):
    print(f"🕵️ Procesando: {os.path.basename(pdf_path)}...")
    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        texto_total = ""
        for i, page in enumerate(pages):
            texto = pytesseract.image_to_string(page, lang=lang)
            texto_total += f"\n\n=== PÁGINA {i+1} ===\n{texto}"
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(texto_total)
        print(f"✅ Guardado: {output_txt_path}")
    except Exception as e:
        print(f"❌ Error procesando {pdf_path}: {e}")

# Procesar todos los PDFs en la carpeta
for archivo in os.listdir(RUTA_ENTRADA):
    if archivo.lower().endswith(".pdf"):
        ruta_pdf = os.path.join(RUTA_ENTRADA, archivo)
        nombre_sin_ext = os.path.splitext(archivo)[0]
        ruta_salida = os.path.join(RUTA_SALIDA, f"{nombre_sin_ext}.txt")
        ocr_pdf(ruta_pdf, ruta_salida)

print("🎉 ¡OCR completado para todos los PDFs!")
