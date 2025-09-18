"""
Configuración central para el pipeline de DM2
"""
from pathlib import Path
import os

# Configuración de rutas
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Crear directorios si no existen
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Archivos principales
DATA_FILE = "output-glucosa_labeled.csv"
DATA_PATH = BASE_DIR / DATA_FILE

# Configuración del modelo
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# Target y configuraciones específicas del dominio
TARGET = "Clase_DM"
GLUCOSE = "Resultado"
CLASS_ORDER = ["Normal", "Prediabetes", "Diabetes"]

# Variables a excluir (blacklist y PII)
BLACKLIST_COLS = [
    TARGET, GLUCOSE, "interpretacion", "Niveles_Altos_Glucosa",
    "Examen", "Analito", "Grupo_Analito", "Dm", "tipo_dm"
]

PII_COLS = [
    "identificacion", "Nombre_Completo", "nombres", "apellidos",
    "telefono", "direccion", "responsable_registro", "Fecha_Fin_Registro",
    "fecha_nacimiento"
]

# Configuración de métricas médicas
MEDICAL_METRICS = {
    "focus_class": "Diabetes",  # Clase de mayor importancia clínica
    "min_sensitivity": 0.8,     # Mínima sensibilidad aceptable para Diabetes
    "min_specificity": 0.7      # Mínima especificidad aceptable
}

# Configuración de balancing
BALANCING_CONFIG = {
    "method": "combined",  # "smote", "class_weights", "combined"
    "smote_k_neighbors": 3,  # Para datasets pequeños
    "class_weights": "balanced"
}

print(f"✅ Configuración cargada - Base: {BASE_DIR}")
print(f"📁 Data: {DATA_PATH.exists()}")