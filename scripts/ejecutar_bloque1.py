"""
Script de ejecución del Bloque 1 — Preparación del marco de datos.

Uso:
    python scripts/ejecutar_bloque1.py

Genera todos los archivos de datos necesarios para los bloques posteriores.
"""

import sys
import logging
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bloque1_datos import ejecutar_bloque1

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    resultados = ejecutar_bloque1()
