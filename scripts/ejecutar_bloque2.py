"""
Script de ejecucion del Bloque 2 -- Modelo y baselines clasicos.

Uso:
    python scripts/ejecutar_bloque2.py

Requiere: datos generados por ejecutar_bloque1.py
"""

import sys
import logging
from pathlib import Path

# Anadir raiz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bloque2_modelo import ejecutar_bloque2

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    resultados = ejecutar_bloque2()
