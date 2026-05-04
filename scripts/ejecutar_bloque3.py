"""
Script de ejecucion del Bloque 3 -- QAOA en simulador.

Uso:
    python scripts/ejecutar_bloque3.py

Requiere: datos del Bloque 1 y modulo del Bloque 2.
NOTA: La ejecucion puede tardar varios minutos dependiendo del hardware.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bloque3_qaoa import ejecutar_bloque3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    resultados = ejecutar_bloque3()
