"""
Configuración global del TFM.

Centraliza todos los parámetros del proyecto: tickers, fechas,
tamaños de instancia y rutas de archivos. Cualquier cambio de
configuración se hace aquí para garantizar reproducibilidad.
"""

import os
from pathlib import Path

# =============================================================================
# Rutas del proyecto
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTADOS = PROJECT_ROOT / "resultados"

# Crear directorios si no existen
for d in [DATA_RAW, DATA_PROCESSED, RESULTADOS]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Universo de activos
# =============================================================================
# 25 mega-caps USA líquidas y estables, con datos disponibles en Yahoo Finance
# durante todo el periodo 2019-2024.
# Criterio de selección: capitalización >100B USD, volumen diario alto,
# presencia continua en S&P 500 durante el periodo.
TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet (Google)
    "META",   # Meta Platforms (Facebook) — ticker cambió de FB a META en 2022
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    "BRK-B",  # Berkshire Hathaway
    "JPM",    # JPMorgan Chase
    "JNJ",    # Johnson & Johnson
    "V",      # Visa
    "PG",     # Procter & Gamble
    "UNH",    # UnitedHealth Group
    "HD",     # Home Depot
    "MA",     # Mastercard
    "DIS",    # Walt Disney
    "ADBE",   # Adobe
    "CRM",    # Salesforce
    "NFLX",   # Netflix
    "PFE",    # Pfizer
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    "CSCO",   # Cisco Systems
    "XOM",    # Exxon Mobil
    "WMT",    # Walmart
]

N_ACTIVOS = len(TICKERS)  # 25

# =============================================================================
# Ventana temporal
# =============================================================================
FECHA_INICIO = "2019-01-01"
FECHA_FIN = "2024-12-31"

# Separación entrenamiento / evaluación out-of-sample
FECHA_CORTE_TRAIN = "2022-12-31"  # Train: 2019-01-01 a 2022-12-31
FECHA_INICIO_TEST = "2023-01-01"  # Test:  2023-01-01 a 2024-12-31

# =============================================================================
# Parámetros del modelo
# =============================================================================
# Tamaños de instancia para benchmark
TAMANOS_INSTANCIA = [8, 12, 16]

# Cardinalidad por instancia: k = n/4 (redondeado)
def cardinalidad(n: int) -> int:
    """Calcula la cardinalidad k = round(n/4) para una instancia de tamaño n."""
    return round(n / 4)

# Tasa libre de riesgo anualizada (para cálculo de Sharpe)
# Usamos un promedio representativo del periodo 2019-2024
RISK_FREE_RATE = 0.02

# Días de trading por año (para anualización)
TRADING_DAYS_PER_YEAR = 252

# =============================================================================
# Parámetros de experimentación
# =============================================================================
N_SEMILLAS = 10
SEMILLA_BASE = 42

# QAOA (para bloques posteriores, pero centralizados aquí)
QAOA_P_VALUES = [1, 2]
QAOA_MAX_ITER = 100
QAOA_SHOTS = 1024
QAOA_OPTIMIZER = "COBYLA"

# Parámetro de riesgo para el modelo media-varianza QUBO
# q ∈ [0, 1]: q=0 maximiza retorno puro, q=1 minimiza riesgo puro
RISK_FACTOR = 0.5

# Factor de penalización para restricción de cardinalidad en QUBO
# Se multiplica por la escala del problema para obtener P adaptativo
PENALTY_FACTOR = 10.0

# Simulated Annealing
SA_N_ITER = 10000
SA_T_INIT = 10.0
SA_ALPHA = 0.995  # Tasa de enfriamiento geométrico

# =============================================================================
# Archivos de salida del Bloque 1
# =============================================================================
ARCHIVO_PRECIOS_RAW = DATA_RAW / "precios_cierre.csv"
ARCHIVO_PRECIOS_CLEAN = DATA_PROCESSED / "precios_cierre_limpio.csv"
ARCHIVO_RETORNOS = DATA_PROCESSED / "retornos_diarios.csv"
ARCHIVO_RETORNOS_TRAIN = DATA_PROCESSED / "retornos_train.csv"
ARCHIVO_RETORNOS_TEST = DATA_PROCESSED / "retornos_test.csv"
ARCHIVO_COV_MATRIX = DATA_PROCESSED / "matriz_covarianzas.csv"
ARCHIVO_MEDIA_RETORNOS = DATA_PROCESSED / "media_retornos.csv"
ARCHIVO_STATS_DESCRIPTIVAS = DATA_PROCESSED / "estadisticas_descriptivas.csv"
ARCHIVO_REPORTE_DATOS = RESULTADOS / "reporte_bloque1.txt"

# =============================================================================
# Archivos de salida del Bloque 2
# =============================================================================
ARCHIVO_RESULTADOS_CLASICOS = RESULTADOS / "resultados_clasicos.csv"
ARCHIVO_REPORTE_BLOQUE2 = RESULTADOS / "reporte_bloque2.txt"
