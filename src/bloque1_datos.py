"""
Bloque 1 — Preparación del marco de datos.

Módulo que implementa todo el pipeline de datos del TFM:
  1. Descarga de precios de cierre ajustados desde Yahoo Finance.
  2. Limpieza y validación de datos.
  3. Cálculo de retornos logarítmicos diarios.
  4. Separación train/test.
  5. Cálculo de estadísticos base (media de retornos, matriz de covarianzas).
  6. Estadísticas descriptivas y reporte.

Todas las funciones son deterministas y reproducibles dado el mismo input.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    TICKERS,
    FECHA_INICIO,
    FECHA_FIN,
    FECHA_CORTE_TRAIN,
    FECHA_INICIO_TEST,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    ARCHIVO_PRECIOS_RAW,
    ARCHIVO_PRECIOS_CLEAN,
    ARCHIVO_RETORNOS,
    ARCHIVO_RETORNOS_TRAIN,
    ARCHIVO_RETORNOS_TEST,
    ARCHIVO_COV_MATRIX,
    ARCHIVO_MEDIA_RETORNOS,
    ARCHIVO_STATS_DESCRIPTIVAS,
    ARCHIVO_REPORTE_DATOS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. DESCARGA DE DATOS
# =============================================================================
def descargar_precios(
    tickers: list[str] = TICKERS,
    inicio: str = FECHA_INICIO,
    fin: str = FECHA_FIN,
    guardar: bool = True,
) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados desde Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        Lista de tickers a descargar.
    inicio, fin : str
        Fechas en formato 'YYYY-MM-DD'.
    guardar : bool
        Si True, guarda el CSV en data/raw/.

    Returns
    -------
    pd.DataFrame
        DataFrame con precios de cierre ajustados (índice=fecha, columnas=tickers).
    """
    logger.info(f"Descargando datos de {len(tickers)} tickers: {inicio} a {fin}")

    # Descargar todos los tickers a la vez (más eficiente)
    datos_raw = yf.download(
        tickers=tickers,
        start=inicio,
        end=fin,
        auto_adjust=True,  # Usa precios ajustados por splits y dividendos
        progress=True,
    )

    # yf.download con múltiples tickers devuelve un MultiIndex
    # con niveles ['Price', 'Ticker']. Extraemos 'Close'.
    if isinstance(datos_raw.columns, pd.MultiIndex):
        # El nivel 'Price' contiene 'Close', 'High', 'Low', etc.
        # Seleccionamos solo 'Close' y eliminamos el nivel Price
        precios = datos_raw.xs("Close", level="Price", axis=1)
    else:
        precios = datos_raw[["Close"]]
        precios.columns = tickers

    # Asegurar que las columnas estén en el orden de TICKERS
    precios = precios[tickers]

    logger.info(f"Datos descargados: {precios.shape[0]} filas x {precios.shape[1]} columnas")
    logger.info(f"Rango de fechas: {precios.index[0].date()} a {precios.index[-1].date()}")

    if guardar:
        precios.to_csv(ARCHIVO_PRECIOS_RAW)
        logger.info(f"Precios brutos guardados en {ARCHIVO_PRECIOS_RAW}")

    return precios


# =============================================================================
# 2. LIMPIEZA Y VALIDACIÓN
# =============================================================================
def limpiar_precios(precios: pd.DataFrame, guardar: bool = True) -> pd.DataFrame:
    """
    Limpia y valida los datos de precios.

    Pasos:
      1. Elimina filas donde todos los valores son NaN (festivos, etc.).
      2. Forward-fill para días con datos parciales (máx 5 días consecutivos).
      3. Verifica que no queden NaN residuales.
      4. Verifica que no haya precios negativos o cero.
      5. Ordena por fecha.

    Parameters
    ----------
    precios : pd.DataFrame
        Precios brutos descargados.
    guardar : bool
        Si True, guarda CSV limpio.

    Returns
    -------
    pd.DataFrame
        Precios limpios validados.
    """
    logger.info("Iniciando limpieza de datos...")

    n_original = len(precios)

    # 1. Eliminar filas completamente vacías
    precios = precios.dropna(how="all")
    n_tras_dropna = len(precios)
    if n_original != n_tras_dropna:
        logger.info(f"  Eliminadas {n_original - n_tras_dropna} filas completamente vacías")

    # 2. Contar NaN antes de rellenar
    nan_antes = precios.isna().sum()
    tickers_con_nan = nan_antes[nan_antes > 0]
    if len(tickers_con_nan) > 0:
        logger.warning(f"  Tickers con NaN antes de limpieza:")
        for ticker, count in tickers_con_nan.items():
            logger.warning(f"    {ticker}: {count} NaN ({count/len(precios)*100:.2f}%)")

    # 3. Forward-fill (máximo 5 días consecutivos para no inventar datos)
    precios = precios.ffill(limit=5)

    # 4. Backward-fill residual (para NaN al inicio)
    precios = precios.bfill(limit=5)

    # 5. Verificar NaN residuales
    nan_despues = precios.isna().sum()
    tickers_nan_residual = nan_despues[nan_despues > 0]
    if len(tickers_nan_residual) > 0:
        logger.error(f"  ¡ALERTA! NaN residuales tras limpieza:")
        for ticker, count in tickers_nan_residual.items():
            logger.error(f"    {ticker}: {count} NaN restantes")
        # Eliminar columnas con NaN residuales (no deberían existir con mega-caps)
        precios = precios.dropna(axis=1)
        logger.warning(f"  Columnas con NaN eliminadas. Quedan {precios.shape[1]} tickers.")

    # 6. Verificar precios válidos
    assert (precios > 0).all().all(), "¡Error! Se encontraron precios <= 0"

    # 7. Ordenar por fecha
    precios = precios.sort_index()

    logger.info(f"Datos limpios: {precios.shape[0]} filas x {precios.shape[1]} columnas")

    if guardar:
        precios.to_csv(ARCHIVO_PRECIOS_CLEAN)
        logger.info(f"Precios limpios guardados en {ARCHIVO_PRECIOS_CLEAN}")

    return precios


# =============================================================================
# 3. CÁLCULO DE RETORNOS
# =============================================================================
def calcular_retornos(
    precios: pd.DataFrame,
    tipo: str = "log",
    guardar: bool = True,
) -> pd.DataFrame:
    """
    Calcula retornos diarios a partir de precios de cierre.

    Parameters
    ----------
    precios : pd.DataFrame
        Precios de cierre limpios.
    tipo : str
        'log' para retornos logarítmicos, 'simple' para retornos simples.
    guardar : bool
        Si True, guarda CSV.

    Returns
    -------
    pd.DataFrame
        Retornos diarios.
    """
    if tipo == "log":
        retornos = np.log(precios / precios.shift(1))
    elif tipo == "simple":
        retornos = precios.pct_change()
    else:
        raise ValueError(f"Tipo de retorno no válido: {tipo}. Usar 'log' o 'simple'.")

    # Eliminar la primera fila (NaN por el shift)
    retornos = retornos.dropna()

    logger.info(f"Retornos {tipo} calculados: {retornos.shape[0]} observaciones")

    if guardar:
        retornos.to_csv(ARCHIVO_RETORNOS)
        logger.info(f"Retornos guardados en {ARCHIVO_RETORNOS}")

    return retornos


# =============================================================================
# 4. SEPARACIÓN TRAIN / TEST
# =============================================================================
def separar_train_test(
    retornos: pd.DataFrame,
    fecha_corte: str = FECHA_CORTE_TRAIN,
    fecha_inicio_test: str = FECHA_INICIO_TEST,
    guardar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa retornos en conjuntos de entrenamiento y test.

    Parameters
    ----------
    retornos : pd.DataFrame
        Retornos diarios completos.
    fecha_corte : str
        Última fecha del periodo de entrenamiento.
    fecha_inicio_test : str
        Primera fecha del periodo de test.
    guardar : bool
        Si True, guarda CSVs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (retornos_train, retornos_test)
    """
    retornos_train = retornos.loc[:fecha_corte]
    retornos_test = retornos.loc[fecha_inicio_test:]

    logger.info(f"Train: {retornos_train.shape[0]} días ({retornos_train.index[0].date()} a {retornos_train.index[-1].date()})")
    logger.info(f"Test:  {retornos_test.shape[0]} días ({retornos_test.index[0].date()} a {retornos_test.index[-1].date()})")

    if guardar:
        retornos_train.to_csv(ARCHIVO_RETORNOS_TRAIN)
        retornos_test.to_csv(ARCHIVO_RETORNOS_TEST)
        logger.info("Retornos train/test guardados")

    return retornos_train, retornos_test


# =============================================================================
# 5. ESTADÍSTICOS BASE PARA EL MODELO
# =============================================================================
def calcular_parametros_modelo(
    retornos_train: pd.DataFrame,
    guardar: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calcula los parámetros base del modelo media-varianza a partir
    de los retornos de entrenamiento.

    Parameters
    ----------
    retornos_train : pd.DataFrame
        Retornos diarios del periodo de entrenamiento.
    guardar : bool
        Si True, guarda CSVs.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        (media_retornos_anualizados, matriz_covarianzas_anualizada)
    """
    # Media de retornos diarios → anualizada
    media_diaria = retornos_train.mean()
    media_anual = media_diaria * TRADING_DAYS_PER_YEAR

    # Matriz de covarianzas diaria → anualizada
    cov_diaria = retornos_train.cov()
    cov_anual = cov_diaria * TRADING_DAYS_PER_YEAR

    logger.info("Parámetros del modelo calculados (anualizados):")
    logger.info(f"  Retorno medio anual: [{media_anual.min():.4f}, {media_anual.max():.4f}]")
    logger.info(f"  Volatilidad anual:   [{np.sqrt(np.diag(cov_anual)).min():.4f}, {np.sqrt(np.diag(cov_anual)).max():.4f}]")

    if guardar:
        media_anual.to_csv(ARCHIVO_MEDIA_RETORNOS, header=["retorno_medio_anual"])
        cov_anual.to_csv(ARCHIVO_COV_MATRIX)
        logger.info("Media y covarianza guardados")

    return media_anual, cov_anual


# =============================================================================
# 6. ESTADÍSTICAS DESCRIPTIVAS
# =============================================================================
def generar_estadisticas_descriptivas(
    precios: pd.DataFrame,
    retornos: pd.DataFrame,
    retornos_train: pd.DataFrame,
    media_anual: pd.Series,
    cov_anual: pd.DataFrame,
    guardar: bool = True,
) -> pd.DataFrame:
    """
    Genera tabla de estadísticas descriptivas por activo.

    Incluye: precio medio, retorno anualizado, volatilidad anualizada,
    Sharpe ratio, retorno acumulado, asimetría y curtosis.
    """
    vol_anual = np.sqrt(np.diag(cov_anual))
    sharpe = (media_anual - RISK_FREE_RATE) / vol_anual

    # Retorno acumulado total del periodo de entrenamiento
    retorno_acumulado = (np.exp(retornos_train.sum()) - 1) * 100  # en %

    stats = pd.DataFrame({
        "Precio_medio_USD": precios.mean().round(2),
        "Retorno_anual_%": (media_anual * 100).round(2),
        "Volatilidad_anual_%": (vol_anual * 100).round(2),
        "Sharpe_ratio": sharpe.round(3),
        "Retorno_acumulado_train_%": retorno_acumulado.round(2),
        "Asimetria": retornos_train.skew().round(3),
        "Curtosis": retornos_train.kurtosis().round(3),
    })

    stats = stats.sort_values("Sharpe_ratio", ascending=False)

    logger.info("\n--- Estadísticas descriptivas (ordenadas por Sharpe) ---")
    logger.info(f"\n{stats.to_string()}")

    if guardar:
        stats.to_csv(ARCHIVO_STATS_DESCRIPTIVAS)
        logger.info(f"\nEstadísticas guardadas en {ARCHIVO_STATS_DESCRIPTIVAS}")

    return stats


# =============================================================================
# 7. REPORTE CONSOLIDADO
# =============================================================================
def generar_reporte(
    precios: pd.DataFrame,
    precios_limpios: pd.DataFrame,
    retornos: pd.DataFrame,
    retornos_train: pd.DataFrame,
    retornos_test: pd.DataFrame,
    media_anual: pd.Series,
    cov_anual: pd.DataFrame,
    stats: pd.DataFrame,
) -> str:
    """
    Genera un reporte textual consolidado del Bloque 1 para la memoria.
    """
    vol_anual = np.sqrt(np.diag(cov_anual))

    lineas = [
        "=" * 70,
        "REPORTE BLOQUE 1 — PREPARACIÓN DEL MARCO DE DATOS",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. DATOS DESCARGADOS",
        f"   Fuente: Yahoo Finance (precios de cierre ajustados)",
        f"   Tickers: {len(precios.columns)} activos",
        f"   Periodo: {precios.index[0].date()} a {precios.index[-1].date()}",
        f"   Días de trading: {len(precios)}",
        "",
        "2. LIMPIEZA",
        f"   Filas originales: {len(precios)}",
        f"   Filas tras limpieza: {len(precios_limpios)}",
        f"   Tickers finales: {len(precios_limpios.columns)}",
        f"   NaN residuales: {precios_limpios.isna().sum().sum()}",
        "",
        "3. RETORNOS",
        f"   Tipo: logarítmicos",
        f"   Observaciones totales: {len(retornos)}",
        f"   Periodo entrenamiento: {retornos_train.index[0].date()} a {retornos_train.index[-1].date()} ({len(retornos_train)} días)",
        f"   Periodo test: {retornos_test.index[0].date()} a {retornos_test.index[-1].date()} ({len(retornos_test)} días)",
        "",
        "4. PARÁMETROS DEL MODELO (anualizados, periodo train)",
        f"   Retorno medio anual:",
        f"     Mínimo:  {media_anual.min():.4f} ({media_anual.idxmin()})",
        f"     Máximo:  {media_anual.max():.4f} ({media_anual.idxmax()})",
        f"     Mediana: {media_anual.median():.4f}",
        f"   Volatilidad anual:",
        f"     Mínima:  {vol_anual.min():.4f} ({precios_limpios.columns[np.argmin(vol_anual)]})",
        f"     Máxima:  {vol_anual.max():.4f} ({precios_limpios.columns[np.argmax(vol_anual)]})",
        f"     Mediana: {np.median(vol_anual):.4f}",
        "",
        "5. MATRIZ DE COVARIANZAS",
        f"   Dimensión: {cov_anual.shape[0]}x{cov_anual.shape[1]}",
        f"   Definida positiva: {np.all(np.linalg.eigvalsh(cov_anual) > 0)}",
        f"   Correlación media (off-diagonal): {cov_anual.values[np.triu_indices(len(cov_anual), k=1)].mean() / (vol_anual.mean()**2):.4f}",
        "",
        "6. TOP 5 ACTIVOS POR SHARPE RATIO (train)",
    ]

    top5 = stats.head(5)
    for ticker in top5.index:
        row = top5.loc[ticker]
        lineas.append(
            f"   {ticker:6s}  Ret={row['Retorno_anual_%']:+7.2f}%  "
            f"Vol={row['Volatilidad_anual_%']:6.2f}%  "
            f"Sharpe={row['Sharpe_ratio']:+.3f}"
        )

    lineas.extend([
        "",
        "7. BOTTOM 5 ACTIVOS POR SHARPE RATIO (train)",
    ])
    bottom5 = stats.tail(5)
    for ticker in bottom5.index:
        row = bottom5.loc[ticker]
        lineas.append(
            f"   {ticker:6s}  Ret={row['Retorno_anual_%']:+7.2f}%  "
            f"Vol={row['Volatilidad_anual_%']:6.2f}%  "
            f"Sharpe={row['Sharpe_ratio']:+.3f}"
        )

    lineas.extend([
        "",
        "8. ARCHIVOS GENERADOS",
        f"   {ARCHIVO_PRECIOS_RAW}",
        f"   {ARCHIVO_PRECIOS_CLEAN}",
        f"   {ARCHIVO_RETORNOS}",
        f"   {ARCHIVO_RETORNOS_TRAIN}",
        f"   {ARCHIVO_RETORNOS_TEST}",
        f"   {ARCHIVO_COV_MATRIX}",
        f"   {ARCHIVO_MEDIA_RETORNOS}",
        f"   {ARCHIVO_STATS_DESCRIPTIVAS}",
        "",
        "=" * 70,
        "FIN DEL REPORTE BLOQUE 1",
        "=" * 70,
    ])

    reporte = "\n".join(lineas)

    with open(ARCHIVO_REPORTE_DATOS, "w", encoding="utf-8") as f:
        f.write(reporte)

    logger.info(f"\nReporte guardado en {ARCHIVO_REPORTE_DATOS}")

    return reporte


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================
def ejecutar_bloque1() -> dict:
    """
    Ejecuta el pipeline completo del Bloque 1.

    Returns
    -------
    dict
        Diccionario con todos los resultados intermedios:
        - precios_raw, precios_limpios
        - retornos, retornos_train, retornos_test
        - media_anual, cov_anual
        - stats, reporte
    """
    print("\n" + "=" * 70)
    print("BLOQUE 1 — PREPARACIÓN DEL MARCO DE DATOS")
    print("=" * 70 + "\n")

    # Paso 1: Descarga
    print("[1/6] Descargando precios de Yahoo Finance...")
    precios_raw = descargar_precios()

    # Paso 2: Limpieza
    print("[2/6] Limpiando y validando datos...")
    precios_limpios = limpiar_precios(precios_raw)

    # Paso 3: Retornos
    print("[3/6] Calculando retornos logarítmicos...")
    retornos = calcular_retornos(precios_limpios)

    # Paso 4: Split train/test
    print("[4/6] Separando train/test...")
    retornos_train, retornos_test = separar_train_test(retornos)

    # Paso 5: Parámetros del modelo
    print("[5/6] Calculando parámetros del modelo...")
    media_anual, cov_anual = calcular_parametros_modelo(retornos_train)

    # Paso 6: Estadísticas y reporte
    print("[6/6] Generando estadísticas y reporte...")
    stats = generar_estadisticas_descriptivas(
        precios_limpios, retornos, retornos_train, media_anual, cov_anual
    )
    reporte = generar_reporte(
        precios_raw, precios_limpios, retornos,
        retornos_train, retornos_test,
        media_anual, cov_anual, stats
    )

    print("\n" + reporte)
    print("\n[OK] Bloque 1 completado con exito.\n")

    return {
        "precios_raw": precios_raw,
        "precios_limpios": precios_limpios,
        "retornos": retornos,
        "retornos_train": retornos_train,
        "retornos_test": retornos_test,
        "media_anual": media_anual,
        "cov_anual": cov_anual,
        "stats": stats,
        "reporte": reporte,
    }
