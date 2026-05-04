"""
Bloque 2 — Modelo media-varianza y baselines clasicos.

Implementa:
  1. Formulacion del problema de seleccion de carteras con cardinalidad.
  2. Transformacion a QUBO (Quadratic Unconstrained Binary Optimization).
  3. Baseline exacto: busqueda exhaustiva.
  4. Baseline heuristico: Simulated Annealing sobre QUBO.
  5. Evaluacion comparativa de resultados.

Formulacion QUBO:
  Variables: x_i in {0,1} para i=1..n (1 si el activo i esta en cartera)
  Pesos iguales: w_i = x_i / k (k = cardinalidad)
  
  Objetivo (minimizar):
    f(x) = q/k^2 * x^T Sigma x  -  (1-q)/k * mu^T x
           |--- riesgo ---|        |--- retorno ---|
  
  Restriccion: sum(x_i) = k  (exactamente k activos)
  
  QUBO (con penalizacion):
    h(x) = f(x) + P * (sum(x_i) - k)^2
  
  Matriz QUBO Q (upper triangular):
    Q_ii     = q*Sigma_ii/k^2 - (1-q)*mu_i/k + P*(1 - 2k)
    Q_ij i<j = 2*q*Sigma_ij/k^2 + 2*P
"""

import logging
import time
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

from src.config import (
    TICKERS,
    TAMANOS_INSTANCIA,
    RISK_FREE_RATE,
    RISK_FACTOR,
    PENALTY_FACTOR,
    SA_N_ITER,
    SA_T_INIT,
    SA_ALPHA,
    N_SEMILLAS,
    SEMILLA_BASE,
    ARCHIVO_MEDIA_RETORNOS,
    ARCHIVO_COV_MATRIX,
    ARCHIVO_RESULTADOS_CLASICOS,
    ARCHIVO_REPORTE_BLOQUE2,
    cardinalidad,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. SELECCION DE SUBCONJUNTO DE ACTIVOS
# =============================================================================
def seleccionar_subconjunto(
    n: int,
    mu: pd.Series,
    sigma: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Selecciona los primeros n activos del universo.

    Parameters
    ----------
    n : int
        Numero de activos a seleccionar.
    mu : pd.Series
        Retornos medios anualizados de todos los activos.
    sigma : pd.DataFrame
        Matriz de covarianzas anualizada completa.

    Returns
    -------
    tuple[list[str], np.ndarray, np.ndarray]
        (tickers_sub, mu_sub, sigma_sub) con arrays numpy.
    """
    tickers_sub = list(mu.index[:n])
    mu_sub = mu.iloc[:n].values
    sigma_sub = sigma.iloc[:n, :n].values
    return tickers_sub, mu_sub, sigma_sub


# =============================================================================
# 2. CONSTRUCCION DE LA MATRIZ QUBO
# =============================================================================
def construir_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    q: float = RISK_FACTOR,
    penalty_factor: float = PENALTY_FACTOR,
) -> tuple[np.ndarray, float, float]:
    """
    Construye la matriz QUBO para el problema de seleccion de carteras.

    La funcion objetivo QUBO es:
      h(x) = x^T Q x  (forma upper-triangular, con diagonal incluyendo
                        terminos lineales ya que x_i^2 = x_i)

    Parameters
    ----------
    mu : np.ndarray, shape (n,)
        Retornos medios anualizados.
    sigma : np.ndarray, shape (n, n)
        Matriz de covarianzas anualizada.
    k : int
        Cardinalidad (numero de activos a seleccionar).
    q : float
        Factor de riesgo en [0,1]. q=0: maximiza retorno, q=1: minimiza riesgo.
    penalty_factor : float
        Multiplicador para calcular P = penalty_factor * escala_objetivo.

    Returns
    -------
    tuple[np.ndarray, float, float]
        (Q, P, offset) donde:
        - Q: matriz QUBO (n x n, upper triangular)
        - P: penalizacion utilizada
        - offset: constante P*k^2 (para reconstruir energia real)
    """
    n = len(mu)

    # --- Construir QUBO sin penalizacion para estimar escala ---
    Q_obj = np.zeros((n, n))
    for i in range(n):
        # Diagonal: terminos cuadraticos + lineales
        Q_obj[i, i] = q * sigma[i, i] / k**2 - (1 - q) * mu[i] / k
        for j in range(i + 1, n):
            # Off-diagonal (upper triangular)
            Q_obj[i, j] = 2 * q * sigma[i, j] / k**2

    # --- Calcular penalizacion adaptativa ---
    escala = np.max(np.abs(Q_obj))
    if escala == 0:
        escala = 1.0
    P = penalty_factor * escala

    # --- Anadir penalizacion a la matriz QUBO ---
    Q = Q_obj.copy()
    for i in range(n):
        Q[i, i] += P * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] += 2 * P

    offset = P * k**2

    logger.info(f"QUBO construida: n={n}, k={k}, q={q:.2f}, P={P:.6f}")
    logger.info(f"  Escala objetivo: {escala:.6f}, Offset: {offset:.6f}")

    return Q, P, offset


# =============================================================================
# 3. EVALUACION DE SOLUCIONES
# =============================================================================
def evaluar_qubo(x: np.ndarray, Q: np.ndarray) -> float:
    """Calcula la energia QUBO: h(x) = x^T Q x."""
    return float(x @ Q @ x)


def evaluar_cartera(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    rf: float = RISK_FREE_RATE,
    q: float = RISK_FACTOR,
) -> dict:
    """
    Evalua una solucion como cartera financiera.

    Parameters
    ----------
    x : np.ndarray
        Vector binario de seleccion.
    mu, sigma : np.ndarray
        Retornos y covarianzas del subconjunto.
    k : int
        Cardinalidad objetivo.
    rf : float
        Tasa libre de riesgo.
    q : float
        Factor de riesgo (para calcular objetivo).

    Returns
    -------
    dict
        Metricas: retorno, volatilidad, sharpe, objetivo, factible, n_seleccionados.
    """
    n_sel = int(np.sum(x))
    factible = (n_sel == k)

    if n_sel == 0:
        return {
            "retorno": 0.0, "volatilidad": 0.0, "sharpe": 0.0,
            "objetivo": float("inf"), "factible": False, "n_seleccionados": 0,
        }

    # Pesos iguales entre activos seleccionados
    w = x / n_sel
    retorno = float(w @ mu)
    varianza = float(w @ sigma @ w)
    volatilidad = float(np.sqrt(varianza))
    sharpe = (retorno - rf) / volatilidad if volatilidad > 0 else 0.0

    # Objetivo original (sin penalizacion)
    objetivo = q * varianza - (1 - q) * retorno

    return {
        "retorno": retorno,
        "volatilidad": volatilidad,
        "sharpe": sharpe,
        "objetivo": objetivo,
        "factible": factible,
        "n_seleccionados": n_sel,
    }


# =============================================================================
# 4. BUSQUEDA EXHAUSTIVA (BASELINE EXACTO)
# =============================================================================
def busqueda_exhaustiva(
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    q: float = RISK_FACTOR,
) -> tuple[np.ndarray, dict, list[dict]]:
    """
    Busqueda exhaustiva sobre todas las combinaciones C(n, k).
    Garantiza encontrar el optimo global.

    Parameters
    ----------
    mu : np.ndarray, shape (n,)
    sigma : np.ndarray, shape (n, n)
    k : int
    q : float

    Returns
    -------
    tuple[np.ndarray, dict, list[dict]]
        (mejor_x, mejor_eval, todas_evaluaciones)
    """
    n = len(mu)
    n_combinaciones = 1
    for i in range(k):
        n_combinaciones = n_combinaciones * (n - i) // (i + 1)

    logger.info(f"Busqueda exhaustiva: C({n},{k}) = {n_combinaciones} combinaciones")

    mejor_obj = float("inf")
    mejor_x = None
    mejor_eval = None
    todas = []

    t0 = time.time()

    for combo in combinations(range(n), k):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1

        ev = evaluar_cartera(x, mu, sigma, k, q=q)
        todas.append({"x": x.copy(), **ev})

        if ev["objetivo"] < mejor_obj:
            mejor_obj = ev["objetivo"]
            mejor_x = x.copy()
            mejor_eval = ev.copy()

    tiempo = time.time() - t0
    mejor_eval["tiempo"] = tiempo
    mejor_eval["metodo"] = "Exhaustiva"

    logger.info(f"  Optimo encontrado en {tiempo:.4f}s")
    logger.info(f"  Ret={mejor_eval['retorno']*100:+.2f}%  "
                f"Vol={mejor_eval['volatilidad']*100:.2f}%  "
                f"Sharpe={mejor_eval['sharpe']:+.3f}")

    return mejor_x, mejor_eval, todas


# =============================================================================
# 5. SIMULATED ANNEALING SOBRE QUBO
# =============================================================================
def simulated_annealing(
    Q: np.ndarray,
    k: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    seed: int = 42,
    n_iter: int = SA_N_ITER,
    T_init: float = SA_T_INIT,
    alpha: float = SA_ALPHA,
    q: float = RISK_FACTOR,
) -> tuple[np.ndarray, dict]:
    """
    Simulated Annealing sobre la formulacion QUBO.

    Usa bit-flip como movimiento vecino. La penalizacion en Q
    desalienta soluciones infactibles.

    Parameters
    ----------
    Q : np.ndarray
        Matriz QUBO.
    k : int
        Cardinalidad objetivo.
    mu, sigma : np.ndarray
        Para evaluacion financiera del resultado.
    seed : int
        Semilla para reproducibilidad.
    n_iter, T_init, alpha : float
        Parametros de SA.

    Returns
    -------
    tuple[np.ndarray, dict]
        (mejor_x, evaluacion) con metricas financieras y tiempo.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    # Inicializar con solucion factible aleatoria (k bits a 1)
    x = np.zeros(n, dtype=int)
    indices = rng.choice(n, size=k, replace=False)
    x[indices] = 1

    energia = evaluar_qubo(x, Q)
    mejor_x = x.copy()
    mejor_energia = energia

    T = T_init
    t0 = time.time()

    for it in range(n_iter):
        # Movimiento vecino: flip un bit aleatorio
        bit = rng.integers(0, n)
        x_nuevo = x.copy()
        x_nuevo[bit] = 1 - x_nuevo[bit]

        energia_nueva = evaluar_qubo(x_nuevo, Q)
        delta = energia_nueva - energia

        # Criterio de aceptacion Metropolis
        if delta < 0 or rng.random() < np.exp(-delta / T):
            x = x_nuevo
            energia = energia_nueva

            if energia < mejor_energia:
                mejor_energia = energia
                mejor_x = x.copy()

        # Enfriar
        T *= alpha

    tiempo = time.time() - t0

    # Evaluar la mejor solucion como cartera
    ev = evaluar_cartera(mejor_x, mu, sigma, k, q=q)
    ev["tiempo"] = tiempo
    ev["energia_qubo"] = mejor_energia
    ev["metodo"] = f"SA(seed={seed})"
    ev["seed"] = seed

    return mejor_x, ev


# =============================================================================
# 6. EJECUCION DEL BENCHMARK CLASICO COMPLETO
# =============================================================================
def ejecutar_benchmark_clasico(
    mu_full: pd.Series,
    sigma_full: pd.DataFrame,
    tamanos: list[int] = TAMANOS_INSTANCIA,
    n_semillas: int = N_SEMILLAS,
    semilla_base: int = SEMILLA_BASE,
    q: float = RISK_FACTOR,
) -> dict:
    """
    Ejecuta el benchmark clasico completo para todos los tamanos de instancia.

    Para cada tamano n:
      1. Selecciona subconjunto de activos.
      2. Construye QUBO.
      3. Ejecuta busqueda exhaustiva (optimo exacto).
      4. Ejecuta SA con multiples semillas.
      5. Calcula gaps relativos.

    Returns
    -------
    dict
        Resultados organizados por tamano de instancia.
    """
    resultados = {}

    for n in tamanos:
        k = cardinalidad(n)
        print(f"\n{'='*60}")
        print(f"INSTANCIA n={n}, k={k}  (C({n},{k}) combinaciones)")
        print(f"{'='*60}")

        # Seleccionar subconjunto
        tickers_sub, mu_sub, sigma_sub = seleccionar_subconjunto(n, mu_full, sigma_full)
        print(f"Activos: {tickers_sub}")

        # Construir QUBO
        Q, P, offset = construir_qubo(mu_sub, sigma_sub, k, q=q)

        # --- Baseline exacto ---
        print(f"\n[1/2] Busqueda exhaustiva...")
        mejor_x_exact, eval_exact, todas = busqueda_exhaustiva(
            mu_sub, sigma_sub, k, q=q
        )
        obj_optimo = eval_exact["objetivo"]

        print(f"  Optimo: Ret={eval_exact['retorno']*100:+.2f}%  "
              f"Vol={eval_exact['volatilidad']*100:.2f}%  "
              f"Sharpe={eval_exact['sharpe']:+.3f}  "
              f"Obj={obj_optimo:.6f}  "
              f"Tiempo={eval_exact['tiempo']:.4f}s")

        # Activos seleccionados en el optimo
        seleccionados = [tickers_sub[i] for i in range(n) if mejor_x_exact[i] == 1]
        print(f"  Cartera optima: {seleccionados}")

        # --- Baseline heuristico: SA ---
        print(f"\n[2/2] Simulated Annealing ({n_semillas} semillas)...")
        resultados_sa = []

        for s in range(n_semillas):
            seed = semilla_base + s
            _, eval_sa = simulated_annealing(
                Q, k, mu_sub, sigma_sub, seed=seed, q=q
            )

            # Calcular gap relativo al optimo
            if eval_sa["factible"] and obj_optimo != 0:
                gap = abs(eval_sa["objetivo"] - obj_optimo) / abs(obj_optimo) * 100
            elif eval_sa["factible"] and obj_optimo == 0:
                gap = abs(eval_sa["objetivo"]) * 100
            else:
                gap = float("inf")

            eval_sa["gap_pct"] = gap
            resultados_sa.append(eval_sa)

        # Resumen SA
        sa_factibles = [r for r in resultados_sa if r["factible"]]
        n_factibles = len(sa_factibles)

        print(f"  Factibles: {n_factibles}/{n_semillas}")
        if sa_factibles:
            gaps = [r["gap_pct"] for r in sa_factibles]
            sharpes = [r["sharpe"] for r in sa_factibles]
            tiempos = [r["tiempo"] for r in sa_factibles]
            print(f"  Gap medio: {np.mean(gaps):.2f}%  (std={np.std(gaps):.2f}%)")
            print(f"  Sharpe medio: {np.mean(sharpes):+.3f}  (std={np.std(sharpes):.3f})")
            print(f"  Tiempo medio: {np.mean(tiempos):.4f}s")

            # Mejor SA
            mejor_sa = min(sa_factibles, key=lambda r: r["objetivo"])
            print(f"  Mejor SA: Ret={mejor_sa['retorno']*100:+.2f}%  "
                  f"Vol={mejor_sa['volatilidad']*100:.2f}%  "
                  f"Sharpe={mejor_sa['sharpe']:+.3f}  "
                  f"Gap={mejor_sa['gap_pct']:.2f}%")

        # Guardar resultados de esta instancia
        resultados[n] = {
            "k": k,
            "tickers": tickers_sub,
            "Q": Q,
            "P": P,
            "offset": offset,
            "exacto": {"x": mejor_x_exact, "eval": eval_exact, "todas": todas},
            "sa": resultados_sa,
            "sa_factibles": sa_factibles,
            "obj_optimo": obj_optimo,
        }

    return resultados


# =============================================================================
# 7. GENERAR TABLA COMPARATIVA Y REPORTE
# =============================================================================
def generar_tabla_comparativa(resultados: dict) -> pd.DataFrame:
    """
    Genera un DataFrame consolidado con los resultados de todos
    los metodos y tamanos de instancia.
    """
    filas = []

    for n, res in resultados.items():
        k = res["k"]
        ev = res["exacto"]["eval"]

        # Fila del exacto
        filas.append({
            "n": n, "k": k, "Metodo": "Exhaustiva",
            "Retorno_%": ev["retorno"] * 100,
            "Volatilidad_%": ev["volatilidad"] * 100,
            "Sharpe": ev["sharpe"],
            "Objetivo": ev["objetivo"],
            "Gap_%": 0.0,
            "Factible": True,
            "Tiempo_s": ev["tiempo"],
        })

        # Filas de SA (media y std de factibles)
        sa_f = res["sa_factibles"]
        if sa_f:
            filas.append({
                "n": n, "k": k, "Metodo": "SA (media)",
                "Retorno_%": np.mean([r["retorno"] for r in sa_f]) * 100,
                "Volatilidad_%": np.mean([r["volatilidad"] for r in sa_f]) * 100,
                "Sharpe": np.mean([r["sharpe"] for r in sa_f]),
                "Objetivo": np.mean([r["objetivo"] for r in sa_f]),
                "Gap_%": np.mean([r["gap_pct"] for r in sa_f]),
                "Factible": True,
                "Tiempo_s": np.mean([r["tiempo"] for r in sa_f]),
            })
            filas.append({
                "n": n, "k": k, "Metodo": "SA (std)",
                "Retorno_%": np.std([r["retorno"] for r in sa_f]) * 100,
                "Volatilidad_%": np.std([r["volatilidad"] for r in sa_f]) * 100,
                "Sharpe": np.std([r["sharpe"] for r in sa_f]),
                "Objetivo": np.std([r["objetivo"] for r in sa_f]),
                "Gap_%": np.std([r["gap_pct"] for r in sa_f]),
                "Factible": True,
                "Tiempo_s": np.std([r["tiempo"] for r in sa_f]),
            })
            # Mejor SA
            mejor_sa = min(sa_f, key=lambda r: r["objetivo"])
            filas.append({
                "n": n, "k": k, "Metodo": "SA (mejor)",
                "Retorno_%": mejor_sa["retorno"] * 100,
                "Volatilidad_%": mejor_sa["volatilidad"] * 100,
                "Sharpe": mejor_sa["sharpe"],
                "Objetivo": mejor_sa["objetivo"],
                "Gap_%": mejor_sa["gap_pct"],
                "Factible": True,
                "Tiempo_s": mejor_sa["tiempo"],
            })

        # Tasa de factibilidad SA
        n_fact = len(sa_f)
        n_total = len(res["sa"])
        filas.append({
            "n": n, "k": k, "Metodo": "SA (factibilidad)",
            "Retorno_%": 0, "Volatilidad_%": 0, "Sharpe": 0,
            "Objetivo": 0, "Gap_%": 0,
            "Factible": f"{n_fact}/{n_total}",
            "Tiempo_s": 0,
        })

    df = pd.DataFrame(filas)
    return df


def generar_reporte_bloque2(resultados: dict, tabla: pd.DataFrame) -> str:
    """Genera reporte textual del Bloque 2."""
    lineas = [
        "=" * 70,
        "REPORTE BLOQUE 2 -- MODELO Y BASELINES CLASICOS",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "FORMULACION",
        f"  Modelo: media-varianza discretizado con cardinalidad fija",
        f"  Transformacion: QUBO (Quadratic Unconstrained Binary Optimization)",
        f"  Factor de riesgo (q): {RISK_FACTOR}",
        f"  Factor de penalizacion: {PENALTY_FACTOR}x escala",
        "",
    ]

    for n, res in resultados.items():
        k = res["k"]
        ev_ex = res["exacto"]["eval"]
        sa_f = res["sa_factibles"]

        lineas.extend([
            "-" * 50,
            f"INSTANCIA n={n}, k={k}",
            "-" * 50,
            f"  Activos: {', '.join(res['tickers'])}",
            "",
            f"  BUSQUEDA EXHAUSTIVA (optimo global):",
            f"    Retorno anual:  {ev_ex['retorno']*100:+.2f}%",
            f"    Volatilidad:    {ev_ex['volatilidad']*100:.2f}%",
            f"    Sharpe ratio:   {ev_ex['sharpe']:+.3f}",
            f"    Objetivo:       {ev_ex['objetivo']:.6f}",
            f"    Tiempo:         {ev_ex['tiempo']:.4f}s",
            f"    Cartera: {[res['tickers'][i] for i in range(n) if res['exacto']['x'][i]==1]}",
            "",
        ])

        if sa_f:
            gaps = [r["gap_pct"] for r in sa_f]
            lineas.extend([
                f"  SIMULATED ANNEALING ({len(res['sa'])} semillas):",
                f"    Factibles:      {len(sa_f)}/{len(res['sa'])}",
                f"    Gap medio:      {np.mean(gaps):.2f}% (std={np.std(gaps):.2f}%)",
                f"    Sharpe medio:   {np.mean([r['sharpe'] for r in sa_f]):+.3f}",
                f"    Tiempo medio:   {np.mean([r['tiempo'] for r in sa_f]):.4f}s",
            ])
            mejor_sa = min(sa_f, key=lambda r: r["objetivo"])
            lineas.extend([
                f"    Mejor SA:       Obj={mejor_sa['objetivo']:.6f}  "
                f"Gap={mejor_sa['gap_pct']:.2f}%  Sharpe={mejor_sa['sharpe']:+.3f}",
            ])
        lineas.append("")

    lineas.extend([
        "=" * 70,
        "TABLA COMPARATIVA CONSOLIDADA",
        "=" * 70,
        "",
        tabla.to_string(index=False),
        "",
        "=" * 70,
        "FIN DEL REPORTE BLOQUE 2",
        "=" * 70,
    ])

    reporte = "\n".join(lineas)

    with open(ARCHIVO_REPORTE_BLOQUE2, "w", encoding="utf-8") as f:
        f.write(reporte)

    logger.info(f"Reporte guardado en {ARCHIVO_REPORTE_BLOQUE2}")
    return reporte


# =============================================================================
# PIPELINE COMPLETO BLOQUE 2
# =============================================================================
def ejecutar_bloque2() -> dict:
    """
    Ejecuta el pipeline completo del Bloque 2.

    Carga datos del Bloque 1, ejecuta benchmark clasico, genera tablas y reporte.
    """
    print("\n" + "=" * 70)
    print("BLOQUE 2 -- MODELO Y BASELINES CLASICOS")
    print("=" * 70)

    # Cargar datos del Bloque 1
    print("\n[0] Cargando datos del Bloque 1...")
    mu_full = pd.read_csv(ARCHIVO_MEDIA_RETORNOS, index_col=0).squeeze()
    sigma_full = pd.read_csv(ARCHIVO_COV_MATRIX, index_col=0)
    print(f"  Cargados: {len(mu_full)} activos, covarianza {sigma_full.shape}")

    # Ejecutar benchmark
    print("\n[1] Ejecutando benchmark clasico...")
    resultados = ejecutar_benchmark_clasico(mu_full, sigma_full)

    # Generar tabla y reporte
    print(f"\n{'='*60}")
    print("[2] Generando tabla comparativa y reporte...")
    tabla = generar_tabla_comparativa(resultados)
    tabla.to_csv(ARCHIVO_RESULTADOS_CLASICOS, index=False)
    print(f"  Tabla guardada en {ARCHIVO_RESULTADOS_CLASICOS}")

    reporte = generar_reporte_bloque2(resultados, tabla)
    print("\n" + reporte)
    print("\n[OK] Bloque 2 completado con exito.\n")

    resultados["tabla"] = tabla
    resultados["reporte"] = reporte
    return resultados
