"""
Bloque 3 -- QAOA en simulador.

Implementa QAOA (Quantum Approximate Optimization Algorithm) para
el problema de seleccion de carteras formulado como QUBO en el Bloque 2.

Arquitectura QAOA:
  1. Conversion QUBO -> Hamiltoniano Ising: H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
  2. Circuito QAOA con p capas:
     - Estado inicial: |+>^n
     - Capa l: exp(-i gamma_l H_C) * exp(-i beta_l H_M)
     - H_M = sum_i X_i (mixer estandar)
  3. Medicion y evaluacion del coste QUBO sobre las muestras.
  4. Optimizacion clasica de (gamma, beta) con COBYLA.

Conversion QUBO a Ising:
  x_i = (I - Z_i) / 2   (x_i=0 -> |0>, x_i=1 -> |1>)
  
  H_C = C + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
  donde:
    h_i = -Q_ii/2 - (1/4) sum_{j!=i} Q_ij_full
    J_ij = Q_ij / 4  (para i<j, con Q upper-triangular)
    C = sum_i Q_ii/2 + sum_{i<j} Q_ij/4  (offset constante)
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from src.config import (
    TICKERS,
    TAMANOS_INSTANCIA,
    RISK_FACTOR,
    PENALTY_FACTOR,
    N_SEMILLAS,
    SEMILLA_BASE,
    QAOA_P_VALUES,
    QAOA_MAX_ITER,
    QAOA_SHOTS,
    QAOA_OPTIMIZER,
    RISK_FREE_RATE,
    ARCHIVO_MEDIA_RETORNOS,
    ARCHIVO_COV_MATRIX,
    RESULTADOS,
    cardinalidad,
)
from src.bloque2_modelo import (
    seleccionar_subconjunto,
    construir_qubo,
    evaluar_cartera,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONVERSION QUBO -> ISING
# =============================================================================
def qubo_a_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convierte una matriz QUBO (upper-triangular) a coeficientes Ising.

    QUBO: f(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j
    Ising: H = C + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Matriz QUBO upper-triangular.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (h, J, offset) donde:
        - h: coeficientes lineales Z_i, shape (n,)
        - J: coeficientes cuadraticos Z_i Z_j, shape (n, n) upper-triangular
        - offset: constante C
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        # Offset from diagonal
        offset += Q[i, i] / 2.0

        # Linear coefficient h_i
        h[i] = -Q[i, i] / 2.0

        for j in range(i + 1, n):
            # Offset from off-diagonal
            offset += Q[i, j] / 4.0

            # Linear contributions from off-diagonal
            h[i] -= Q[i, j] / 4.0
            h[j] -= Q[i, j] / 4.0

            # Quadratic Ising coupling
            J[i, j] = Q[i, j] / 4.0

    return h, J, offset


# =============================================================================
# 2. CIRCUITO QAOA
# =============================================================================
def construir_circuito_qaoa(
    h: np.ndarray,
    J: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> QuantumCircuit:
    """
    Construye el circuito QAOA con p capas.

    Parameters
    ----------
    h : np.ndarray, shape (n,)
        Coeficientes lineales Ising.
    J : np.ndarray, shape (n, n)
        Coeficientes cuadraticos Ising (upper-triangular).
    gammas : np.ndarray, shape (p,)
        Parametros del operador de problema.
    betas : np.ndarray, shape (p,)
        Parametros del operador mixer.

    Returns
    -------
    QuantumCircuit
        Circuito QAOA listo para ejecucion.
    """
    n = len(h)
    p = len(gammas)
    qc = QuantumCircuit(n)

    # Estado inicial: |+>^n
    for i in range(n):
        qc.h(i)

    # p capas QAOA
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # --- Operador de problema: exp(-i * gamma * H_C) ---
        # Terminos Z_i Z_j (cuadraticos)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * gamma * J[i, j], i, j)

        # Terminos Z_i (lineales)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gamma * h[i], i)

        # --- Operador mixer: exp(-i * beta * H_M) ---
        for i in range(n):
            qc.rx(2 * beta, i)

    # Medicion
    qc.measure_all()

    return qc


# =============================================================================
# 3. EVALUACION DE COSTE DESDE COUNTS
# =============================================================================
def evaluar_counts_qubo(counts: dict, Q: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Calcula el valor esperado del coste QUBO a partir de los counts
    de medicion, y devuelve la mejor solucion encontrada.

    Parameters
    ----------
    counts : dict
        Diccionario {bitstring: count} del resultado de medicion.
    Q : np.ndarray
        Matriz QUBO.

    Returns
    -------
    tuple[float, np.ndarray]
        (valor_esperado, mejor_bitstring_como_array)
    """
    total_shots = sum(counts.values())
    valor_esperado = 0.0
    mejor_coste = float("inf")
    mejor_x = None

    for bitstring, count in counts.items():
        # Qiskit devuelve bitstrings en orden inverso (qubit 0 es el ultimo)
        x = np.array([int(b) for b in reversed(bitstring)], dtype=int)
        coste = float(x @ Q @ x)
        valor_esperado += coste * count / total_shots

        if coste < mejor_coste:
            mejor_coste = coste
            mejor_x = x.copy()

    return valor_esperado, mejor_x


# =============================================================================
# 4. EJECUCION QAOA COMPLETA
# =============================================================================
def ejecutar_qaoa(
    Q: np.ndarray,
    k: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    p: int = 1,
    shots: int = QAOA_SHOTS,
    max_iter: int = QAOA_MAX_ITER,
    seed: int = 42,
    noise_model: NoiseModel = None,
    etiqueta: str = "QAOA",
) -> dict:
    """
    Ejecuta QAOA completo: construye Ising, optimiza parametros, evalua.

    Parameters
    ----------
    Q : np.ndarray
        Matriz QUBO.
    k : int
        Cardinalidad objetivo.
    mu, sigma : np.ndarray
        Retornos y covarianzas para evaluacion financiera.
    p : int
        Numero de capas QAOA.
    shots : int
        Numero de mediciones por evaluacion.
    max_iter : int
        Iteraciones maximas del optimizador clasico.
    seed : int
        Semilla para reproducibilidad.
    noise_model : NoiseModel, optional
        Modelo de ruido. None = simulacion ideal.
    etiqueta : str
        Nombre descriptivo para el resultado.

    Returns
    -------
    dict
        Resultado completo: metricas financieras, parametros optimos,
        historial de optimizacion, informacion del circuito.
    """
    n = Q.shape[0]

    # Convertir a Ising
    h, J, offset = qubo_a_ising(Q)

    # Configurar simulador
    sim = AerSimulator(method="automatic")
    if noise_model is not None:
        sim = AerSimulator(noise_model=noise_model, method="automatic")

    # Contador de evaluaciones
    eval_count = [0]
    historial_coste = []

    def funcion_coste(params):
        """Funcion objetivo para el optimizador clasico."""
        gammas = params[:p]
        betas = params[p:]

        qc = construir_circuito_qaoa(h, J, gammas, betas)
        result = sim.run(qc, shots=shots, seed_simulator=seed).result()
        counts = result.get_counts()

        valor_esperado, _ = evaluar_counts_qubo(counts, Q)
        eval_count[0] += 1
        historial_coste.append(valor_esperado)

        return valor_esperado

    # Parametros iniciales aleatorios
    rng = np.random.default_rng(seed)
    params_init = rng.uniform(0, np.pi, size=2 * p)

    # Optimizacion
    t0 = time.time()

    resultado_opt = scipy_minimize(
        funcion_coste,
        params_init,
        method=QAOA_OPTIMIZER,
        options={"maxiter": max_iter},
    )

    # Evaluacion final con parametros optimos
    gammas_opt = resultado_opt.x[:p]
    betas_opt = resultado_opt.x[p:]
    qc_final = construir_circuito_qaoa(h, J, gammas_opt, betas_opt)
    result_final = sim.run(qc_final, shots=shots, seed_simulator=seed).result()
    counts_final = result_final.get_counts()
    _, mejor_x = evaluar_counts_qubo(counts_final, Q)

    tiempo = time.time() - t0

    # Evaluar como cartera
    ev = evaluar_cartera(mejor_x, mu, sigma, k, q=RISK_FACTOR)

    # Informacion del circuito
    qc_info = construir_circuito_qaoa(h, J, gammas_opt, betas_opt)
    depth = qc_info.depth()
    n_gates = qc_info.count_ops()

    return {
        "metodo": etiqueta,
        "p": p,
        "seed": seed,
        "retorno": ev["retorno"],
        "volatilidad": ev["volatilidad"],
        "sharpe": ev["sharpe"],
        "objetivo": ev["objetivo"],
        "factible": ev["factible"],
        "n_seleccionados": ev["n_seleccionados"],
        "tiempo": tiempo,
        "n_evaluaciones": eval_count[0],
        "coste_final_qubo": resultado_opt.fun,
        "gammas": gammas_opt.tolist(),
        "betas": betas_opt.tolist(),
        "historial_coste": historial_coste,
        "mejor_x": mejor_x.tolist(),
        "profundidad_circuito": depth,
        "n_qubits": n,
        "shots": shots,
        "con_ruido": noise_model is not None,
    }


# =============================================================================
# 5. OBTENER MODELO DE RUIDO
# =============================================================================
def obtener_noise_model() -> NoiseModel:
    """
    Obtiene un modelo de ruido realista basado en un backend fake de IBM.
    Usa FakeManilaV2 (5 qubits, ruido calibrado con datos reales).
    """
    backend = FakeManilaV2()
    noise_model = NoiseModel.from_backend(backend)
    logger.info(f"Noise model cargado: FakeManilaV2")
    return noise_model


# =============================================================================
# 6. BENCHMARK QAOA COMPLETO
# =============================================================================
def ejecutar_benchmark_qaoa(
    mu_full: pd.Series,
    sigma_full: pd.DataFrame,
    tamanos: list[int] = None,
    p_values: list[int] = QAOA_P_VALUES,
    n_semillas: int = N_SEMILLAS,
    semilla_base: int = SEMILLA_BASE,
    q: float = RISK_FACTOR,
) -> dict:
    """
    Ejecuta el benchmark QAOA para todos los tamanos, profundidades p,
    en simulador ideal y con ruido.
    """
    # Limitar tamanos a n<=5 para ruido (FakeManila tiene 5 qubits)
    # Para ideal, usamos todos los tamanos
    if tamanos is None:
        tamanos = TAMANOS_INSTANCIA

    noise_model = obtener_noise_model()

    resultados = {}

    for n in tamanos:
        k = cardinalidad(n)
        print(f"\n{'='*60}")
        print(f"QAOA — INSTANCIA n={n}, k={k}")
        print(f"{'='*60}")

        tickers_sub, mu_sub, sigma_sub = seleccionar_subconjunto(n, mu_full, sigma_full)
        Q, P, offset = construir_qubo(mu_sub, sigma_sub, k, q=q)

        print(f"Activos: {tickers_sub}")

        resultados_n = {"k": k, "tickers": tickers_sub}

        for p_val in p_values:
            # --- QAOA Ideal ---
            print(f"\n  [QAOA p={p_val} IDEAL] Ejecutando {n_semillas} semillas...")
            res_ideal = []
            for s in range(n_semillas):
                seed = semilla_base + s
                r = ejecutar_qaoa(
                    Q, k, mu_sub, sigma_sub,
                    p=p_val, seed=seed, noise_model=None,
                    etiqueta=f"QAOA_p{p_val}_ideal",
                )
                res_ideal.append(r)

                if s == 0:
                    print(f"    Circuito: {r['n_qubits']} qubits, "
                          f"profundidad={r['profundidad_circuito']}")

            fact_ideal = [r for r in res_ideal if r["factible"]]
            print(f"    Factibles: {len(fact_ideal)}/{n_semillas}")
            if fact_ideal:
                print(f"    Sharpe medio: {np.mean([r['sharpe'] for r in fact_ideal]):+.3f}")
                print(f"    Tiempo medio: {np.mean([r['tiempo'] for r in fact_ideal]):.2f}s")

            resultados_n[f"qaoa_p{p_val}_ideal"] = res_ideal

            # --- QAOA con Ruido (solo si n <= 5 qubits por FakeManila) ---
            if n <= 5:
                print(f"\n  [QAOA p={p_val} RUIDO] Ejecutando {n_semillas} semillas...")
                res_ruido = []
                for s in range(n_semillas):
                    seed = semilla_base + s
                    r = ejecutar_qaoa(
                        Q, k, mu_sub, sigma_sub,
                        p=p_val, seed=seed, noise_model=noise_model,
                        etiqueta=f"QAOA_p{p_val}_ruido",
                    )
                    res_ruido.append(r)

                fact_ruido = [r for r in res_ruido if r["factible"]]
                print(f"    Factibles: {len(fact_ruido)}/{n_semillas}")
                if fact_ruido:
                    print(f"    Sharpe medio: "
                          f"{np.mean([r['sharpe'] for r in fact_ruido]):+.3f}")

                resultados_n[f"qaoa_p{p_val}_ruido"] = res_ruido
            else:
                # Para n>5, simulamos ruido con un noise model generico simplificado
                print(f"\n  [QAOA p={p_val} RUIDO] n={n} > 5 qubits: "
                      f"usando modelo de ruido generico...")
                noise_gen = NoiseModel()
                from qiskit_aer.noise import depolarizing_error
                # Error de 1 qubit: 0.1%, 2 qubits: 1%
                noise_gen.add_all_qubit_quantum_error(
                    depolarizing_error(0.001, 1), ['rz', 'rx', 'h']
                )
                noise_gen.add_all_qubit_quantum_error(
                    depolarizing_error(0.01, 2), ['rzz']
                )

                res_ruido = []
                for s in range(n_semillas):
                    seed = semilla_base + s
                    r = ejecutar_qaoa(
                        Q, k, mu_sub, sigma_sub,
                        p=p_val, seed=seed, noise_model=noise_gen,
                        etiqueta=f"QAOA_p{p_val}_ruido",
                    )
                    res_ruido.append(r)

                fact_ruido = [r for r in res_ruido if r["factible"]]
                print(f"    Factibles: {len(fact_ruido)}/{n_semillas}")
                if fact_ruido:
                    print(f"    Sharpe medio: "
                          f"{np.mean([r['sharpe'] for r in fact_ruido]):+.3f}")

                resultados_n[f"qaoa_p{p_val}_ruido"] = res_ruido

        resultados[n] = resultados_n

    return resultados


# =============================================================================
# 7. GENERAR TABLA Y REPORTE BLOQUE 3
# =============================================================================
def generar_tabla_qaoa(resultados: dict, resultados_clasicos: dict = None) -> pd.DataFrame:
    """
    Genera tabla comparativa consolidada incluyendo resultados
    clasicos (si disponibles) y QAOA.
    """
    filas = []

    for n, res in resultados.items():
        k = res["k"]

        # Resultados clasicos (si se proporcionan)
        if resultados_clasicos and n in resultados_clasicos:
            rc = resultados_clasicos[n]
            ev_ex = rc["exacto"]["eval"]
            filas.append({
                "n": n, "k": k, "Metodo": "Exhaustiva",
                "Retorno_%": ev_ex["retorno"] * 100,
                "Vol_%": ev_ex["volatilidad"] * 100,
                "Sharpe": ev_ex["sharpe"],
                "Objetivo": ev_ex["objetivo"],
                "Gap_%": 0.0,
                "Factib": "1/1",
                "Tiempo_s": ev_ex["tiempo"],
                "Qubits": "-", "Prof": "-",
            })
            sa_f = rc["sa_factibles"]
            if sa_f:
                filas.append({
                    "n": n, "k": k, "Metodo": "SA (media)",
                    "Retorno_%": np.mean([r["retorno"] for r in sa_f]) * 100,
                    "Vol_%": np.mean([r["volatilidad"] for r in sa_f]) * 100,
                    "Sharpe": np.mean([r["sharpe"] for r in sa_f]),
                    "Objetivo": np.mean([r["objetivo"] for r in sa_f]),
                    "Gap_%": np.mean([r["gap_pct"] for r in sa_f]),
                    "Factib": f"{len(sa_f)}/{len(rc['sa'])}",
                    "Tiempo_s": np.mean([r["tiempo"] for r in sa_f]),
                    "Qubits": "-", "Prof": "-",
                })

        # Resultados QAOA
        obj_optimo = None
        if resultados_clasicos and n in resultados_clasicos:
            obj_optimo = resultados_clasicos[n]["obj_optimo"]

        for key in sorted(res.keys()):
            if not key.startswith("qaoa_"):
                continue

            lista = res[key]
            factibles = [r for r in lista if r["factible"]]

            if not factibles:
                etiqueta = key.replace("_", " ").upper()
                filas.append({
                    "n": n, "k": k, "Metodo": f"{etiqueta} (media)",
                    "Retorno_%": 0, "Vol_%": 0, "Sharpe": 0,
                    "Objetivo": 0, "Gap_%": float("inf"),
                    "Factib": f"0/{len(lista)}",
                    "Tiempo_s": np.mean([r["tiempo"] for r in lista]),
                    "Qubits": lista[0]["n_qubits"],
                    "Prof": lista[0]["profundidad_circuito"],
                })
                continue

            # Calcular gaps
            for r in factibles:
                if obj_optimo is not None and obj_optimo != 0:
                    r["gap_pct"] = abs(r["objetivo"] - obj_optimo) / abs(obj_optimo) * 100
                else:
                    r["gap_pct"] = 0.0

            parts = key.split("_")  # qaoa_p1_ideal
            p_val = parts[1]
            modo = parts[2]
            label = f"QAOA {p_val} {modo}"

            filas.append({
                "n": n, "k": k, "Metodo": f"{label} (media)",
                "Retorno_%": np.mean([r["retorno"] for r in factibles]) * 100,
                "Vol_%": np.mean([r["volatilidad"] for r in factibles]) * 100,
                "Sharpe": np.mean([r["sharpe"] for r in factibles]),
                "Objetivo": np.mean([r["objetivo"] for r in factibles]),
                "Gap_%": np.mean([r["gap_pct"] for r in factibles]),
                "Factib": f"{len(factibles)}/{len(lista)}",
                "Tiempo_s": np.mean([r["tiempo"] for r in factibles]),
                "Qubits": factibles[0]["n_qubits"],
                "Prof": factibles[0]["profundidad_circuito"],
            })

    return pd.DataFrame(filas)


def generar_reporte_bloque3(resultados: dict, tabla: pd.DataFrame) -> str:
    """Genera reporte textual del Bloque 3."""
    lineas = [
        "=" * 70,
        "REPORTE BLOQUE 3 -- QAOA EN SIMULADOR",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "CONFIGURACION QAOA",
        f"  Profundidades (p): {QAOA_P_VALUES}",
        f"  Optimizador clasico: {QAOA_OPTIMIZER}",
        f"  Max iteraciones: {QAOA_MAX_ITER}",
        f"  Shots por evaluacion: {QAOA_SHOTS}",
        f"  Semillas: {N_SEMILLAS}",
        "",
    ]

    for n, res in resultados.items():
        k = res["k"]
        lineas.extend([
            "-" * 50,
            f"INSTANCIA n={n}, k={k}",
            f"Activos: {', '.join(res['tickers'])}",
            "-" * 50,
        ])

        for key in sorted(res.keys()):
            if not key.startswith("qaoa_"):
                continue

            lista = res[key]
            factibles = [r for r in lista if r["factible"]]
            label = key.replace("_", " ").upper()

            lineas.append(f"\n  {label}:")
            lineas.append(f"    Factibles: {len(factibles)}/{len(lista)}")

            if factibles:
                lineas.extend([
                    f"    Sharpe medio:  {np.mean([r['sharpe'] for r in factibles]):+.3f} "
                    f"(std={np.std([r['sharpe'] for r in factibles]):.3f})",
                    f"    Objetivo medio: {np.mean([r['objetivo'] for r in factibles]):.6f}",
                    f"    Tiempo medio:  {np.mean([r['tiempo'] for r in factibles]):.2f}s",
                    f"    Evals medio:   {np.mean([r['n_evaluaciones'] for r in factibles]):.0f}",
                    f"    Circuito: {factibles[0]['n_qubits']} qubits, "
                    f"prof={factibles[0]['profundidad_circuito']}",
                ])

        lineas.append("")

    lineas.extend([
        "=" * 70,
        "TABLA COMPARATIVA COMPLETA (clasicos + QAOA)",
        "=" * 70,
        "",
        tabla.to_string(index=False),
        "",
        "=" * 70,
        "FIN DEL REPORTE BLOQUE 3",
        "=" * 70,
    ])

    reporte = "\n".join(lineas)
    ruta = RESULTADOS / "reporte_bloque3.txt"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(reporte)
    logger.info(f"Reporte guardado en {ruta}")
    return reporte


# =============================================================================
# PIPELINE COMPLETO BLOQUE 3
# =============================================================================
def ejecutar_bloque3() -> dict:
    """
    Ejecuta el pipeline completo del Bloque 3.
    """
    print("\n" + "=" * 70)
    print("BLOQUE 3 -- QAOA EN SIMULADOR")
    print("=" * 70)

    # Cargar datos
    print("\n[0] Cargando datos...")
    mu_full = pd.read_csv(ARCHIVO_MEDIA_RETORNOS, index_col=0).squeeze()
    sigma_full = pd.read_csv(ARCHIVO_COV_MATRIX, index_col=0)

    # Cargar resultados clasicos del Bloque 2
    from src.bloque2_modelo import ejecutar_benchmark_clasico
    print("[0.1] Ejecutando baselines clasicos (referencia)...")
    resultados_clasicos = ejecutar_benchmark_clasico(mu_full, sigma_full)

    # Ejecutar QAOA
    print("\n[1] Ejecutando benchmark QAOA...")
    resultados_qaoa = ejecutar_benchmark_qaoa(mu_full, sigma_full)

    # Generar tabla y reporte
    print(f"\n{'='*60}")
    print("[2] Generando tabla comparativa y reporte...")
    tabla = generar_tabla_qaoa(resultados_qaoa, resultados_clasicos)
    tabla.to_csv(RESULTADOS / "resultados_qaoa.csv", index=False)

    reporte = generar_reporte_bloque3(resultados_qaoa, tabla)
    print("\n" + reporte)
    print("\n[OK] Bloque 3 completado con exito.\n")

    return {
        "clasicos": resultados_clasicos,
        "qaoa": resultados_qaoa,
        "tabla": tabla,
        "reporte": reporte,
    }
