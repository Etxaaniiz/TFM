
# ESPECIFICACIÓN TÉCNICA FINAL v1.0
## TFM: Optimización de Carteras mediante Computación Cuántica Híbrida

## 1. Contexto General del Proyecto

Descripción del TFM

Este proyecto de fin de máster tiene como objetivo estudiar la viabilidad práctica de la computación cuántica híbrida aplicada a la optimización de carteras financieras bajo restricciones reales.

El problema abordado consiste en seleccionar una cartera de inversión formada por un subconjunto fijo de activos, utilizando pesos equiponderados y restricciones de cardinalidad, de manera que se optimice el equilibrio entre rentabilidad esperada y riesgo.

Se realizará un estudio comparativo entre métodos clásicos de optimización y enfoques cuánticos híbridos basados en la formulación del problema como un problema de optimización binaria cuadrática (QUBO).

El propósito principal del trabajo NO es demostrar una ventaja cuántica frente a métodos clásicos exactos, sino evaluar rigurosamente:

La calidad de las soluciones obtenidas.
La factibilidad de las soluciones generadas.
El coste computacional requerido.
La escalabilidad de cada método.
La utilidad práctica de mixers restringidos frente a QAOA estándar.
El impacto de diferentes profundidades de circuito cuántico.
El posible beneficio del uso de JaspQAOA para acelerar la simulación.
Objetivos específicos
Implementar un modelo clásico de optimización de carteras basado en Markowitz con restricciones de cardinalidad y pesos equiponderados.
Formular el problema como un modelo QUBO y transformarlo a un Hamiltoniano de Ising compatible con algoritmos cuánticos.
Implementar y evaluar solucionadores clásicos de referencia:
Gurobi (solución exacta).
ExactSolver de dimod.
Simulated Annealing.
Implementar y evaluar solucionadores cuánticos híbridos:
QAOA estándar.
QAOA con Constrained XY Mixer.
JaspQAOA.
Comparar todos los métodos utilizando métricas homogéneas:
Valor de la función objetivo.
GAP respecto a Gurobi.
Ratio de Sharpe.
Rentabilidad esperada.
Volatilidad.
Factibilidad de las soluciones.
Tiempo de ejecución.
Consumo de recursos computacionales.
Analizar cómo evoluciona el rendimiento al aumentar:
El número de activos considerados (N).
La cardinalidad de la cartera (K).
La profundidad del circuito QAOA (p).
Generar automáticamente todos los resultados necesarios para la elaboración de la memoria final del TFM.
Restricciones de implementación

Este proyecto debe desarrollarse siguiendo los siguientes principios:

Todo el código debe ser completamente reproducible.
Todos los experimentos deben poder ejecutarse mediante archivos de configuración.
Los resultados deben almacenarse automáticamente en formatos estructurados.
El desarrollo inicial se realizará localmente.
Los experimentos computacionalmente se ejecutarán posteriormente en Google Colab Pro mediante un simple git clone del repositorio.
Toda la arquitectura del proyecto debe diseñarse para facilitar la generación automática de figuras, tablas y material reutilizable para la memoria final.
Filosofía de desarrollo

Este documento constituye la especificación técnica oficial del proyecto.

Cualquier implementación realizada a partir de esta especificación debe seguir estrictamente las decisiones aquí definidas.

El objetivo es que cualquier desarrollador o sistema de inteligencia artificial pueda implementar, ejecutar y reproducir completamente el proyecto sin necesidad de información adicional ni decisiones de diseño no documentadas.

Si durante el desarrollo surge una situación no contemplada explícitamente en esta especificación, deberá priorizarse:

La reproducibilidad experimental.
La consistencia metodológica.
La simplicidad de implementación.
La generación de resultados útiles para el análisis científico del TFM.

---

## 2. Preguntas de investigación

RQ1. ¿XY-QAOA mejora la factibilidad respecto a QAOA estándar?

RQ2. ¿Cómo afecta la profundidad p al rendimiento?

RQ3. ¿Cómo escala cada método con N?

RQ4. ¿JaspQAOA aporta mejoras prácticas?

RQ5. ¿Existe alguna ventaja práctica frente a Gurobi?

---

## 3. Formulación matemática

Variables binarias:

x_i ∈ {0,1}

Restricción:

Σ x_i = K

Pesos equiponderados:

w_i = x_i / K

Función objetivo:

min λ·wᵀΣw − (1−λ)·μᵀw

con λ = 0.5 inicialmente.

Penalización cardinalidad:

P(Σx_i − K)^2

Penalización inicial:

P = 10 × max(|Q_ij|)

---

## 4. Datos financieros

Fuente: Yahoo Finance

Periodo:

2019-01-01 → 2024-12-31

Frecuencia:

Diaria.

Índices:

- S&P500
- NASDAQ100
- IBEX35

Proceso:

1. Descargar Adjusted Close.
2. Eliminar activos con NaN.
3. Retornos logarítmicos.
4. Rentabilidades anualizadas.
5. Covarianzas anualizadas.

---

## 5. Diseño experimental

Instancias validación:

N=[6,8,10]
K=[2,2,3]

Instancias principales:

N=[12,14,16]
K=[3,4,4]

Escalabilidad:

N=[18,20]
K=round(N/4)

Generar 5 subconjuntos aleatorios por cada N.

Total instancias: 40.

Semillas globales:

42-51.

---

## 6. Solvers

1. Gurobi
2. ExactSolver
3. Simulated Annealing
4. QAOA
5. XY-QAOA
6. JaspQAOA

---

## 7. Estructura repositorio

TFM/

data/raw/
data/processed/
data/instances/

src/data/
src/portfolio/
src/solvers/
src/quantum/
src/experiments/
src/metrics/
src/utils/

scripts/
configs/
results/
figures/
tables/
tests/
docs/

---

## 8. Contratos de funciones

download_data(tickers,start,end)
→ DataFrame precios

compute_returns(prices)
→ DataFrame retornos

compute_statistics(returns)
→ mu, Sigma

build_qubo(mu,Sigma,K,lambda,penalty)
→ Q

qubo_to_ising(Q)
→ h,J,offset

solve_gurobi(instance)
→ dict resultados

solve_qaoa(instance,config)
→ dict resultados

Todos los solvers DEVUELVEN el mismo formato.

---

## 9. Formato estándar resultados

dataset
solver
N
K
instance_id
seed
p
objective
energy
gap
sharpe
expected_return
volatility
feasible
runtime_seconds
memory_mb

---

## 10. Configuración YAML

solver: qaoa

problem:
  N: 12
  K: 3
  lambda: 0.5

algorithm:
  p: 2
  optimizer: COBYLA
  maxiter: 100
  shots: 1024

execution:
  seed: 42

---

## 11. Validaciones obligatorias

Markowitz:
comparación manual.

QUBO:
error < 1e-6.

Ising:
equivalencia exacta.

Gurobi:
restricción cardinalidad cumplida.

QAOA:
bitstrings válidos.

---

## 12. Parámetros experimentales

SA:
num_reads=1000
num_sweeps=1000

QAOA:
p=[1,2,3,4]
maxiter=100
shots=1024

10 semillas.

Abortar >30 min.

JASP:
N=[10,12,14,16]
p=[2,3,4]

---

## 13. Estrategia Local/Colab

LOCAL:
Desarrollo.
Unit tests.
N<=12.

COLAB PRO:
Experimentos finales.
N>=14.
Jasp.
Escalabilidad.

Flujo:

git clone

pip install -r requirements.txt

python scripts/run_experiment.py --config ...

Guardar automáticamente en Drive.

---

## 14. Scripts

prepare_data.py

generate_instances.py

run_gurobi.py

run_exact.py

run_sa.py

run_qaoa.py

run_xy.py

run_jasp.py

generate_figures.py

generate_tables.py

---

## 15. Outputs esperados

CSV resultados.

PNG figuras.

TEX tablas.

JSON configuración ejecutada.

Log ejecución.

---

## 16. Figuras memoria

Tiempo vs N

Gap vs N

Sharpe vs N

Factibilidad vs N

Gap vs p

Tiempo vs p

XY vs QAOA

---

## 17. Tablas memoria

Resultados clásicos.

Resultados cuánticos.

Escalabilidad.

Hipótesis.

Hardware utilizado.

---

## 18. Tests

Cobertura mínima 80%.

test_data.py
test_qubo.py
test_ising.py
test_gurobi.py
test_qaoa.py

---

## 19. Criterios éxito

XY mejora factibilidad ≥10%.

Documentar ausencia de ventaja frente a Gurobi.

JASP mejora tiempos ≥20% para p altos.

---

## 20. Orden implementación

1 Datos
2 Estadísticas financieras
3 Markowitz
4 Gurobi
5 QUBO
6 Ising
7 ExactSolver
8 SA
9 QAOA
10 XY
11 JASP
12 Escalabilidad
13 Figuras
14 Tablas
15 Memoria

NO avanzar sin validar la fase previa.

---

## 21. Orden experimentos

Validación matemática.

Benchmarks clásicos.

QAOA.

XY.

JASP.

Escalabilidad.

Experimentos finales.

---

## 22. Riesgos

Tiempos excesivos.

Problemas de memoria.

Versiones incompatibles.

Mitigación:

requirements.txt fijo.

Uso intensivo de Colab.

Checkpoint automático.

---

## 23. Entregables

Repositorio GitHub.

Código reproducible.

Resultados completos.

Memoria TFM.

Presentación.

---

## 24. Principio fundamental

Toda decisión experimental debe quedar registrada automáticamente para garantizar reproducibilidad total.
