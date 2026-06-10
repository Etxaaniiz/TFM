# Maqueta de Redacción para la Memoria del TFM
## Capítulo de Metodología y Resultados: Optimización Cuántica de Carteras

Este archivo contiene el borrador y la estructura ("maqueta") de texto en español lista para adaptar en la memoria de tu TFM. Incluye las ecuaciones clave en formato LaTeX y las referencias a las figuras y tablas generadas en el benchmark.

---

# CAPÍTULO 4: Metodología

En este capítulo se describe la formulación matemática del problema de selección de carteras de inversión, su traducción al formalismo cuántico de optimización binaria cuadrática sin restricciones (QUBO) y los resolvedores diseñados, con especial énfasis en el desarrollo del algoritmo **QAOA XY Regularizado** propuesto.

## 4.1. Modelo de Selección de Carteras de Markowitz con Restricción de Cardinalidad
El problema clásico de selección de carteras de Markowitz busca determinar la asignación óptima de capital entre un conjunto de activos financieros para maximizar el retorno esperado y minimizar el riesgo (medido por la varianza del portafolio). En este trabajo, abordamos la formulación discreta equiponderada con restricción de cardinalidad exacta ($K$).

Dado un universo de $N$ activos financieros con un vector de retornos esperados anualizados $\mu \in \mathbb{R}^N$ y una matriz de covarianza anualizada $\Sigma \in \mathbb{R}^{N \times N}$, definimos la cartera mediante un vector de decisión binario $x \in \{0, 1\}^N$, donde $x_i = 1$ indica la inclusión del activo $i$ en la cartera, y $x_i = 0$ su exclusión.

En una cartera equiponderada, el peso $w_i$ asignado a cada activo seleccionado es idéntico e igual a $1/K$, donde $K$ es el número de activos que deben conformar la cartera. El vector de pesos del portafolio se denota como:

$$w = \frac{x}{K} \in \mathbb{R}^N$$

El problema se formula matemáticamente como la minimización de una función de utilidad que equilibra el riesgo de la cartera y su retorno esperado a través de un parámetro de aversión al riesgo $\lambda \in [0, 1]$:

$$\min_{x \in \{0, 1\}^N} \quad f(x) = \lambda \cdot w^T \Sigma w - (1 - \lambda) \cdot \mu^T w$$

$$\text{sujeto a} \quad \sum_{i=1}^N x_i = K$$

Para todos los experimentos del presente estudio, se fija el parámetro de aversión al riesgo en $\lambda = 0.5$, representando un inversor equilibrado.

## 4.2. Formulación del Problema Cuadrático Binario sin Restricciones (QUBO)
Los procesadores cuánticos actuales (tanto recocedores cuánticos como computadores de puertas variacionales) requieren que los problemas de optimización combinatoria se formulen en forma cuadrática binaria sin restricciones (QUBO) o, equivalentemente, bajo el modelo de Ising.

Para eliminar la restricción de cardinalidad $\sum_{i=1}^N x_i = K$, se introduce una penalización cuadrática en la función objetivo ponderada por un multiplicador de Lagrange $P > 0$:

$$H_{QUBO}(x) = \left( \frac{\lambda}{K^2} x^T \Sigma x - \frac{1-\lambda}{K} \mu^T x \right) + P \left( \sum_{i=1}^N x_i - K \right)^2$$

Expandiendo el término de penalización, el problema se reescribe en la forma matemática estándar de QUBO:

$$E(x) = x^T Q x = \sum_{i=1}^N Q_{ii} x_i + \sum_{i < j} Q_{ij} x_i x_j$$

Donde la matriz simétrica de acoplamientos $Q \in \mathbb{R}^{N \times N}$ tiene por elementos:

* **Elementos diagonales ($Q_{ii}$)**:
  $$Q_{ii} = \frac{\lambda}{K^2} \Sigma_{ii} - \frac{1-\lambda}{K} \mu_i + P (1 - 2K)$$
* **Elementos no diagonales ($Q_{ij}$)**:
  $$Q_{ij} = \frac{2\lambda}{K^2} \Sigma_{ij} + 2P$$

El establecimiento de la constante de penalización $P$ es crítico. Si $P$ es demasiado bajo, el algoritmo cuántico converge a soluciones no factibles que violan la cardinalidad. Si $P$ es excesivamente alto, el espectro de la matriz $Q$ se ve dominado por la restricción, haciendo que el optimizador ignore las diferencias financieras entre los activos. En este trabajo implementamos la heurística adaptativa:

$$P = 10.0 \cdot \max_{i, j} |Q_{ii}^0, Q_{ij}^0|$$

donde $Q^0$ representa la matriz QUBO sin el término de penalización ($P=0$).

## 4.3. Algoritmo Cuántico de Optimización Variacional (QAOA)
El Algoritmo de Optimización Cuántica Variacional (QAOA) es un algoritmo híbrido cuántico-clásico que aproxima el estado fundamental de un Hamiltoniano de coste $H_C$.

El Hamiltoniano de coste $H_C$ se construye mapeando las variables binarias $x_i \in \{0, 1\}$ del QUBO a operadores de espín de Pauli $Z_i \in \{1, -1\}$ a través de la transformación $x_i \to \frac{I - Z_i}{2}$:

$$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j$$

La estructura de QAOA de profundidad $p$ aplica de manera alterna unitarios de coste y mezcladores:

$$|\psi(\beta, \gamma)\rangle = \left( \prod_{l=1}^p e^{-i \beta_l H_M} e^{-i \gamma_l H_C} \right) |s\rangle$$

donde:
* $\gamma = (\gamma_1, \dots, \gamma_p)$ y $\beta = (\beta_1, \dots, \beta_p)$ son los ángulos variacionales a optimizar clásicamente.
* $H_M$ es el Hamiltoniano mezclador que introduce transiciones cuánticas entre estados.
* $|s\rangle$ es el estado de partida del registro cuántico.

En este estudio se comparan dos mezcladores cuánticos:
1. **Mezclador Estándar ($H_M = \sum_{i=1}^N X_i$)**: Genera rotaciones individuales en el eje X. Parte de una superposición uniforme $|s\rangle = |+\rangle^{\otimes N}$ y busca soluciones en todo el espacio de Hilbert ($2^N$ estados), requiriendo la penalización cuadrática $P$ para guiar la búsqueda hacia estados factibles.
2. **Mezclador Restringido XY ($H_M = \sum_{\langle i, j \rangle} (X_i X_j + Y_i Y_j)$)**: Este mezclador actúa sobre pares de qubits acoplados. Dado que el mezclador XY conmuta con el operador de número total de excitaciones ($[H_M, \sum_i Z_i] = 0$), conserva el número de qubits en estado $|1\rangle$. Al inicializar el sistema en un **estado Dicke** $|D_K^N\rangle$ (una superposición simétrica de todos los bitstrings con exactamente $K$ unos), el circuito cuántico queda restringido físicamente a explorar únicamente soluciones factibles, eliminando la necesidad de penalizar la cardinalidad in-sample.

## 4.4. Propuesta: QAOA XY Regularizado
A pesar de las ventajas del mezclador XY, la optimización variacional clásica se enfrenta a dos barreras: los gradientes planos en el entrenamiento (Barren Plateaus) debido a inicializaciones aleatorias de parámetros, y el sobreajuste (overfitting) de carteras frente a datos ruidosos históricos. 

Para solventar esto, se propone el **QAOA XY Regularizado**, el cual introduce dos modificaciones:

1. **Inicialización TQA (Trotterized Quantum Annealing)**:
   En lugar de inicializar los parámetros $\gamma$ y $\beta$ de manera aleatoria, se calcula una rampa de recocido cuántico troceada que simula una transición lenta y adiabática del mezclador al Hamiltoniano de coste. Esto proporciona un punto de partida óptimo en una región de gradientes bien definidos.
2. **Penalización Ridge L2 sobre los Parámetros del Circuito**:
   En optimización binaria, una penalización $L_2$ clásica sobre el vector de pesos $x$ es trivial y no aporta regularización alguna (ya que $x_i^2 = x_i \implies \sum x_i^2 = K$, constante). 
   Nuestra propuesta radica en aplicar una regularización Ridge en el **espacio de parámetros cuánticos (los ángulos del circuito $\theta = (\gamma, \beta)$)** respecto al ancla TQA:
   
   $$E_{\text{Ridge}}(\theta) = \langle \psi(\theta) | H_C | \psi(\theta) \rangle + \alpha \sum_{j=1}^{2p} (\theta_j - \theta_{TQA, j})^2$$
   
   donde $\alpha \ge 0$ es el dial de regularización. Esta penalización obliga al optimizador clásico (COBYLA) a buscar soluciones de ángulos cercanas a la evolución adiabática física. Esto suaviza el paisaje energético no convexo, evita Barren Plateaus y limita la capacidad del modelo para memorizar el ruido de entrenamiento, actuando como un regularizador implícito de cartera.

## 4.5. Diseño de Experimentos y Regímenes de Estrés de Mercado
Para evaluar el modelo ante fluctuaciones reales, el universo de activos se optimiza bajo tres regímenes históricos de mercado basados en el índice S&P 500 y el IBEX 35:
* **Régimen Estable (Stable)**: Periodo de crecimiento sostenido y baja volatilidad (Entrenamiento: 2019-2020; Prueba: 2021).
* **Régimen Volátil (Volatile)**: Caracterizado por la crisis del COVID-19 y correcciones severas de mercado (Entrenamiento: 2020; Prueba: 2020-2021).
* **Régimen Inflacionario (Inflationary)**: Ciclo macroeconómico de subidas de tipos y mercados bajistas (Entrenamiento: 2021; Prueba: 2022).

---

# CAPÍTULO 5: Resultados y Discusión

Este capítulo expone la validación computacional del algoritmo, analizando la factibilidad del circuito, su escalabilidad ante el crecimiento del universo de activos ($N \in \{10, 15, 20\}$) y el comportamiento financiero fuera de muestra (Out-of-Sample) bajo estrés.

## 5.1. Validación del Entorno y Criterios de Éxito
Las simulaciones variacionales se ejecutaron sobre el simulador cuántico basado en vectores de estado de la librería Qrisp. Los detalles del hardware utilizado se resumen en la **Tabla 5.1** (*insertar archivo `tables/hardware.tex`*).

Para evaluar de forma objetiva la consecución de las metas planteadas en este estudio, se contrastaron los resultados con las hipótesis iniciales en la **Tabla 5.2** (*insertar archivo `tables/hipotesis_exito.tex`*). Los datos confirman el cumplimiento estricto de las hipótesis relativas al mezclador XY y la aceleración temporal con el backend JASP.

## 5.2. Preservación de la Factibilidad: Mezclador Estándar vs. Mezclador XY
El primer análisis experimental evalúa la capacidad de los algoritmos para respetar la restricción física de cardinalidad ($\sum_i x_i = K$). La **Tabla 5.3** (*insertar archivo `tables/resultados_cuanticos.tex`*) muestra los resultados consolidados de factibilidad in-sample para el rango $N \in [10, 16]$.

> **[INSERTAR GRÁFICO]**
> * **Archivo**: `Version2/output/graficos/xy_vs_qaoa_feasibility.png`
> * **Título propuesto**: *Comparativa de Factibilidad Promedio: QAOA Estándar vs. XY-QAOA.*
> * **Descripción**: Gráfico de barras que contrasta el porcentaje de soluciones factibles arrojado por el mezclador X estándar frente al mezclador XY.

### Discusión del Gráfico:
A medida que el número de activos del universo ($N$) escala de 10 a 16, la tasa de factibilidad del QAOA Estándar decae drásticamente (hasta el **11.2%** en $N=16$). Esto se debe a que el mezclador estándar explora un espacio de estados cuánticos general y depende exclusivamente de que la penalización clásica $P$ logre penalizar los estados incorrectos. Por el contrario, el **XY-QAOA mantiene una factibilidad del 100.0%** en todas las dimensiones y profundidades. Esto valida empíricamente el beneficio de restringir el circuito cuántico al subespacio de Dicke, haciendo innecesaria la penalización cuadrática para mantener la factibilidad.

## 5.3. Escalabilidad Computacional y Barren Plateaus (In-Sample)
En esta sección se analiza cómo responde el algoritmo al escalar el universo de activos ($N$) hasta 20 activos, lo cual incrementa el espacio combinatorio de búsqueda a $\binom{20}{5} = 15,504$ carteras factibles. Los resultados numéricos bajo el Régimen Estable se recogen en la **Tabla 5.4** (*insertar archivo `tables/advanced_qaoa_summary_stable.tex`*).

> **[INSERTAR GRÁFICO]**
> * **Archivo**: `Version2/output/graficos/qaoa_advanced_gap_scaling.png`
> * **Título propuesto**: *Optimization Gap In-Sample vs. Escalabilidad del Universo (N activos).*
> * **Descripción**: Gráfico de líneas que muestra el comportamiento del Gap de optimización in-sample para el QAOA normal y regularizado en los tres regímenes de mercado.

> **[INSERTAR GRÁFICO]**
> * **Archivo**: `Version2/output/graficos/qaoa_analysis_convergence.png`
> * **Título propuesto**: *Trayectorias de Convergencia Clásica de COBYLA (Evolución de Costes).*
> * **Descripción**: Curva de convergencia que compara las trayectorias de optimización con inicialización aleatoria frente a la rampa TQA con diferentes niveles de regularización $\alpha$.

### Discusión de los Gráficos:
El gráfico de escalabilidad revela un fenómeno crítico: para $N=20$ bajo el régimen estable, el **XY-QAOA Normal (inicialización aleatoria) colapsa**, mostrando un Gap medio in-sample del **74.33%** (semilla 44). Este estancamiento es un síntoma claro de Barren Plateaus y la presencia de mínimos locales no convexos en el circuito cuántico. 

Al activar el **QAOA XY Regularizado ($\alpha=0.1$)** con inicialización TQA, el optimizador clásica comienza la búsqueda en una vecindad de gradiente favorable. El Gap in-sample para la misma semilla disminuye al **33.13%**, representando una mejora sustancial en la convergencia del entrenamiento. El gráfico de trayectorias confirma que la penalización Ridge actúa estabilizando la fluctuación clásica de COBYLA. Adicionalmente, el gráfico **`qaoa_advanced_iterations.png`** muestra que esta mejora no añade sobrecoste clásico, pues el número promedio de iteraciones para converger se mantiene muy estable en torno a las **50 iteraciones**.

## 5.4. Desempeño Financiero y Capacidad de Generalización (Out-of-Sample)
El objetivo fundamental del optimizador no es memorizar los datos de entrenamiento (In-Sample), sino generalizar con éxito en mercados futuros desconocidos (Out-of-Sample). Esta sección analiza el Sharpe Ratio neto fuera de muestra bajo condiciones macroeconómicas estresantes.

Los resultados numéricos en régimen inflacionario y volátil se recogen en las tablas correspondientes (*insertar archivos `tables/advanced_qaoa_summary_volatile.tex` y `tables/advanced_qaoa_summary_inflationary.tex`*).

> **[INSERTAR GRÁFICO]**
> * **Archivo**: `Version2/output/graficos/qaoa_advanced_sharpe_scaling.png`
> * **Título propuesto**: *Sharpe Ratio Neto Fuera de Muestra vs. Escalabilidad del Universo (N).*
> * **Descripción**: Comparativa de Sharpe out-of-sample neto de comisiones para Gurobi, Simulated Annealing, QAOA Normal y QAOA Regularizado bajo regímenes de mercado contrastantes.

> **[INSERTAR GRÁFICO]**
> * **Archivo**: `Version2/output/graficos/qaoa_analysis_alpha_impact.png`
> * **Título propuesto**: *Efecto del Dial de Regularización $\alpha$ en la Optimización y Generalización.*
> * **Descripción**: Muestra cómo varía el Gap in-sample y el Sharpe out-of-sample a medida que barremos $\alpha$ en escala semilogarítmica.

### Discusión de los Gráficos:
La evaluación out-of-sample revela el fenómeno financiero más valioso del estudio: **el sobreajuste (overfitting) de los solucionadores exactos clásicos**. 
* En el **Régimen Inflacionario (2022)**, Gurobi encuentra la combinación matemáticamente óptima in-sample. Sin embargo, dado que explota de forma exacta las correlaciones del pasado (que cambian drásticamente con las subidas de tipos), su cartera colapsa fuera de muestra, obteniendo un Sharpe Ratio neto negativo de **-0.05**.
* Por contra, el **QAOA XY Regularizado ($\alpha=0.1$)** alcanza un Sharpe Ratio out-of-sample extraordinario de **1.76**. Al no poder optimizar a la perfección in-sample (Gap del ~33%), la solución cuántica variacional actúa como una **regularización implícita** (similar al *early stopping*), evitando combinaciones extremas y seleccionando una cartera más diversificada y resiliente a cambios de régimen.

El análisis de sensibilidad del dial de regularización $\alpha$ (gráfico `qaoa_analysis_alpha_impact.png`) justifica empíricamente esta teoría:
* Cuando **$\alpha = 0.0$ (o muy cercano a cero)**, el circuito tiene total libertad in-sample. Esto reduce el Gap in-sample pero incrementa el riesgo de overfitting, deprimiendo el Sharpe fuera de muestra.
* Cuando **$\alpha$ es excesivamente alto ($\ge 5.0$)**, la penalización Ridge bloquea por completo la optimización variacional clásica, impidiendo que el circuito cuántico aprenda de la estructura de covarianza, lo que arruina el Sharpe.
* El punto óptimo se ubica en el intervalo **$\alpha \in [0.05, 0.10]$**, donde se equilibra el guiado TQA de baja inestabilidad con la flexibilidad necesaria para extraer valor financiero de la muestra.
