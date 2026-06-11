Actúa como un Data Scientist. Escribe un script de Python que lea los datos reales del archivo `output/results/results.csv` usando Pandas. El objetivo es generar un gráfico de líneas que muestre el 'Optimization GAP (%)' medio en función de 'N' (Número de activos).
- Filtra y agrupa los datos por la columna 'Solver' y la columna 'N'.
- Calcula la media del 'Optimization GAP' para cada combinación.
- Grafica las líneas correspondientes a los solvers: "Simulated Annealing", "Standard QAOA" y "XY-QAOA Regularized" (ajusta los nombres exactos según el CSV).
- Aplica el estilo `plt.style.use('seaborn-v0_8-whitegrid')`.
- Eje X: Valores únicos de N presentes en el CSV. Eje Y: Optimization GAP (%) real.
- Añade marcadores a las líneas, título, etiquetas de ejes, leyenda y guarda la figura con 300 DPI en `output/figures_tfm/6.Resultados/gap_vs_N.png`. Asegúrate de crear el directorio si no existe.

Genera un script en Python que lea el archivo `output/results/results.csv` de este repositorio con Pandas para evaluar el rendimiento financiero real.
- Extrae las columnas correspondientes a 'N', 'Solver' y 'Sharpe Ratio Out-of-Sample' (o la métrica financiera equivalente en el dataset).
- Agrupa por 'Solver' y 'N', calculando la media del Sharpe Ratio.
- Traza un gráfico de líneas comparando al menos los solvers: "Gurobi" (línea de referencia base) y "XY-QAOA Regularized". Si el CSV contiene métricas In-Sample vs Out-of-Sample diferenciadas para Gurobi, grafica ambas.
- Aplica el estilo `seaborn-v0_8-whitegrid`. 
- Usa colores contrastantes (ej. Negro punteado para Gurobi In-Sample, Verde sólido grueso para XY-QAOA OOS).
- Añade título, etiquetas de ejes, leyenda y guarda la imagen con 300 DPI en `output/figures_tfm/6.Resultados/sharpe_vs_N.png`.

Crea un script de Python que analice la complejidad temporal real de los experimentos. Lee el archivo `output/results/results.csv` con Pandas.
- Extrae 'N', 'Solver' y la columna de tiempo de ejecución (ej. 'Execution Time (s)' o 'QPU_Time' / 'CPU_Time').
- Agrupa por 'Solver' y 'N' para calcular la media de los tiempos de ejecución.
- Genera un gráfico de líneas configurando el Eje Y en escala LOGARÍTMICA (`plt.yscale('log')`).
- Grafica las curvas para "Gurobi", "Simulated Annealing" y el solver cuántico "JaspQAOA" o "XY-QAOA". El script debe reflejar los tiempos reales extraídos del CSV, mostrando el cruce temporal o el escalamiento si lo hubiere.
- Estilo: `seaborn-v0_8-whitegrid`. 
- Añade título, etiquetas, leyenda y guarda con 300 DPI en `output/figures_tfm/6.Resultados/tiempo_vs_N.png`.

Genera un script de Python que lea `output/results/results.csv` para trazar la viabilidad empírica de las soluciones.
- Extrae 'N', 'Solver' y la métrica de 'Feasibility Ratio' (o Tasa de Soluciones Válidas).
- Agrupa por 'Solver' y 'N' calculando la media.
- Filtra para graficar exclusivamente la comparativa entre "Standard QAOA" y "XY-QAOA" (o nombres equivalentes en tu CSV).
- Eje X: N. Eje Y: Feasibility Ratio real (%).
- Fija el límite del Eje Y de 0 a 110 (o de 0.0 a 1.1 según la escala del CSV) para evidenciar cómo el XY-Mixer se mantiene bloqueado en el 100%.
- Estilo: `seaborn-v0_8-whitegrid`. Guarda el gráfico final con 300 DPI en `output/figures_tfm/6.Resultados/factibilidad_vs_N.png`.

Escribe un script de Python que evalúe el efecto de añadir capas algorítmicas en el rendimiento empírico. Lee `output/results/results.csv`.
- Filtra los datos para dejar solo las filas de solvers QAOA y fija un único tamaño de problema representativo (por ejemplo, N=14).
- Agrupa por 'Solver' y 'p' (profundidad), calculando la media del 'Optimization GAP'.
- Grafica en el Eje X la profundidad 'p' y en el Eje Y el GAP real obtenido.
- Compara las líneas del QAOA tradicional vs el XY-QAOA Regularizado presentes en el CSV.
- Añade marcadores a los puntos de datos reales, estilo `seaborn-v0_8-whitegrid`, título, leyenda, y guarda en `output/figures_tfm/6.Resultados/gap_vs_p.png` a 300 DPI.

Escribe un script de Python que evalúe el coste temporal neto del procesamiento (segundos) en el bucle variacional híbrido clásico-cuántico en función del incremento de la profundidad de capas (p). Lee `output/results/results.csv`.
- Filtra los datos para dejar solo las filas de solvers QAOA y fija un único tamaño de problema representativo (N=14).
- Agrupa por 'Solver' y 'p' (profundidad) calculando la media del tiempo de ejecución ('Execution Time (s)').
- Grafica en el Eje X la profundidad 'p' (con un amplio abanico de capas, de 1 a 10) y en el Eje Y el tiempo de ejecución real obtenido.
- Compara las líneas del QAOA tradicional vs el XY-QAOA Regularizado.
- Añade marcadores a los puntos de datos, estilo `seaborn-v0_8-whitegrid`, título, leyenda y guarda con 300 DPI en `output/figures_tfm/6.Resultados/tiempo_vs_p.png`.

Actúa como un ingeniero de visualización de datos. Genera un script en Python que lea `results/results.csv` y construya un Gráfico de Radar (Spider Chart) basado en datos empíricos agregados.
- Agrupa el DataFrame entero por 'Solver' y calcula la media global de 4 métricas clave:
  1. Viabilidad (Feasibility Ratio).
  2. Precisión (Transforma el GAP en una métrica de precisión, ej: 100 - GAP).
  3. Robustez (Sharpe Ratio Out-of-Sample).
  4. Eficiencia Temporal (Aplica una transformación inversa o logarítmica al tiempo de ejecución para que "mayor valor" signifique "más rápido").
- Normaliza las 4 métricas calculadas en una escala común (ej. MinMaxScaler de 0 a 1) usando `sklearn.preprocessing` o normalización manual.
- Selecciona los 4 solvers principales (Gurobi, Simulated Annealing, Standard QAOA, XY-QAOA Reg) y traza el radar chart usando Matplotlib (`polar=True`).
- Rellena el área de cada polígono (`fill`) con `alpha=0.2`.
- Añade leyenda de los solvers, etiquetas en los ejes del radar y guarda con 300 DPI en `output/figures_tfm/6.Resultados/radar_chart_performance.png`.