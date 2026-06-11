Actúa como un ingeniero de visualización de datos. Escribe un script en Python utilizando `matplotlib` (específicamente `matplotlib.patches` y `ax.annotate`) para generar un diagrama de flujo horizontal conceptual que represente el mapeo matemático del TFM.
- Configura una figura apaisada (ej. 12x4) sin ejes numéricos (`ax.axis('off')`).
- Dibuja 3 cajas rectangulares alineadas horizontalmente.
- Caja 1 (Izquierda): Texto "Datos de Entrada\nMatriz de covarianza $\Sigma$\nVector de retornos esperados $\mu$". Color de fondo azul claro.
- Caja 2 (Centro): Texto "Formulación Matemática QUBO\nFunción objetivo con restricciones\nincorporadas (Penalización $A$)". Color de fondo verde claro.
- Caja 3 (Derecha): Texto "Hamiltoniano de Ising\nRed de espines formada por cúbits\nAcoplamientos $J_{ij}$ y Campos $h_i$". Color de fondo púrpura claro.
- Dibuja flechas gruesas (usando `FancyArrowPatch` o `ax.annotate`) que conecten la Caja 1 con la 2, y la 2 con la 3.
- Utiliza una tipografía clara y formal (tamaño 12-14).
- Guarda la figura con 300 DPI en `output/figures_tfm/3.Desarrollo/diagrama_flujo_mapeo.png`. Crea la carpeta si no existe.

Crea un script en Python utilizando `matplotlib` para generar un diagrama conceptual (tipo Diagrama de Euler/Venn modificado) que ilustre la reducción del espacio de búsqueda.
- Configura una figura cuadrada (ej. 8x8) y oculta los ejes (`ax.axis('off')`).
- Dibuja un círculo grande (o una elipse) que ocupe la mayor parte del lienzo. Rellénalo con un color rojo muy tenue (`alpha=0.2`). Añade una etiqueta de texto en la parte superior: "Espacio Completo de Hilbert ($2^N$)\nEstados no factibles explorados por QAOA Estándar".
- Dentro de este gran círculo, dibuja un círculo mucho más pequeño en el centro. Rellénalo con un color verde sólido o azul intenso (`alpha=0.7`). 
- Añade una flecha apuntando a este círculo pequeño con la etiqueta: "Subespacio de Dicke $\\binom{N}{K}$\nConfiguraciones válidas (XY-QAOA)".
- Puedes añadir pequeños puntos grises dispersos en el círculo grande y puntos blancos en el pequeño para simular los autoestados cuánticos.
- Guarda la figura con 300 DPI en `output/figures_tfm/3.Desarrollo/espacio_dicke_combinatorio.png`.

Actúa como un Ingeniero Cuántico. Genera un script de Python que utilice `qiskit` para dibujar un esquema conceptual del circuito XY-QAOA y exportarlo como imagen mediante Matplotlib.
- Importa `QuantumCircuit` de `qiskit`. Crea un circuito de 4 cúbits (N=4).
- Paso 1: Añade una instrucción/caja personalizada (usando `Instruction` o `Gate` ficticia) que abarque los 4 cúbits llamada "Init: Estado de Dicke $|D_K^N\\rangle$".
- Añade una barrera visual (`circuit.barrier()`).
- Paso 2: Añade una caja personalizada que abarque todos los cúbits llamada "Coste: $U_C(\\gamma)$".
- Paso 3: Añade cajas personalizadas entre pares de cúbits (ej. q0-q1, q1-q2, q2-q3) llamadas "Mixer: $U_M^{XY}(\\beta)$ ($XX+YY$)".
- Repite el bloque de Coste y Mixer para simbolizar una segunda capa (p=2) o añade unos puntos suspensivos ("...").
- Paso 4: Añade operaciones de medición estándar al final del circuito para todos los cúbits.
- Usa `circuit.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})` para renderizar el circuito.
- Guarda la figura final generada por Qiskit con 300 DPI en `output/figures_tfm/3.Desarrollo/topologia_circuito_xy.png`.


Actúa como Investigador Cuántico y Data Scientist. Escribe un script en Python que evalúe y compare empíricamente el paisaje de la función de coste con y sin regularización.
- Utiliza la lógica de tu repositorio (`src.quantum` o el simulador que utilices) para definir un problema QAOA muy pequeño (ej. N=4, K=2) con profundidad p=1.
- Define una malla bidimensional (grid) para los parámetros $\gamma \in [0, \pi]$ y $\beta \in [0, \pi]$ con al menos 30x30 puntos.
- Para cada par $(\gamma, \beta)$, calcula el valor esperado de la energía del Hamiltoniano. Guarda estos valores en una matriz 2D `Z_sin_reg`.
- Define un ancla TQA imaginaria, por ejemplo $(\gamma_{TQA}, \beta_{TQA}) = (1.0, 1.0)$.
- Calcula la matriz regularizada `Z_con_reg = Z_sin_reg + \alpha * ((\gamma - \gamma_{TQA})^2 + (\beta - \beta_{TQA})^2)` usando un $\alpha=5.0$ (o un valor que logre un suavizado visible).
- Crea una figura con 2 subgráficos tridimensionales (1x2 `projection='3d'`).
- Subgráfico Izquierdo ("Sin regularización"): Traza la superficie `plot_surface` de `Z_sin_reg` usando colormap 'plasma'. El paisaje se verá rugoso.
- Subgráfico Derecho ("Con Regularización Ridge $L_2$"): Traza la superficie de `Z_con_reg`. Se verá como un embudo suavizado hacia el ancla TQA.
- Etiqueta los ejes X como '$\gamma$', Y como '$\beta$' y Z como 'Coste Esperado'.
- Aplica `plt.style.use('seaborn-v0_8-whitegrid')` y guarda con 300 DPI en `output/figures_tfm/3.Desarrollo/paisaje_regularizacion_ridge.png`.