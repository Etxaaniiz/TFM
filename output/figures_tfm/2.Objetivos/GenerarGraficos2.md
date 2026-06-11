Actúa como un físico cuántico y experto en visualización de datos. Genera un script en Python usando matplotlib y numpy para crear una figura conceptual de 1x2 (dos subgráficos 3D lado a lado) que ilustre el fenómeno de las "Mesetas Estériles" (Barren Plateaus) en circuitos variacionales.
- Configura el estilo con `plt.style.use('seaborn-v0_8-whitegrid')` y el tamaño de la figura a 12x5.
- Genera una malla de datos (X, Y) desde -5 hasta 5.
- Subgráfico 1 (Izquierda - "A. Paisaje Optimizable"): Crea una función de coste convexa, como un paraboloide clásico `Z1 = X**2 + Y**2`. Grafícalo usando `plot_surface` con un mapa de color 'viridis'. Esto representa un paisaje con gradientes fuertes donde un optimizador puede descender fácilmente al mínimo (0,0).
- Subgráfico 2 (Derecha - "B. Meseta Estéril"): Crea un paisaje plano con un agujero muy estrecho en el centro. Usa la función `Z2 = -np.exp(-(X**2 + Y**2) / 0.1)`. Grafícalo con el mismo mapa de color. Esto ilustra un 'Barren Plateau' donde el gradiente es cero en casi todo el espacio, salvo en un pozo de potencial minúsculo.
- Elimina los números de los ejes X, Y, Z (`set_xticklabels([])`, etc.) ya que es un gráfico conceptual abstracto.
- Añade títulos a cada subgráfico ("A. Paisaje Optimizable" y "B. Meseta Estéril / Barren Plateau").
- Guarda la figura con 300 DPI exactos en './output/figures_tfm/2.Objetivos/barren_plateaus.png'. Asegúrate de crear el directorio si no existe.

Actúa como un experto en topología y computación cuántica. Escribe un script en Python usando matplotlib para generar una figura conceptual 3D de 1x2 (dos subgráficos) que demuestre la reducción del espacio de búsqueda al usar el subespacio de Dicke (XY-Mixer) frente al espacio de Hilbert completo (Standard QAOA).
- Usa `plt.style.use('seaborn-v0_8-whitegrid')` y tamaño de figura 12x5.
- El concepto se ilustrará usando un hipercubo booleano de 3 cúbits (N=3). Genera los 8 vértices posibles del cubo 3D: (0,0,0), (0,0,1), ..., (1,1,1).
- Subgráfico 1 (Izquierda - "QAOA Estándar: Espacio Completo 2^N"): Dibuja los 8 vértices como puntos 3D grandes. Pinta en rojo translúcido los estados inválidos y en verde brillante los estados donde la suma de bits sea exactamente K=1 (es decir, (1,0,0), (0,1,0), (0,0,1)). Dibuja las aristas del cubo en gris claro para dar sensación de volumen.
- Subgráfico 2 (Derecha - "XY-QAOA: Subespacio de Dicke"): Grafica ÚNICAMENTE los 3 vértices verdes válidos (K=1). Dibuja un plano triangular semitransparente (color verde claro, alpha=0.3) que conecte estos tres puntos (1,0,0), (0,1,0), (0,0,1). Esto representa la restricción geométrica del ansatz XY que confina la búsqueda a este plano exacto, omitiendo el resto del cubo.
- Ajusta el ángulo de visión de ambos gráficos (`view_init(elev=20, azim=30)`) para que sean idénticos y fáciles de comparar.
- Oculta los ejes numéricos para mayor limpieza.
- Añade títulos descriptivos a los subgráficos.
- Guarda la figura final con 300 DPI en './output/figures_tfm/2.Objetivos/xy_vs_qaoa_feasibility.png'.