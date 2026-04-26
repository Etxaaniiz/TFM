# Memoria de trabajo del TFM (borrador vivo)

Este documento sirve como cuaderno de trabajo durante el desarrollo del TFM.
La version definitiva se redactara en LaTeX al final, reutilizando este contenido.

## 1. Titulo provisional
Comparativa reproducible entre optimizacion clasica y cuantica hibrida en seleccion discreta de carteras

## 2. Objetivo general
Evaluar de forma reproducible el rendimiento de metodos clasicos y cuanticos hibridos para seleccion discreta de carteras, considerando calidad de solucion, robustez y costo computacional.

## 3. Pregunta de investigacion
En que condiciones un enfoque cuantico hibrido NISQ (QAOA, con VQE opcional) resulta competitivo frente a metodos clasicos en seleccion discreta de activos.

## 4. Alcance acordado
- Datos historicos reales (sin dependencia de seguimiento en tiempo real).
- Formulacion QUBO/Ising del problema media-varianza discretizado con cardinalidad.
- Comparacion de al menos dos baselines clasicos y un metodo cuantico principal (QAOA).
- Simulador como nucleo del trabajo; hardware real como extension opcional.
- Instancias pequenas y medianas para asegurar profundidad de analisis.

## 5. Hipotesis
- H1: En instancias pequenas, metodos clasicos exactos dominan en calidad y costo.
- H2: QAOA puede aproximar soluciones competitivas en escenarios concretos.
- H3: Ruido y profundidad de circuito degradan rendimiento en entornos NISQ reales.

## 6. Metricas de evaluacion
### 6.1 Calidad financiera
- Rentabilidad esperada
- Volatilidad
- Ratio de Sharpe

### 6.2 Calidad de optimizacion
- Valor de funcion objetivo
- Gap relativo frente al optimo o mejor solucion conocida
- Tasa de factibilidad

### 6.3 Costo y robustez
- Tiempo de ejecucion
- Numero de evaluaciones
- Shots, qubits y profundidad de circuito
- Variabilidad entre semillas y escenarios de ruido

## 7. Registro de decisiones (bitacora)
Anotar aqui decisiones importantes con fecha y justificacion.

## 8. Registro de experimentos
Formato sugerido por experimento:
- Fecha:
- Dataset y periodo:
- Tamano de instancia:
- Metodo:
- Configuracion:
- Resultado principal:
- Observaciones:

## 9. Resultados consolidados
Espacio para tablas y conclusiones parciales que luego pasaran a LaTeX.

## 10. Riesgos y mitigaciones
- Riesgo: dependencia de hardware real.
- Mitigacion: resultados principales en simulador.

- Riesgo: alcance excesivo.
- Mitigacion: limitar metodos y tamanos de instancia.

## 11. Pendientes inmediatos
- Cerrado el 2026-04-27: universo de activos y ventana temporal.
- Cerrado el 2026-04-27: baselines clasicos concretos.
- Cerrado el 2026-04-27: protocolo experimental minimo.

## 12. Decisiones cerradas (MVP experimental)
### 12.1 Universo de activos y periodo
- Universo inicial: 25 acciones liquidas y estables de mercado USA (mega caps) con datos diarios en Yahoo Finance.
- Ventana temporal total: 2019-01-01 a 2024-12-31.
- Esquema temporal:
	- Entrenamiento/calibracion: 2019-01-01 a 2022-12-31.
	- Evaluacion out-of-sample: 2023-01-01 a 2024-12-31.

### 12.2 Formulacion y tamanos
- Formulacion base: media-varianza discretizada con restriccion de cardinalidad fija.
- Transformacion a QUBO/Ising para ejecucion cuantica y comparacion homogenea.
- Tamanos de instancia para benchmark: n = 8, 12, 16 activos.
- Cardinalidad por instancia: k = n/4 (redondeo al entero mas cercano).

### 12.3 Baselines clasicos (fijados)
- Baseline clasico exacto: busqueda exhaustiva sobre combinaciones factibles (referencia de optimo) para n <= 16.
- Baseline clasico heuristico: Simulated Annealing sobre QUBO (multiple semillas).
- Metodo cuantico principal: QAOA (p=1 y p=2) en simulador ideal y con ruido.

### 12.4 Configuracion minima de experimentos
- Repeticiones por configuracion: 10 semillas.
- QAOA:
	- Optimizador clasico: COBYLA.
	- Iteraciones maximas: 100.
	- Shots por evaluacion: 1024.
- Simuladores:
	- Ideal: statevector/qasm segun configuracion.
	- Con ruido: noise model basado en backend fake de IBM.

### 12.5 Metricas finales obligatorias
- Financieras: rentabilidad esperada, volatilidad, Sharpe.
- Optimizacion: valor objetivo, gap relativo, factibilidad.
- Computacionales: tiempo, evaluaciones, shots, profundidad, qubits.
- Robustez: media y desviacion estandar entre semillas.

### 12.6 Criterio de exito del TFM
Se considera exitoso si se entrega una comparativa reproducible con resultados consistentes en n=8,12,16, analisis de ruido y discusion critica de limites NISQ, aunque no exista ventaja cuantica frente a clasico.
