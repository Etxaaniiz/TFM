import os
import json

def main():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# TFM: Optimización de Carteras mediante Computación Cuántica Híbrida\n",
                    "\n",
                    "Este cuaderno automatiza por completo la preparación, ejecución y análisis de los experimentos para el Trabajo de Fin de Máster (TFM).\n",
                    "\n",
                    "### Instrucciones de uso:\n",
                    "1. Configura el entorno de Google Colab (ejecución estándar en CPU).\n",
                    "2. Ejecuta cada celda secuencialmente.\n",
                    "3. Una vez finalizado, los resultados se copiarán automáticamente a tu Google Drive para garantizar persistencia total."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Conectar Google Drive y Clonar/Configurar Repositorio"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Montar Google Drive para guardar los resultados\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Detectar la carpeta del proyecto y cambiar de directorio de trabajo\n",
                    "import os\n",
                    "target_dir = None\n",
                    "if os.path.exists('/content/TFM_final'):\n",
                    "    target_dir = '/content/TFM_final'\n",
                    "elif os.path.exists('TFM_final'):\n",
                    "    target_dir = 'TFM_final'\n",
                    "else:\n",
                    "    for root, dirs, files in os.walk('/content'):\n",
                    "        if 'requirements.txt' in files and '.git' not in root:\n",
                    "            target_dir = root\n",
                    "            break\n",
                    "if target_dir:\n",
                    "    print(f'🎯 Carpeta del proyecto encontrada en: {target_dir}')\n",
                    "    %cd {target_dir}\n",
                    "else:\n",
                    "    print('❌ Advertencia: No se encontró el proyecto, el directorio actual es:', os.getcwd())\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Instalar dependencias necesarias desde el archivo requirements.txt\n",
                    "# Nota: Colab Pro ya cuenta con muchas librerías, pero instalaremos las de computación cuántica y optimización\n",
                    "!pip install -r requirements.txt"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Preparar Datos e Instancias"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Descargar los datos históricos y calcular estadísticas (mu y Sigma)\n",
                    "!python scripts/prepare_data.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generar las 40 instancias de validación, principales y escalabilidad\n",
                    "!python scripts/generate_instances.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Ejecutar Experimentos Clásicos"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Gurobi (Solución Exacta de referencia)\n",
                    "!python scripts/run_gurobi.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. ExactSolver (Brute force de dimod)\n",
                    "!python scripts/run_exact.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Simulated Annealing (dwave-neal)\n",
                    "!python scripts/run_sa.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Ejecutar Experimentos Cuánticos Híbridos"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. QAOA Estándar (X Mixer)\n",
                    "!python scripts/run_qaoa.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. XY-QAOA (Constrained Mixer)\n",
                    "!python scripts/run_xy.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Análisis Detallado del QAOA XY Regularizado (Fase 3: Barridos de Alpha y Profundidad)\n",
                    "\n",
                    "Esta sección ejecuta el benchmark científico detallado para comparar el QAOA XY regularizado contra el normal y Gurobi. Analiza la convergencia clásica (mitigación de Barren Plateaus) y el impacto del dial de regularización alpha (alivio del overfitting)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Ejecutar el benchmark científico detallado\n",
                    "!python scripts/run_regularized_qaoa_benchmark.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Graficar los resultados analíticos del benchmark\n",
                    "!python scripts/plot_regularized_qaoa_benchmark.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mostrar las gráficas analíticas generadas\n",
                    "from IPython.display import Image, display\n",
                    "print('--- Grafico 1: Impacto del Coeficiente Alpha en In-Sample vs Out-of-Sample ---')\n",
                    "display(Image(filename='figures/qaoa_analysis_alpha_impact.png'))\n",
                    "print('\\n--- Grafico 2: Evolución de Convergencia del Optimizador Clásico (Barren Plateaus) ---')\n",
                    "display(Image(filename='figures/qaoa_analysis_convergence.png'))\n",
                    "print('\\n--- Grafico 3: Impacto de la Profundidad p en el Optimization Gap ---')\n",
                    "display(Image(filename='figures/qaoa_analysis_depth_gap.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5.5 Benchmark Avanzado: Escalabilidad, Estrés y Eficiencia Temporal\n",
                    "\n",
                    "Esta sección ejecuta el benchmark científico avanzado que evalúa la escalabilidad con N (10, 15, 20 activos), el impacto de los 3 regímenes de estrés (Estable, Volátil, Inflacionario) y la eficiencia clásica (número de iteraciones de COBYLA)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Ejecutar el benchmark avanzado\n",
                    "!python scripts/run_advanced_qaoa_benchmark.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Graficar los resultados del benchmark avanzado\n",
                    "!python scripts/plot_advanced_qaoa_benchmark.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mostrar las gráficas analíticas avanzadas generadas\n",
                    "from IPython.display import Image, display\n",
                    "print('--- Grafico 1: Escalabilidad del Gap In-Sample por Régimen ---')\n",
                    "display(Image(filename='figures/qaoa_advanced_gap_scaling.png'))\n",
                    "print('\\n--- Grafico 2: Eficiencia Clásica (Iteraciones COBYLA) ---')\n",
                    "display(Image(filename='figures/qaoa_advanced_iterations.png'))\n",
                    "print('\\n--- Grafico 3: Escalabilidad del Sharpe Ratio por Régimen ---')\n",
                    "display(Image(filename='figures/qaoa_advanced_sharpe_scaling.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Análisis de Resultados Generales y Generación de Gráficos/Tablas"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generar figuras de resultados generales (PNG)\n",
                    "!python scripts/generate_figures.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generar tablas de resultados generales (LaTeX en formato .tex)\n",
                    "!python scripts/generate_tables.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generar tablas de resultados avanzados de regularización (LaTeX en formato .tex)\n",
                    "!python scripts/generate_advanced_qaoa_tables.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Persistencia y Copia de Seguridad en Google Drive"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Crear carpeta en Drive y guardar todos los entregables\n",
                    "!mkdir -p /content/drive/MyDrive/TFM_resultados\n",
                    "!cp -r results /content/drive/MyDrive/TFM_resultados/\n",
                    "!cp -r figures /content/drive/MyDrive/TFM_resultados/\n",
                    "!cp -r tables /content/drive/MyDrive/TFM_resultados/\n",
                    "print(\"Copia de seguridad completada con éxito. Todos los resultados están a salvo en tu Google Drive.\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    filepath = "TFM_Execution_Colab.ipynb"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Jupyter Notebook successfully created at {filepath}")

if __name__ == "__main__":
    main()
