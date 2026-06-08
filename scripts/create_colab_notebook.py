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
                    "1. Configura el entorno de Google Colab Pro para usar una GPU (opcional, recomendado para acelerar JaspQAOA: *Entorno de ejecución* -> *Cambiar tipo de entorno de ejecución* -> *T4 GPU* o superior).\n",
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
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 6. JaspQAOA (JIT con JAX y Terminal Sampling)\n",
                    "!python scripts/run_jasp.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Análisis de Resultados y Generación de Gráficos/Tablas"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generar figuras de resultados (PNG)\n",
                    "!python scripts/generate_figures.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generar tablas de resultados (LaTeX en formato .tex)\n",
                    "!python scripts/generate_tables.py"
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
