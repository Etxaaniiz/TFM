import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_wbs():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis('off')
    
    # Define styles
    root_style = dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="none")
    wp_style = dict(boxstyle="round,pad=0.4", facecolor="#0284C7", edgecolor="none")
    task_style = dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9", edgecolor="#CBD5E1")
    
    # Root node
    ax.text(0.5, 0.9, "TFM: Optimización Cuántica de Carteras\n(XY-QAOA + Regularización Ridge)", 
            ha="center", va="center", color="white", weight="bold", fontsize=11, bbox=root_style)
    
    # WPs
    wps = [
        ("WP1: Revisión y\nFundamentos", 0.1, ["1.1 Estado del arte", "1.2 Formulación Markowitz", "1.3 Complejidad NP-hard"]),
        ("WP2: Modelado\nMatemático", 0.3, ["2.1 Mapeo a QUBO", "2.2 Hamiltoniano Ising", "2.3 Regularización L2"]),
        ("WP3: Desarrollo e\nImplementación", 0.5, ["3.1 Ingesta de datos", "3.2 Estado de Dicke", "3.3 Mezclador XY"]),
        ("WP4: Benchmarking\ny Simulación", 0.7, ["4.1 Gurobi y SA", "4.2 Simulación QAOA", "4.3 Análisis métricas"]),
        ("WP5: Documentación\ny Diseminación", 0.9, ["5.1 Redacción Memoria", "5.2 Generación Figuras", "5.3 Defensa TFM"])
    ]
    
    for title, x_pos, tasks in wps:
        # Line from root to WP
        ax.plot([0.5, x_pos], [0.82, 0.65], color="#94A3B8", lw=1.5, zorder=1)
        # WP Box
        ax.text(x_pos, 0.65, title, ha="center", va="center", color="white", weight="bold", fontsize=9, bbox=wp_style, zorder=2)
        
        # Tasks under WP
        y_start = 0.48
        for i, task in enumerate(tasks):
            y_pos = y_start - i * 0.12
            ax.plot([x_pos, x_pos], [0.58, y_pos], color="#CBD5E1", lw=1.0, zorder=1)
            ax.text(x_pos, y_pos, task, ha="center", va="center", color="#334155", fontsize=8, bbox=task_style, zorder=2)
            
    plt.tight_layout()
    os.makedirs("output/figures", exist_ok=True)
    plt.savefig("output/figures/wbs_tree.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Guardado: output/figures/wbs_tree.png")

def generate_gantt():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    
    tasks = [
        ("WP1: Revisión Bibliográfica y Fundamentos", 0, 4, "#0284C7", False),
        ("WP2: Modelado Matemático y QUBO", 3, 5, "#EA580C", True),      # Critical path
        ("WP3: Implementación Software e Ingesta", 7, 7, "#EA580C", True), # Critical path
        ("WP4: Campaña Experimental y Benchmarks", 13, 6, "#EA580C", True),# Critical path
        ("WP5: Redacción de Memoria y Conclusiones", 17, 7, "#10B981", False)
    ]
    
    y_pos = np.arange(len(tasks))
    
    for i, (task_name, start_week, duration, color, is_crit) in enumerate(tasks):
        ax.barh(i, duration, left=start_week, height=0.5, align='center', color=color, alpha=0.85, edgecolor='#1E293B', linewidth=1)
        label_text = f"{task_name}" + (" (Camino Crítico)" if is_crit else "")
        ax.text(start_week + duration/2, i, f"{duration} sem.", ha='center', va='center', color='white', weight='bold', fontsize=9)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[0] for t in tasks], fontsize=9, color="#1E293B")
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Semanas de Proyecto (Cronograma)", fontsize=10, weight='bold', color="#1E293B")
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 2))
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Custom legend
    legend_elements = [
        patches.Patch(facecolor='#EA580C', edgecolor='#1E293B', label='Camino Crítico (WP2 $\\rightarrow$ WP3 $\\rightarrow$ WP4)'),
        patches.Patch(facecolor='#0284C7', edgecolor='#1E293B', label='Fase Previa (WP1)'),
        patches.Patch(facecolor='#10B981', edgecolor='#1E293B', label='Fase Final / Diseminación (WP5)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=9)
    
    plt.title("Cronograma de Trabajo y Diagrama de Gantt del TFM", weight='bold', fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig("output/figures/gantt_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Guardado: output/figures/gantt_chart.png")

if __name__ == "__main__":
    generate_wbs()
    generate_gantt()
