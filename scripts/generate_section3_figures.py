import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

# ==============================================================================
# PHYSICS EMULATOR FOR LANDSCAPE SIMULATION
# ==============================================================================

def apply_cost_layer(state, E_diag, gamma):
    return state * np.exp(-1j * gamma * E_diag)

def apply_rx_mixer_layer(state, beta, N):
    state = state.reshape([2] * N)
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    for j in range(N):
        state = cos_b * state - 1j * sin_b * np.roll(state, shift=1, axis=j)
    return state.flatten()

def run_emulator_qaoa_rx_p1(N, Q, gamma, beta):
    # Initial state is uniform superposition
    state = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
    
    # Precompute energies
    E_diag = np.zeros(2**N)
    for i in range(2**N):
        x = np.array([int(b) for b in bin(i)[2:].zfill(N)])
        E_diag[i] = x.T @ Q @ x
        
    # Apply cost layer
    state = apply_cost_layer(state, E_diag, gamma)
    
    # Apply RX mixer layer
    state = apply_rx_mixer_layer(state, beta, N)
    
    probs = np.abs(state) ** 2
    expected_energy = np.sum(E_diag * probs)
    return expected_energy

# ==============================================================================
# STYLING
# ==============================================================================

def setup_thesis_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass
        
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })

# ==============================================================================
# PLOT GENERATORS
# ==============================================================================

def generate_plot_3_1_flowchart(output_dir):
    print("Generando diagrama_flujo_mapeo.png...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Box configurations
    # We draw them using text boxes with rounded bounding boxes (bbox)
    box_style_1 = dict(boxstyle="round,pad=0.8", facecolor="#E0F2FE", edgecolor="#0284C7", lw=2)
    box_style_2 = dict(boxstyle="round,pad=0.8", facecolor="#DCFCE7", edgecolor="#15803D", lw=2)
    box_style_3 = dict(boxstyle="round,pad=0.8", facecolor="#F3E8FF", edgecolor="#7E22CE", lw=2)
    
    text1 = "Datos de Entrada\n\nMatriz de covarianza $\Sigma$\nVector de retornos esperados $\mu$"
    text2 = "Formulación Matemática QUBO\n\nFunción objetivo con restricciones\nincorporadas (Penalización $A$)"
    text3 = "Hamiltoniano de Ising\n\nRed de espines formada por cúbits\nAcoplamientos $J_{ij}$ y Campos $h_i$"
    
    # Box centers on coordinate system
    ax.text(2.0, 5.0, text1, ha='center', va='center', bbox=box_style_1, fontsize=11, weight='bold', color='#1E293B')
    ax.text(6.0, 5.0, text2, ha='center', va='center', bbox=box_style_2, fontsize=11, weight='bold', color='#1E293B')
    ax.text(10.0, 5.0, text3, ha='center', va='center', bbox=box_style_3, fontsize=11, weight='bold', color='#1E293B')
    
    # Connecting Arrows
    arrow1 = FancyArrowPatch((3.6, 5.0), (4.4, 5.0), arrowstyle='-|>', mutation_scale=20, color='#64748B', linewidth=3.0)
    arrow2 = FancyArrowPatch((7.6, 5.0), (8.4, 5.0), arrowstyle='-|>', mutation_scale=20, color='#64748B', linewidth=3.0)
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(3, 7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagrama_flujo_mapeo.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado diagrama_flujo_mapeo.png")

def generate_plot_3_2_venn(output_dir):
    print("Generando espacio_dicke_combinatorio.png...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    # Ellipse/Circle 1: Full Hilbert Space (Standard QAOA 2^N)
    large_circle = plt.Circle((4.0, 4.0), 3.2, facecolor='#FEE2E2', edgecolor='#EF4444', alpha=0.3, linestyle='--', linewidth=2.0)
    ax.add_patch(large_circle)
    ax.text(4.0, 6.7, "Espacio Completo de Hilbert ($2^N$)\nEstados no factibles explorados por QAOA Estándar", 
            ha='center', va='center', fontsize=11, color='#991B1B', weight='bold')
    
    # Circle 2: Dicke Subspace (XY-QAOA)
    small_circle = plt.Circle((4.0, 4.0), 0.9, facecolor='#10B981', edgecolor='#047857', alpha=0.75, linewidth=2.0)
    ax.add_patch(small_circle)
    
    # Draw gray dots randomly in the large circle (outside small circle)
    np.random.seed(1337)
    for _ in range(160):
        r = np.random.uniform(0.95, 3.1)
        theta = np.random.uniform(0, 2*np.pi)
        x = 4.0 + r * np.cos(theta)
        y = 4.0 + r * np.sin(theta)
        ax.scatter(x, y, color='#94A3B8', s=15, alpha=0.6, edgecolors='none')
        
    # Draw white dots inside the small circle
    for _ in range(18):
        r = np.random.uniform(0, 0.8)
        theta = np.random.uniform(0, 2*np.pi)
        x = 4.0 + r * np.cos(theta)
        y = 4.0 + r * np.sin(theta)
        ax.scatter(x, y, color='white', s=18, alpha=0.9, edgecolors='none', zorder=5)
        
    # Annotation arrow to Dicke Subspace
    ax.annotate("Subespacio de Dicke $\\binom{N}{K}$\nConfiguraciones válidas (XY-QAOA)",
                xy=(4.2, 4.2), xytext=(5.8, 1.8),
                arrowprops=dict(facecolor='#047857', edgecolor='#047857', lw=1.5, arrowstyle="->", shrinkB=5,
                                connectionstyle="arc3,rad=-0.1"),
                fontsize=11, color='#047857', weight='bold', ha='center')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "espacio_dicke_combinatorio.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado espacio_dicke_combinatorio.png")

def generate_plot_3_3_circuit(output_dir):
    print("Generando topologia_circuito_xy.png...")
    # Create a circuit of 4 qubits
    qc = QuantumCircuit(4)
    
    # Step 1: Dicke State Init
    init_gate = Gate("Init: Estado de Dicke |D_K^N>", 4, [])
    qc.append(init_gate, [0, 1, 2, 3])
    qc.barrier()
    
    # Step 2: Cost layer U_C(gamma) for p=1
    cost_gate = Gate("Coste: U_C(gamma)", 4, [])
    qc.append(cost_gate, [0, 1, 2, 3])
    
    # Step 3: XY Mixer layer U_M^XY(beta)
    mixer_gate = Gate("Mixer: U_M^{XY}(beta) (XX+YY)", 2, [])
    qc.append(mixer_gate, [0, 1])
    qc.append(mixer_gate, [1, 2])
    qc.append(mixer_gate, [2, 3])
    qc.barrier()
    
    # Repetition for p=2 (to demonstrate layering)
    cost_gate_p2 = Gate("Coste: U_C(gamma_2)", 4, [])
    mixer_gate_p2 = Gate("Mixer: U_M^{XY}(beta_2) (XX+YY)", 2, [])
    qc.append(cost_gate_p2, [0, 1, 2, 3])
    qc.append(mixer_gate_p2, [0, 1])
    qc.append(mixer_gate_p2, [1, 2])
    qc.append(mixer_gate_p2, [2, 3])
    qc.barrier()
    
    # Step 4: Measurement
    qc.measure_all()
    
    # Render with Qiskit's matplotlib drawer
    fig = qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    fig.savefig(os.path.join(output_dir, "topologia_circuito_xy.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Guardado topologia_circuito_xy.png")

def generate_plot_3_4_regularization(output_dir):
    print("Generando paisaje_regularizacion_ridge.png...")
    
    # N=4, K=2, p=1, simple symmetric matrix Q
    N = 4
    np.random.seed(42)
    Q = np.array([
        [-0.15,  0.10, -0.05,  0.22],
        [ 0.10, -0.25,  0.08, -0.12],
        [-0.05,  0.08, -0.18,  0.05],
        [ 0.22, -0.12,  0.05, -0.10]
    ])
    
    # Grid 30x30 for gamma and beta in [0, pi]
    gamma_vals = np.linspace(0, np.pi, 30)
    beta_vals = np.linspace(0, np.pi, 30)
    Gamma, Beta = np.meshgrid(gamma_vals, beta_vals)
    
    Z_sin_reg = np.zeros((30, 30))
    for i, g in enumerate(gamma_vals):
        for j, b in enumerate(beta_vals):
            Z_sin_reg[j, i] = run_emulator_qaoa_rx_p1(N, Q, g, b)
            
    # TQA Anchor at (1.0, 1.0) and alpha=5.0
    gamma_tqa, beta_tqa = 1.0, 1.0
    alpha = 5.0
    Z_con_reg = Z_sin_reg + alpha * ((Gamma - gamma_tqa)**2 + (Beta - beta_tqa)**2)
    
    fig = plt.figure(figsize=(13, 5.5))
    
    # 1. Left Plot: Sin Regularización
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(Gamma, Beta, Z_sin_reg, cmap='plasma', edgecolor='none', alpha=0.9, antialiased=True)
    ax1.set_title("A. Paisaje de Coste Sin Regularización (Rugoso)", weight='bold', pad=15)
    ax1.set_xlabel(r'$\gamma$ (Coste)')
    ax1.set_ylabel(r'$\beta$ (Mezclador)')
    ax1.set_zlabel('Coste Esperado')
    
    ax1.view_init(elev=30, azim=135)
    
    # 2. Right Plot: Con Regularización Ridge L2
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(Gamma, Beta, Z_con_reg, cmap='plasma', edgecolor='none', alpha=0.9, antialiased=True)
    ax2.set_title(r"B. Paisaje Con Regularización Ridge $L_2$ (Suave)", weight='bold', pad=15)
    ax2.set_xlabel(r'$\gamma$ (Coste)')
    ax2.set_ylabel(r'$\beta$ (Mezclador)')
    ax2.set_zlabel('Coste Esperado')
    
    ax2.view_init(elev=30, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paisaje_regularizacion_ridge.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado paisaje_regularizacion_ridge.png")

# ==============================================================================
# MAIN ROUTINE
# ==============================================================================

def main():
    setup_thesis_style()
    output_dir = "output/figures_tfm/3.Desarrollo"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_plot_3_1_flowchart(output_dir)
    generate_plot_3_2_venn(output_dir)
    generate_plot_3_3_circuit(output_dir)
    generate_plot_3_4_regularization(output_dir)
    
    print("\nGeneración de figuras de la Sección 3 completada con éxito.")

if __name__ == "__main__":
    main()
