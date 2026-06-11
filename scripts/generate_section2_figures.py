import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def generate_barren_plateaus(output_dir):
    print("Generando barren_plateaus.png...")
    fig = plt.figure(figsize=(12, 5.5))
    
    # Grid of data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # 1. Optimizable Landscape (Convex paraboloid)
    Z1 = X**2 + Y**2
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none', alpha=0.95, antialiased=True)
    ax1.set_title("A. Paisaje Optimizable", weight='bold', pad=15)
    
    # Clean axes
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.view_init(elev=25, azim=45)
    
    # 2. Barren Plateau Landscape (Flat plane with a narrow Gaussian well)
    # Z2 = -exp(-(x^2+y^2)/0.15)
    Z2 = -np.exp(-(X**2 + Y**2) / 0.15)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis', edgecolor='none', alpha=0.95, antialiased=True)
    ax2.set_title("B. Meseta Estéril / Barren Plateau", weight='bold', pad=15)
    ax2.set_zlim(-1.1, 0.1)
    
    # Clean axes
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.view_init(elev=25, azim=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "barren_plateaus.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado barren_plateaus.png")

def generate_xy_vs_qaoa_feasibility(output_dir):
    print("Generando xy_vs_qaoa_feasibility.png...")
    fig = plt.figure(figsize=(12, 5.5))
    
    # 8 Boolean hypercube vertices for N=3 qubits
    vertices = {
        (0, 0, 0): "000",
        (1, 0, 0): "100",
        (0, 1, 0): "010",
        (0, 0, 1): "001",
        (1, 1, 0): "110",
        (1, 0, 1): "101",
        (0, 1, 1): "011",
        (1, 1, 1): "111"
    }
    
    # Valid vertices (K=1 active qubits)
    valid_vertices = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    
    # Hypercube edges (pairs differing by exactly 1 bit)
    edges = [
        ((0,0,0), (1,0,0)), ((0,0,0), (0,1,0)), ((0,0,0), (0,0,1)),
        ((1,0,0), (1,1,0)), ((1,0,0), (1,0,1)),
        ((0,1,0), (1,1,0)), ((0,1,0), (0,1,1)),
        ((0,0,1), (1,0,1)), ((0,0,1), (0,1,1)),
        ((1,1,1), (1,1,0)), ((1,1,1), (1,0,1)), ((1,1,1), (0,1,1))
    ]
    
    # Subplot 1: Standard QAOA (Full Hilbert space 2^N)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("A. QAOA Estándar: Espacio Completo $2^N$", weight='bold', pad=15)
    
    # Draw edges
    for edge in edges:
        p1, p2 = edge
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#CBD5E1', linestyle='--', linewidth=1.2)
        
    # Draw vertices
    for vertex, label in vertices.items():
        if vertex in valid_vertices:
            # Valid K=1 state
            ax1.scatter(vertex[0], vertex[1], vertex[2], color='#10B981', s=150, depthshade=False, edgecolor='black', zorder=10)
            ax1.text(vertex[0] + 0.05, vertex[1] + 0.05, vertex[2] + 0.05, f"$|{label}\\rangle$", color='#047857', weight='bold', fontsize=9)
        else:
            # Invalid state
            ax1.scatter(vertex[0], vertex[1], vertex[2], color='#EF4444', s=90, depthshade=False, alpha=0.6, zorder=5)
            ax1.text(vertex[0] + 0.05, vertex[1] + 0.05, vertex[2] + 0.05, f"$|{label}\\rangle$", color='#B91C1C', alpha=0.7, fontsize=8)
            
    # Set display properties
    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=30)
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_zlim(-0.2, 1.2)
    
    # Subplot 2: XY-QAOA (Dicke Subspace K=1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title(r"B. XY-QAOA: Subespacio de Dicke $\binom{N}{K}$", weight='bold', pad=15)
    
    # Draw triangular plane for K=1 (100, 010, 001)
    verts = [valid_vertices]
    poly = Poly3DCollection(verts, alpha=0.25, facecolor='#10B981', edgecolor='#059669', linewidth=1.5)
    ax2.add_collection3d(poly)
    
    # Draw valid vertices
    for vertex in valid_vertices:
        label = vertices[vertex]
        ax2.scatter(vertex[0], vertex[1], vertex[2], color='#10B981', s=150, depthshade=False, edgecolor='black', zorder=10)
        ax2.text(vertex[0] + 0.05, vertex[1] + 0.05, vertex[2] + 0.05, f"$|{label}\\rangle$", color='#047857', weight='bold', fontsize=9)
        
    # Draw standard grid references for comparison (very faint edges)
    for edge in edges:
        p1, p2 = edge
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#CBD5E1', linestyle=':', linewidth=0.8, alpha=0.4)
        
    # Set display properties
    ax2.set_axis_off()
    ax2.view_init(elev=20, azim=30)
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_zlim(-0.2, 1.2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xy_vs_qaoa_feasibility.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado xy_vs_qaoa_feasibility.png")

def main():
    setup_thesis_style()
    output_dir = "output/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_barren_plateaus(output_dir)
    generate_xy_vs_qaoa_feasibility(output_dir)
    print("\nTodo completado de manera óptima.")

if __name__ == "__main__":
    main()
