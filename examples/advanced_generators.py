"""
Advanced generators example - demonstrating complex fiber network structures.
"""
import sys
sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')

from fibernet.gen import advanced, variants
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
from fibernet.analysis.advanced import SpectralAnalyzer, PoreAnalyzer

def main():
    print("=" * 60)
    print("Advanced Generators Example")
    print("=" * 60)
    
    # 1. Voronoi-based networks
    print("\n1. Voronoi Network (2D cellular structure)")
    voronoi_2d = advanced.voronoi_network_2d(
        num_seeds=50, box_size=(40, 40), seed=42,
    )
    print(f"   Fibers: {voronoi_2d.num_fibers}, Crosslinks: {voronoi_2d.num_crosslinks}")
    
    # 2. Electrospun networks
    print("\n2. Electrospun Network (nanofiber mat)")
    electrospun = advanced.electrospun_network(
        num_fibers=200, fiber_length=30, box_size=(60, 60),
        radius_mean=0.2, waviness=0.3, seed=42,
    )
    print(f"   Fibers: {electrospun.num_fibers}")
    
    # 3. Biomimetic collagen
    print("\n3. Biomimetic Collagen Network")
    collagen = advanced.biomimetic_collagen(
        num_fibers=100, box_size=(50, 50, 20),
        persistence_length=15, bundling_probability=0.3, seed=42,
    )
    print(f"   Fibers: {collagen.num_fibers}, Dimension: {collagen.dimension}D")
    
    # 4. Auxetic structure
    print("\n4. Auxetic Structure (re-entrant honeycomb)")
    auxetic = advanced.auxetic_structure(
        reentrant_angle=0.8, cell_size=5, grid_size=(4, 4),
    )
    print(f"   Fibers: {auxetic.num_fibers}")
    
    # 5. Diamond lattice
    print("\n5. Diamond Lattice (3D)")
    diamond = variants.diamond_lattice_3d(spacing=5, grid_size=(2, 2, 2))
    print(f"   Fibers: {diamond.num_fibers}, Dimension: {diamond.dimension}D")
    
    # 6. Foam-like structure
    print("\n6. Foam-like 3D Structure")
    foam = variants.foam_like_3d(
        box_size=(20, 20, 20), num_cells=30,
        strut_curvature=0.1, seed=42,
    )
    print(f"   Fibers: {foam.num_fibers}")
    
    print("\n" + "=" * 60)
    print("Structural Analysis")
    print("=" * 60)
    
    # Spectral analysis
    print("\nSpectral Analysis (electrospun network):")
    spectral = SpectralAnalyzer(electrospun)
    gap = spectral.spectral_gap()
    entropy = spectral.spectral_entropy()
    print(f"   Spectral gap: {gap:.4f}")
    print(f"   Spectral entropy: {entropy:.4f}")
    
    # Pore analysis
    print("\nPore Size Analysis (electrospun network):")
    pore = PoreAnalyzer(electrospun)
    pore_stats = pore.pore_size_statistics()
    print(f"   Mean pore size: {pore_stats['mean']:.3f}")
    print(f"   Median pore size: {pore_stats['median']:.3f}")
    
    print("\n✓ Advanced generators example complete")

if __name__ == "__main__":
    main()
