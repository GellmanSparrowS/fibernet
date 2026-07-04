"""
Example 18: Advanced 3D Visualization with PyVista

Demonstrates PyVista-based visualization for fiber networks:
- Interactive 3D rendering
- Color coding by properties (length, orientation, radius)
- Screenshots with high resolution
- VTK export for external visualization
- Cross-section views
- Rotation animations

Usage:
    python examples/18_pyvista_visualization.py

Note:
    Requires pyvista: pip install pyvista
    In headless environments, set PYVISTA_OFF_SCREEN=true
"""

import numpy as np
from fibernet import gen
from fibernet.gen.laminates import crossply_laminate
from fibernet.core.material import Material

# Check if PyVista is available
try:
    from fibernet.pyvista_viz import PyVistaVisualizer, PYVISTA_AVAILABLE
    
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Install with: pip install pyvista")
        exit(1)
    
    print("=" * 70)
    print("  FiberNet - PyVista Visualization Examples")
    print("=" * 70)
    
    # 1. Random 3D Network
    print("\n[1/5] Random 3D Network")
    print("-" * 70)
    
    net_random = gen.random_straight_3d(
        num_fibers=100,
        fiber_length=20.0,
        box_size=(50, 50, 50),
        seed=42
    )
    
    viz_random = PyVistaVisualizer(net_random)
    print(f"  Fibers: {len(net_random.fibers)}")
    print(f"  Mesh created: {viz_random.mesh.n_points} points, {viz_random.mesh.n_cells} cells")
    
    # Save screenshot
    viz_random.save_screenshot(
        'random_network.png',
        color='lightblue',
        window_size=(1024, 768)
    )
    print("  Saved: random_network.png")
    
    # 2. Cross-ply Laminate with Color Coding
    print("\n[2/5] Cross-ply Laminate - Color by Length")
    print("-" * 70)
    
    net_laminate = crossply_laminate(
        num_layers=4,
        fibers_per_layer=20,
        layer_thickness=0.5,
        fiber_length=30.0,
        seed=42
    )
    
    viz_laminate = PyVistaVisualizer(net_laminate)
    viz_laminate.color_by_property('length', colormap='viridis')
    
    print(f"  Fibers: {len(net_laminate.fibers)}")
    print(f"  Colored by: length")
    
    viz_laminate.save_screenshot(
        'laminate_colored.png',
        window_size=(1200, 800)
    )
    print("  Saved: laminate_colored.png")
    
    # 3. Color by Orientation
    print("\n[3/5] Random Network - Color by Orientation")
    print("-" * 70)
    
    viz_orient = PyVistaVisualizer(net_random)
    viz_orient.color_by_property('orientation', colormap='hsv')
    
    print(f"  Colored by: orientation angle (0-90°)")
    
    viz_orient.save_screenshot(
        'network_orientation.png',
        window_size=(1200, 800)
    )
    print("  Saved: network_orientation.png")
    
    # 4. Export to VTK
    print("\n[4/5] VTK Export")
    print("-" * 70)
    
    viz_random.export_vtk('random_network.vtk')
    print("  Exported: random_network.vtk")
    print("  Can be opened in ParaView, VisIt, or other VTK-compatible software")
    
    # 5. Multiple Properties
    print("\n[5/5] Color by Radius")
    print("-" * 70)
    
    # Create network with varying radii
    # Use existing random network for radius coloring
    viz_radius = PyVistaVisualizer(net_random)
    viz_radius.color_by_property('radius', colormap='plasma')

    print(f"  Fibers: {len(net_random.fibers)}")
    print(f"  Colored by: radius (uniform in this example)")

    viz_radius.save_screenshot(
        'network_radius.png',
        window_size=(1200, 800)
    )
    print("  Saved: network_radius.png")
    
    print("\n" + "=" * 70)
    print("  PyVista Visualization Examples Complete!")
    print("=" * 70)
    print("""
Generated files:
  - random_network.png
  - laminate_colored.png
  - network_orientation.png
  - random_network.vtk
  - network_radius.png

Advanced features (not shown in headless mode):
  - Interactive 3D rotation and zoom
  - Real-time property updates
  - Animation creation
  - Cross-section planes
  - Multiple viewports

For interactive visualization:
  viz.show()  # Opens interactive window
  viz.show_jupyter()  # For Jupyter notebooks

References:
  - PyVista: https://docs.pyvista.org/
""")

except ImportError as e:
    print(f"PyVista not installed: {e}")
    print("Install with: pip install pyvista")
