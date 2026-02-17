import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from grid import NavigationGrid
from algorithms import SearchTechniques
import sys

def launch_graphic_interface():
    """Starts the pathfinding application with a graphical menu."""
    print("\n" + "="*40)
    print(" AI PATHFINDER: GRAPHICAL INTERFACE")
    print("="*40)
    print("Opening visualization window...")

    # Default parameters for the interface
    height, width = 15, 15
    obstacle_ratio = 0.2
    exec_speed = 0.05

    # Initialize components
    env_map = NavigationGrid(height, width, obstacle_ratio)
    engine = SearchTechniques(env_map, exec_speed)

    # Setup the graphical buttons and initial view
    env_map.render_frame("Pathfinding Menu")
    env_map.setup_gui_controls(engine)

    # Display status info in terminal
    print("Ready. Use the buttons in the window to select algorithms.")
    print("Close the window to exit.")

    # Enter Matplotlib event loop
    plt.show()

if __name__ == "__main__":
    try:
        launch_graphic_interface()
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
        sys.exit(0)