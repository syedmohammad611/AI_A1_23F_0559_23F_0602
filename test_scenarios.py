"""
Test cases for pathfinding algorithms - Best and Worst Case Scenarios
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from grid import NavigationGrid
from algorithms import SearchTechniques
import time

class TestScenarios:
    """Creates specific grid configurations for testing algorithm performance."""
    
    def __init__(self):
        self.results = []
    
    def create_custom_grid(self, height, width, obstacles, start, goal):
        """Create a grid with specific configuration."""
        grid = NavigationGrid(height, width, obstacle_ratio=0)
        grid.matrix = np.zeros((height, width), dtype=int)
        
        # Set obstacles
        for obs in obstacles:
            grid.matrix[obs] = NavigationGrid.CELL_BLOCK
        
        # Set start and goal
        grid.start_pos = start
        grid.goal_pos = goal
        grid.matrix[start] = NavigationGrid.CELL_BEGIN
        grid.matrix[goal] = NavigationGrid.CELL_GOAL
        
        return grid
    
    def run_test(self, grid, algorithm_name, algorithm_func, scenario_type):
        """Execute a single test and collect metrics."""
        grid.clear_render_state()
        
        print(f"\n{'='*60}")
        print(f"Testing: {algorithm_name} - {scenario_type}")
        print(f"{'='*60}")
        print(f"Grid Size: {grid.rows}x{grid.cols}")
        print(f"Start: {grid.start_pos}, Goal: {grid.goal_pos}")
        
        start_time = time.time()
        path = algorithm_func()
        end_time = time.time()
        
        nodes_explored = len(grid.history)
        path_length = len(path) if path else 0
        execution_time = end_time - start_time
        
        result = {
            'algorithm': algorithm_name,
            'scenario': scenario_type,
            'path_found': path is not None,
            'path_length': path_length,
            'nodes_explored': nodes_explored,
            'execution_time': execution_time,
            'grid_size': f"{grid.rows}x{grid.cols}"
        }
        
        print(f"Path Found: {result['path_found']}")
        print(f"Path Length: {result['path_length']}")
        print(f"Nodes Explored: {result['nodes_explored']}")
        print(f"Execution Time: {execution_time:.4f} seconds")
        
        self.results.append(result)
        
        # Show the result
        grid.render_frame(f"{algorithm_name} - {scenario_type} (Complete)")
        plt.pause(2)
        
        return result
    
    # ========== BFS TEST SCENARIOS ==========
    def test_bfs_best_case(self):
        """BFS Best Case: Direct path with no obstacles."""
        grid = self.create_custom_grid(
            height=10, width=10,
            obstacles=[],
            start=(0, 0),
            goal=(0, 3)  # Goal is very close horizontally
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "BFS", engine.run_bfs, "BEST CASE")
    
    def test_bfs_worst_case(self):
        """BFS Worst Case: Goal is far with maze-like obstacles."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[
                # Create a maze forcing BFS to explore many nodes
                (1, i) for i in range(1, 14)
            ] + [
                (3, i) for i in range(0, 13)
            ] + [
                (5, i) for i in range(2, 15)
            ] + [
                (7, i) for i in range(0, 13)
            ] + [
                (9, i) for i in range(2, 15)
            ] + [
                (11, i) for i in range(0, 13)
            ],
            start=(0, 0),
            goal=(14, 14)  # Goal in opposite corner
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "BFS", engine.run_bfs, "WORST CASE")
    
    # ========== DFS TEST SCENARIOS ==========
    def test_dfs_best_case(self):
        """DFS Best Case: Direct path aligned with DFS's exploration order."""
        grid = self.create_custom_grid(
            height=10, width=10,
            obstacles=[],
            start=(0, 0),
            goal=(3, 0)  # Goal is down (DFS explores down-right first)
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "DFS", engine.run_dfs, "BEST CASE")
    
    def test_dfs_worst_case(self):
        """DFS Worst Case: Goal requires backtracking through wrong branches."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[
                # Block right and down paths, forcing DFS into wrong branches
                (0, i) for i in range(2, 15)
            ] + [
                (i, 1) for i in range(1, 14)
            ] + [
                (14, i) for i in range(1, 13)
            ],
            start=(0, 0),
            goal=(14, 14)
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "DFS", engine.run_dfs, "WORST CASE")
    
    # ========== UCS TEST SCENARIOS ==========
    def test_ucs_best_case(self):
        """UCS Best Case: Straight line to goal."""
        grid = self.create_custom_grid(
            height=10, width=10,
            obstacles=[],
            start=(5, 0),
            goal=(5, 4)  # Horizontal line
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "UCS", engine.run_ucs, "BEST CASE")
    
    def test_ucs_worst_case(self):
        """UCS Worst Case: Many equal-cost paths, explores most of grid."""
        grid = self.create_custom_grid(
            height=12, width=12,
            obstacles=[
                # Scattered obstacles creating multiple paths
                (2, 2), (2, 5), (2, 8),
                (5, 1), (5, 4), (5, 7), (5, 10),
                (8, 2), (8, 5), (8, 8),
            ],
            start=(0, 0),
            goal=(11, 11)
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "UCS", engine.run_ucs, "WORST CASE")
    
    # ========== DLS TEST SCENARIOS ==========
    def test_dls_best_case(self):
        """DLS Best Case: Goal within depth limit and easily found."""
        grid = self.create_custom_grid(
            height=10, width=10,
            obstacles=[],
            start=(5, 5),
            goal=(7, 5)  # Close goal, depth 2
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        # Use depth limit of 5
        def run_dls_5():
            return engine.run_dls(depth_cap=5)
        return self.run_test(grid, "DLS (depth=5)", run_dls_5, "BEST CASE")
    
    def test_dls_worst_case(self):
        """DLS Worst Case: Goal just beyond depth limit."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[],
            start=(0, 0),
            goal=(10, 10)  # Manhattan distance = 20, beyond typical depth limit
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        # Use insufficient depth limit
        def run_dls_8():
            return engine.run_dls(depth_cap=8)  # Too shallow!
        return self.run_test(grid, "DLS (depth=8)", run_dls_8, "WORST CASE")
    
    # ========== IDDFS TEST SCENARIOS ==========
    def test_iddfs_best_case(self):
        """IDDFS Best Case: Goal found at shallow depth."""
        grid = self.create_custom_grid(
            height=10, width=10,
            obstacles=[],
            start=(5, 5),
            goal=(6, 6)  # Diagonal neighbor, depth 2
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        def run_iddfs_10():
            return engine.run_iddfs(upper_limit=10)
        return self.run_test(grid, "IDDFS", run_iddfs_10, "BEST CASE")
    
    def test_iddfs_worst_case(self):
        """IDDFS Worst Case: Goal at maximum depth, many iterations."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[
                # Force a long winding path
                (i, 5) for i in range(0, 12)
            ] + [
                (12, i) for i in range(5, 15)
            ],
            start=(0, 0),
            goal=(14, 14)
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        def run_iddfs_25():
            return engine.run_iddfs(upper_limit=25)
        return self.run_test(grid, "IDDFS", run_iddfs_25, "WORST CASE")
    
    # ========== BIDIRECTIONAL TEST SCENARIOS ==========
    def test_bidirectional_best_case(self):
        """Bidirectional Best Case: Clear path, searches meet quickly."""
        grid = self.create_custom_grid(
            height=10, width=10,
            obstacles=[],
            start=(0, 0),
            goal=(9, 9)  # Diagonal, but searches meet in middle
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "Bidirectional", engine.run_bi_search, "BEST CASE")
    
    def test_bidirectional_worst_case(self):
        """Bidirectional Worst Case: Wall between start/goal, searches miss."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[
                # Vertical wall with small opening at bottom
                (i, 7) for i in range(0, 13)
            ],
            start=(0, 0),
            goal=(0, 14)  # Opposite sides of wall
        )
        engine = SearchTechniques(grid, wait_time=0.01)
        return self.run_test(grid, "Bidirectional", engine.run_bi_search, "WORST CASE")
    
    def print_summary(self):
        """Print summary of all test results."""
        print("\n" + "="*80)
        print("TEST SUMMARY - ALL ALGORITHMS")
        print("="*80)
        print(f"{'Algorithm':<20} {'Scenario':<15} {'Found':<8} {'Path':<8} {'Explored':<10} {'Time (s)':<10}")
        print("-"*80)
        
        for result in self.results:
            print(f"{result['algorithm']:<20} {result['scenario']:<15} "
                  f"{'Yes' if result['path_found'] else 'No':<8} "
                  f"{result['path_length']:<8} {result['nodes_explored']:<10} "
                  f"{result['execution_time']:.4f}")
        print("="*80)
    
    def run_all_tests(self):
        """Execute all test scenarios."""
        print("\n" + "🚀 STARTING COMPREHENSIVE ALGORITHM TESTING 🚀\n")
        
        # BFS Tests
        self.test_bfs_best_case()
        self.test_bfs_worst_case()
        
        # DFS Tests
        self.test_dfs_best_case()
        self.test_dfs_worst_case()
        
        # UCS Tests
        self.test_ucs_best_case()
        self.test_ucs_worst_case()
        
        # DLS Tests
        self.test_dls_best_case()
        self.test_dls_worst_case()
        
        # IDDFS Tests
        self.test_iddfs_best_case()
        self.test_iddfs_worst_case()
        
        # Bidirectional Tests
        self.test_bidirectional_best_case()
        self.test_bidirectional_worst_case()
        
        # Print summary
        self.print_summary()
        
        print("\n✅ All tests completed! Close the window to exit.")
        plt.show()

if __name__ == "__main__":
    tester = TestScenarios()
    tester.run_all_tests()