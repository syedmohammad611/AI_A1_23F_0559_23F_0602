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
        grid = NavigationGrid(height, width, block_rate=0)
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
        plt.pause(1) 
        
        return result
    
    # ========== BFS TEST SCENARIOS ==========
    def test_bfs_best_case(self):
        """BFS Best Case: Goal is immediate neighbor (1 step away)."""
        grid = self.create_custom_grid(
            height=8, width=8,
            obstacles=[],
            start=(4, 4),
            goal=(3, 4)  # Directly above - first direction checked (Up)
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        return self.run_test(grid, "BFS", engine.run_bfs, "BEST CASE")
    
    def test_bfs_worst_case(self):
        """BFS Worst Case: Goal far away with serpentine maze."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[
                # Zigzag maze forcing exploration of entire grid
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
            goal=(14, 14)
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        return self.run_test(grid, "BFS", engine.run_bfs, "WORST CASE")
    
    # ========== DFS TEST SCENARIOS ==========
    def test_dfs_best_case(self):
        """DFS Best Case: Goal aligned with first exploration direction (Up)."""
        grid = self.create_custom_grid(
            height=8, width=8,
            obstacles=[],
            start=(5, 4),
            goal=(3, 4)  # Straight up - DFS checks Up first
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        return self.run_test(grid, "DFS", engine.run_dfs, "BEST CASE")
    
    def test_dfs_worst_case(self):
        """DFS Worst Case: Goal requires full exploration and backtracking."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[
                # Force DFS into deep wrong branches
                (0, i) for i in range(2, 15)
            ] + [
                (i, 1) for i in range(1, 14)
            ] + [
                (14, i) for i in range(2, 14)
            ],
            start=(0, 0),
            goal=(14, 14)
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        return self.run_test(grid, "DFS", engine.run_dfs, "WORST CASE")
    
    # ========== UCS TEST SCENARIOS ==========
    def test_ucs_best_case(self):
        """UCS Best Case: Direct adjacent goal, minimal cost."""
        grid = self.create_custom_grid(
            height=8, width=8,
            obstacles=[],
            start=(4, 4),
            goal=(4, 5)  # Right neighbor (cost = 1.0)
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        return self.run_test(grid, "UCS", engine.run_ucs, "BEST CASE")
    
    def test_ucs_worst_case(self):
        """UCS Worst Case: Multiple equal-cost paths."""
        grid = self.create_custom_grid(
            height=12, width=12,
            obstacles=[
                # Scattered obstacles creating many alternative paths
                (2, 2), (2, 5), (2, 8),
                (5, 1), (5, 4), (5, 7), (5, 10),
                (8, 2), (8, 5), (8, 8),
            ],
            start=(0, 0),
            goal=(11, 11)
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        return self.run_test(grid, "UCS", engine.run_ucs, "WORST CASE")
    
    # ========== DLS TEST SCENARIOS ==========
    def test_dls_best_case(self):
        """DLS Best Case: Goal at depth 1 (immediate neighbor)."""
        grid = self.create_custom_grid(
            height=8, width=8,
            obstacles=[],
            start=(4, 4),
            goal=(3, 4)  # Depth 1 - immediate neighbor
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        def run_dls_3():
            return engine.run_dls(depth_cap=3)
        return self.run_test(grid, "DLS (depth=3)", run_dls_3, "BEST CASE")
    
    def test_dls_worst_case(self):
        """DLS Worst Case: Goal beyond depth limit (unreachable)."""
        grid = self.create_custom_grid(
            height=12, width=12,
            obstacles=[],
            start=(0, 0),
            goal=(10, 10)  # Manhattan distance = 20, needs depth > 10
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        def run_dls_5():
            return engine.run_dls(depth_cap=5)  # Too shallow - fails
        return self.run_test(grid, "DLS (depth=5)", run_dls_5, "WORST CASE")
    
    # ========== IDDFS TEST SCENARIOS ==========
    def test_iddfs_best_case(self):
        """IDDFS Best Case: Goal at depth 1 (found in first iteration)."""
        grid = self.create_custom_grid(
            height=8, width=8,
            obstacles=[],
            start=(4, 4),
            goal=(3, 4)  # Immediate neighbor, depth 1
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        def run_iddfs_5():
            return engine.run_iddfs(upper_limit=5)
        return self.run_test(grid, "IDDFS", run_iddfs_5, "BEST CASE")
    
    def test_iddfs_worst_case(self):
        """IDDFS Worst Case: Goal at maximum depth with obstacles."""
        grid = self.create_custom_grid(
            height=12, width=12,
            obstacles=[
                # Create winding path requiring many iterations
                (i, 5) for i in range(1, 10)
            ] + [
                (10, i) for i in range(5, 12)
            ],
            start=(0, 0),
            goal=(11, 11)
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        def run_iddfs_20():
            return engine.run_iddfs(upper_limit=20)
        return self.run_test(grid, "IDDFS", run_iddfs_20, "WORST CASE")
    
    # ========== BIDIRECTIONAL TEST SCENARIOS ==========
    def test_bidirectional_best_case(self):
        """Bidirectional Best Case: Short straight path, quick meeting."""
        grid = self.create_custom_grid(
            height=8, width=8,
            obstacles=[],
            start=(4, 2),
            goal=(4, 6)  # Same row, 4 steps apart
        )
        engine = SearchTechniques(grid, wait_time=0.005)
        return self.run_test(grid, "Bidirectional", engine.run_bi_search, "BEST CASE")
    
    def test_bidirectional_worst_case(self):
        """Bidirectional Worst Case: Wall forces long detour."""
        grid = self.create_custom_grid(
            height=15, width=15,
            obstacles=[
                # Vertical wall with opening at bottom
                (i, 7) for i in range(0, 13)
            ],
            start=(0, 0),
            goal=(0, 14)
        )
        engine = SearchTechniques(grid, wait_time=0.005)
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