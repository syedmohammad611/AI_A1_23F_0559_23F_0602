# AI Pathfinding Visualizer

A Python-based pathfinding visualization tool that demonstrates various **uninformed search algorithms** in a grid environment. This project implements six fundamental "blind" search strategies that navigate from a Start Point (S) to a Target Point (T) while avoiding static obstacles.

The focus is not just on finding paths, but on **visualizing the search process** step-by-step, showing exactly how each algorithm "thinks" and explores the grid.

## 🎯 Project Overview

This AI Pathfinder visualizes how different uninformed search algorithms explore a map, demonstrating:
- Which nodes are checked first
- Which nodes are in the frontier (waiting to be explored)
- Which nodes have been visited
- The final path chosen by the algorithm

## 🔍 Implemented Algorithms

This project implements **ALL six** fundamental uninformed search algorithms:

1. **BFS** - Breadth-First Search
2. **DFS** - Depth-First Search
3. **UCS** - Uniform-Cost Search
4. **DLS** - Depth-Limited Search
5. **IDDFS** - Iterative Deepening Depth-First Search
6. **Bidirectional Search**

## 🧭 Movement Rules

All algorithms follow a **strict clockwise movement order** when expanding nodes:

1. **Up** ⬆️
2. **Right** ➡️
3. **Bottom** ⬇️
4. **Bottom-Right** ↘️ (Diagonal)
5. **Left** ⬅️
6. **Top-Left** ↖️ (Diagonal)

> **Note:** Only main diagonals (Bottom-Right and Top-Left) are considered. Top-Right and Bottom-Left diagonals are excluded.

## 🎨 GUI Visualization Features

The application uses **Matplotlib** to provide an interactive graphical interface with real-time visualization:

### Visual Elements:
- 🟩 **Start Point (S)** - Green cell with "START" label
- 🟥 **Target Point (T)** - Red cell with "END" label
- ⬛ **Obstacles** - Black cells (static walls)
- 🔵 **Frontier Nodes** - Light blue cells (nodes in queue/stack waiting to be explored)
- ⚪ **Explored Nodes** - Gray cells (nodes that have been visited)
- 🟠 **Current Node** - Orange cell (node being currently inspected)
- 🟨 **Final Path** - Yellow highlighted path from Start to Target

### Dynamic Animation:
- ✅ Step-by-step visualization with configurable delay
- ✅ Real-time updates showing the search "flooding" through the grid
- ✅ Interactive buttons for selecting different algorithms
- ✅ Grid reset functionality
- ✅ Color-coded legend for easy interpretation

The GUI does **not** simply jump to the result—it animates the entire search process so users can observe how each algorithm explores the grid differently.

## 📦 Installation

### Prerequisites
- Python 3.x
- pip package manager

### Setup

1. Clone the repository:
