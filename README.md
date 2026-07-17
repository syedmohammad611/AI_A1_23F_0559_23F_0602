# 🧭 PathQuest — Uninformed Search Algorithm Visualizer

An interactive Python application that brings **classical AI search algorithms to life**. PathQuest animates six fundamental uninformed ("blind") search strategies as they navigate a randomly generated grid world from a start point to a goal while avoiding obstacles — showing you *exactly how each algorithm thinks*, step by step.

Rather than jumping straight to the answer, PathQuest renders the entire search process in real time: the frontier expanding, nodes being inspected, dead ends being abandoned, and the final path lighting up across the grid.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![NumPy](https://img.shields.io/badge/NumPy-Grid%20Engine-lightblue)
![License](https://img.shields.io/badge/Purpose-Educational-green)

---

## ✨ Features

- 🎮 **Interactive GUI** — one-click buttons to run any algorithm, reset the grid, or exit
- 🎞️ **Live step-by-step animation** — watch the search "flood" through the grid in real time
- 🧱 **Random obstacle generation** — every reset creates a fresh challenge
- 🎨 **Color-coded search states** — frontier, visited, current node, and final path are all visually distinct
- 📊 **Built-in benchmarking suite** — best-case and worst-case scenarios for every algorithm with path length, nodes explored, and execution time metrics
- ⚖️ **Cost-aware pathfinding** — diagonal moves cost √2, straight moves cost 1 (used by UCS)

## 🔍 Implemented Algorithms

| # | Algorithm | Strategy | Complete? | Optimal? |
|---|-----------|----------|-----------|----------|
| 1 | **BFS** — Breadth-First Search | FIFO queue, explores level by level | ✅ Yes | ✅ Yes (unit costs) |
| 2 | **DFS** — Depth-First Search | LIFO stack, dives deep before backtracking | ✅ Yes (finite grid) | ❌ No |
| 3 | **UCS** — Uniform-Cost Search | Priority queue ordered by path cost | ✅ Yes | ✅ Yes |
| 4 | **DLS** — Depth-Limited Search | DFS with a depth cutoff | ❌ No (if goal beyond limit) | ❌ No |
| 5 | **IDDFS** — Iterative Deepening DFS | Repeated DLS with increasing depth | ✅ Yes | ✅ Yes (unit costs) |
| 6 | **Bidirectional Search** | Simultaneous BFS from start *and* goal | ✅ Yes | ✅ Yes (unit costs) |

## 🧭 Movement Rules

All algorithms expand neighbors in a **strict clockwise order**:

```
1. Up ⬆️   2. Right ➡️   3. Down ⬇️   4. Down-Right ↘️   5. Left ⬅️   6. Up-Left ↖️
```

> Only the main diagonals (Down-Right and Up-Left) are allowed — Up-Right and Down-Left are excluded. Straight moves cost **1.0**, diagonal moves cost **1.414 (√2)**.

## 🎨 Visualization Legend

| Color | Meaning |
|-------|---------|
| 🟩 Green | Start point (`START`) |
| 🟥 Red | Goal point (`END`) |
| ⬛ Black | Obstacle / wall |
| 🔵 Light Blue | Frontier — nodes queued for exploration |
| ⚪ Gray | Visited — nodes already explored |
| 🟠 Orange | Current node being inspected |
| 🟨 Yellow | Final path from start to goal |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/syedmohammad611/PathQuest-Uninformed-Search-Algorithm-Visualizer.git
cd PathQuest-Uninformed-Search-Algorithm-Visualizer

# Install dependencies
pip install -r requirements.txt
```

### Run the Visualizer

```bash
python main.py
```

A window opens with a 15×15 grid and a control panel. Click any algorithm button (**BFS**, **DFS**, **UCS**, **DLS**, **IDDFS**, **Bidirectional**) to watch it search, **Reset Grid** to generate a new map, or **Exit** to close.

### Run the Benchmark Suite

```bash
python test_scenarios.py
```

This executes **12 curated scenarios** — a best case and a worst case for each of the six algorithms (serpentine mazes, walls forcing detours, goals beyond depth limits, etc.) — and prints a comparison table:

```
Algorithm            Scenario        Found    Path     Explored   Time (s)
--------------------------------------------------------------------------------
BFS                  BEST CASE       Yes      2        2          0.0312
BFS                  WORST CASE      Yes      43       197        1.2045
...
```

## 📁 Project Structure

```
├── main.py             # Entry point — launches the interactive GUI
├── grid.py             # NavigationGrid: environment, rendering & GUI controls
├── algorithms.py       # SearchTechniques: all six search implementations
├── utils.py            # SearchNode, priority queue, neighbor & cost helpers
├── test_scenarios.py   # Best/worst-case benchmark suite with metrics
└── requirements.txt    # Dependencies (matplotlib, numpy)
```

## 🧠 What This Project Demonstrates

- Core **AI search theory**: completeness, optimality, and time/space trade-offs of uninformed strategies
- **Data structures in practice**: FIFO queues (BFS), stacks (DFS), priority heaps (UCS), recursion with backtracking (DLS/IDDFS)
- **Bidirectional frontier meeting** and path reconstruction from two parent maps
- Clean, modular Python design with a clear separation between environment, algorithms, and visualization

## 👥 Authors

Developed as an Artificial Intelligence course project (Assignment 1).

- **23F-0559**
- **23F-0602**

---

⭐ *If you found this project helpful for learning search algorithms, consider giving it a star!*
