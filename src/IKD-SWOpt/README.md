# IKD-SWOpt

A planning algorithm suite for autonomous robots, featuring **Incremental KD-Tree with Sliding Window Optimization (IKD-SWOpt)**. This repository implements algorithms for pathfinding, coverage planning, and trajectory optimization as described in the accompanying research paper.

## Table of Contents

- [IKD-SWOpt](#ikd-swopt)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
    - [Running the Planner](#running-the-planner)
    - [Configuration](#configuration)
  - [Directory Structure](#directory-structure)
  - [License](#license)
  - [References](#references)

## Overview

This repository contains the implementation of **IKD-SWOpt**, an innovative planning framework that integrates:

1. **Incremental KD-Tree Construction**: Efficiently manages dynamic obstacles for high-dimensional environments.
2. **Sliding Window Optimization**: Optimizes robot trajectories in real-time using B-splines and L-BFGS.
3. **Astar Search with Potential Fields**: Incorporates heuristic-driven pathfinding enhanced by potential fields.

These algorithms are benchmarked using randomly generated environments, supporting both 2D and 3D robot applications.

## Features

- KD-Tree-based dynamic obstacle handling
- High-performance A* search with potential field integration
- B-spline smoothing for optimized trajectories
- Random environment generation with customizable density
- Visualizations for planning and trajectory evaluation

## Installation

### Prerequisites

- C++ compiler (GCC 9.0 or higher recommended)
- CMake (version 3.10 or later)
- Required libraries:
  - OpenCV (for visualization)
  - Eigen3 (for matrix operations)
  - lbfgs.hpp (included for optimization)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/IKD-SWOpt.git
   cd IKD-SWOpt
   ```

2. Build the project:

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the executable:

   ```bash
   ./IKD-SWOpt
   ```

## Usage

### Running the Planner

After building the project, run the `IKD-SWOpt` executable to test the planning framework. The default configuration generates random environments, performs pathfinding, and optimizes trajectories. Outputs include:

- Visualization windows for paths and trajectories.
- Saved images for each test case in the working directory.

### Configuration

Modify parameters in `IKD-SWOpt.cpp`:

- **Map Dimensions**: `width` and `height`.
- **Obstacle Density**: `obstacleDensity`.
- **Number of Tests**: `numTests`.
- **Optimization Weights**: `obs_weight`, `smooth_weight`, and `length_weight`.

## Directory Structure

```plaintext
IKD-SWOpt/
│
├── README.md                   # Project documentation
├── LICENSE                     # License file
├── requirements.txt            # Python dependencies (optional for additional tools)
├── CMakeLists.txt              # CMake configuration
│
├── src/
│   ├── IKD-SWOpt.cpp           # Main executable
│   ├── ikd_Tree.cpp            # KD-Tree implementation
│   ├── lbfgs.hpp               # Optimization library
│
├── include/
│   ├── ikd_Tree.h              # KD-Tree header
│
├── docs/
│   ├── IKD-SWOpt-Paper.pdf     # Research paper (placeholder)
│   ├── images/                 # Figures and visualizations
│
└── build/                      # Build output (ignored in version control)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- **Research Paper**: "Efficient Planning Algorithms for Autonomous Robots using IKD-SWOpt" (linked as [IKD-SWOpt-Paper.pdf](docs/IKD-SWOpt-Paper.pdf)).
- **Dependencies**:
  - OpenCV: [https://opencv.org/](https://opencv.org/)
  - Eigen3: [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)
  - lbfgs.hpp: [https://github.com/ZJU-FAST-Lab/LBFGS-Lite](https://github.com/ZJU-FAST-Lab/LBFGS-Lite)

.
