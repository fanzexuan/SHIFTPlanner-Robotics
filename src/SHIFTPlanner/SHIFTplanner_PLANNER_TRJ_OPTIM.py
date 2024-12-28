import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

# =========================
# Path Planning and Optimization Module
# =========================

def generate_obstacles(grid, num_obstacles=300, min_size=2, max_size=5):
    """
    Randomly generate rectangular obstacles on the grid map.

    Parameters:
    - grid: 2D numpy array representing the map grid.
    - num_obstacles: Number of obstacles to generate.
    - min_size: Minimum size of obstacles.
    - max_size: Maximum size of obstacles.
    """
    for _ in range(num_obstacles):
        width = random.randint(min_size, max_size)
        height = random.randint(min_size, max_size)
        if grid.shape[0] - height <= 0 or grid.shape[1] - width <= 0:
            continue
        x = random.randint(0, grid.shape[0] - height - 1)
        y = random.randint(0, grid.shape[1] - width - 1)
        grid[x:x+height, y:y+width] = 1  # Mark obstacle cells with 1

def calculate_distance_field_kdtree(grid):
    """
    Calculate the Euclidean distance from each free space cell to the nearest obstacle using KD-Tree.

    Parameters:
    - grid: 2D numpy array representing the map grid.

    Returns:
    - distance_field: 2D numpy array with distance values.
    """
    grid_size = grid.shape
    # Get obstacle coordinates
    obstacle_coords = np.argwhere(grid == 1)
    # Get all grid coordinates
    all_coords = np.indices(grid_size).reshape(2, -1).T

    # Build KD-Tree with obstacle coordinates
    if len(obstacle_coords) > 0:
        kdtree = KDTree(obstacle_coords)
        # Query KD-Tree for nearest obstacle distances to each grid point
        distances, _ = kdtree.query(all_coords)
        # Reshape distances back to grid shape
        distance_field = distances.reshape(grid_size)
    else:
        # If no obstacles, distance is infinity
        distance_field = np.full(grid_size, np.inf)
    return distance_field

def build_kdtree(grid):
    """
    Build a KD-Tree of obstacle coordinates for efficient nearest neighbor queries.

    Parameters:
    - grid: 2D numpy array representing the map grid.

    Returns:
    - kdtree: KDTree object of obstacle coordinates.
    - obstacle_coords: Numpy array of obstacle coordinates.
    """
    obstacle_coords = np.argwhere(grid == 1)
    if len(obstacle_coords) > 0:
        kdtree = KDTree(obstacle_coords)
    else:
        kdtree = None
    return kdtree, obstacle_coords

def heuristic(a, b):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    - a: Tuple (x, y) for point a.
    - b: Tuple (x, y) for point b.

    Returns:
    - Euclidean distance between a and b.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star_search_with_potential_field(start, goal, grid, distance_field, influence_strength=1.0):
    """
    Perform A* search with a potential field to encourage paths away from obstacles.

    Parameters:
    - start: Starting point (x, y).
    - goal: Goal point (x, y).
    - grid: 2D numpy array representing the map grid.
    - distance_field: 2D numpy array with distance values.
    - influence_strength: Weight of the potential field influence.

    Returns:
    - path: List of tuples representing the path from start to goal.
    """
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-connected grid
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    grid_size = grid.shape

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            # Reconstruct path
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            data.reverse()
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]):
                continue
            if grid[neighbor[0], neighbor[1]] == 1:
                continue
            if neighbor in close_set:
                continue

            tentative_g_score = gscore[current] + np.hypot(i, j)

            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                potential = influence_strength / (distance_field[neighbor[0], neighbor[1]] + 1e-5)
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal) + potential
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False  # Path not found

def compute_total_cost(variables, num_points, start, goal, kdtree, grid, obs_weight=1.0, smooth_weight=0.1, length_weight=0.1):
    """
    Compute the total cost for path optimization, including smoothness, obstacle avoidance, and path length.

    Parameters:
    - variables: Flattened list of x and y coordinates for intermediate points.
    - num_points: Total number of points in the path.
    - start: Starting point coordinates.
    - goal: Goal point coordinates.
    - kdtree: KDTree object of obstacle coordinates.
    - grid: 2D numpy array representing the map grid.
    - obs_weight: Weight for obstacle avoidance cost.
    - smooth_weight: Weight for smoothness cost.
    - length_weight: Weight for path length cost.

    Returns:
    - total_cost: Total cost value.
    """
    # Reconstruct the path with start and goal
    path_points = [np.array(start)]
    for i in range(num_points - 2):
        x = variables[2 * i]
        y = variables[2 * i + 1]
        path_points.append(np.array([x, y]))
    path_points.append(np.array(goal))

    total_cost = 0.0
    grid_size = grid.shape

    # Smoothness cost
    for i in range(1, num_points - 1):
        xi_prev = path_points[i - 1]
        xi = path_points[i]
        xi_next = path_points[i + 1]
        smoothness = np.linalg.norm(xi_prev - 2 * xi + xi_next) ** 2
        total_cost += smooth_weight * smoothness

    # Obstacle avoidance cost using KD-Tree for distance calculation
    for i in range(1, num_points - 1):
        xi = path_points[i]
        xi_int = (int(round(xi[0])), int(round(xi[1])))
        # Check if within grid bounds
        if 0 <= xi_int[0] < grid_size[0] and 0 <= xi_int[1] < grid_size[1]:
            # Check if inside an obstacle
            if grid[xi_int[0], xi_int[1]] == 1:
                total_cost += 1e6  # High cost for being inside an obstacle
            else:
                # Calculate distance to nearest obstacle using KD-Tree
                dist, _ = kdtree.query(xi)
                # Use an exponential function to penalize proximity to obstacles
                obs_cost = obs_weight * np.exp(-dist)
                total_cost += obs_cost
        else:
            total_cost += 1e6  # High cost for being outside the grid

    # Path length cost (encourage shorter paths)
    for i in range(num_points - 1):
        xi = path_points[i]
        xi_next = path_points[i + 1]
        length = np.linalg.norm(xi_next - xi)
        total_cost += length_weight * length

    return total_cost

def optimize_path_with_scipy(path, grid, kdtree, obs_weight=1.0, smooth_weight=0.1, length_weight=0.1, max_iterations=100):
    """
    Optimize the path using scipy.optimize.minimize with L-BFGS-B method.

    Parameters:
    - path: Initial path as a list of tuples.
    - grid: 2D numpy array representing the map grid.
    - kdtree: KDTree object of obstacle coordinates.
    - obs_weight: Weight for obstacle avoidance cost.
    - smooth_weight: Weight for smoothness cost.
    - length_weight: Weight for path length cost.
    - max_iterations: Maximum number of iterations for the optimizer.

    Returns:
    - optimized_path: Optimized path as a list of tuples.
    - path_history: List of paths during optimization for animation.
    """
    num_points = len(path)
    grid_size = grid.shape
    # Initial variables (excluding start and goal)
    initial_variables = []
    for i in range(1, num_points - 1):
        initial_variables.extend(path[i])

    # Bounds for variables
    bounds = []
    for _ in range(num_points - 2):
        bounds.append((0, grid_size[0] - 1))  # x bounds
        bounds.append((0, grid.shape[1] - 1))  # y bounds

    # Objective function
    def objective(variables):
        return compute_total_cost(variables, num_points, path[0], path[-1], kdtree, grid, obs_weight, smooth_weight, length_weight)

    # To store path history
    path_history = []

    # Callback function to record path at each iteration
    def callback(variables):
        optimized_path = [path[0]]
        for i in range(num_points - 2):
            x = variables[2 * i]
            y = variables[2 * i + 1]
            optimized_path.append((x, y))
        optimized_path.append(path[-1])
        path_history.append(optimized_path.copy())

    # Optimize
    result = minimize(objective, initial_variables, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iterations, 'disp': True}, callback=callback)

    # Check if optimization was successful
    if not result.success:
        print("Optimization failed:", result.message)

    # Reconstruct the optimized path
    optimized_path = [path[0]]
    optimized_variables = result.x
    for i in range(num_points - 2):
        x = optimized_variables[2 * i]
        y = optimized_variables[2 * i + 1]
        optimized_path.append((x, y))
    optimized_path.append(path[-1])

    return optimized_path, path_history

def smooth_path_with_spline(path, smoothing_factor=0):
    """
    Smooth the path using B-spline interpolation.

    Parameters:
    - path: List of tuples representing the path.
    - smoothing_factor: Smoothing factor for splprep function.

    Returns:
    - smoothed_path: List of tuples representing the smoothed path.
    """
    path = np.array(path)

    # Remove duplicate points
    _, idx = np.unique(path, axis=0, return_index=True)
    path = path[np.sort(idx)]

    x = path[:, 0]
    y = path[:, 1]

    # Need at least 4 points to fit a B-spline
    if len(x) < 4:
        print("Not enough path points to generate a B-spline curve")
        return path

    # Generate B-spline parameters and control points
    tck, u = splprep([x, y], s=smoothing_factor)

    # Generate smooth curve with higher sampling
    unew = np.linspace(0, 1, 500)  # Adjust the number of sampling points as needed
    smooth_x, smooth_y = splev(unew, tck)

    return list(zip(smooth_x, smooth_y))

def animate_path_optimization(grid, path_history, start, goal, case_number, kdtree, distance_field, show_distance_field=True):
    """
    Animate the path optimization process and save it as a GIF.

    Parameters:
    - grid: 2D numpy array representing the map grid.
    - path_history: List of paths during optimization.
    - start: Starting point coordinates.
    - goal: Goal point coordinates.
    - case_number: Identifier for the test case.
    - kdtree: KDTree object of obstacle coordinates.
    - distance_field: 2D numpy array with distance values.
    - show_distance_field: Boolean to control whether to display the distance field.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if show_distance_field:
        # Plot distance field
        cmap = plt.cm.viridis
        cmap.set_bad(color='red')
        distance_plot = ax.imshow(distance_field.T, cmap=cmap, origin='lower', interpolation='nearest')
        fig.colorbar(distance_plot, ax=ax, label='Distance to Nearest Obstacle')
    else:
        # Plot obstacles
        obstacle_coords = np.array(kdtree.data)
        ax.scatter(obstacle_coords[:, 0], obstacle_coords[:, 1], c='gray', s=10, label='Obstacles')

    # Plot start and goal
    ax.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], marker='*', color='red', s=150, label='Goal')

    # Initialize path line
    line_initial, = ax.plot([], [], 'r--', linewidth=1, label='Initial Path')
    line_optimized, = ax.plot([], [], 'b-', linewidth=2, label='Optimizing Path')

    # Adjust legend position
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))

    ax.set_title(f'Path Optimization Process - Test Case {case_number}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_aspect('equal', adjustable='box')

    # Plot the initial path
    x_initial, y_initial = zip(*path_history[0])
    line_initial.set_data(x_initial, y_initial)

    def init():
        line_optimized.set_data([], [])
        return line_optimized,

    def update(frame):
        current_path = path_history[frame]
        x, y = zip(*current_path)
        line_optimized.set_data(x, y)
        ax.set_title(f'Path Optimization - Iteration {frame+1}/{len(path_history)} - Test Case {case_number}')
        return line_optimized,

    ani = animation.FuncAnimation(fig, update, frames=len(path_history), init_func=init,
                                  blit=True, interval=200, repeat=False)

    # Save the animation as a GIF
    ani.save(f"case_{case_number}_optimization.gif", writer='imagemagick')
    plt.show()

def plot_final_smoothed_result(grid, path, optimized_path, smoothed_path, start, goal, case_number, kdtree, distance_field, show_distance_field=True):
    """
    Plot the initial, optimized, and smoothed paths on the grid and save the figure.

    Parameters:
    - grid: 2D numpy array representing the map grid.
    - path: Initial path as a list of tuples.
    - optimized_path: Optimized path as a list of tuples.
    - smoothed_path: Smoothed path as a list of tuples.
    - start: Starting point coordinates.
    - goal: Goal point coordinates.
    - case_number: Identifier for the test case.
    - kdtree: KDTree object of obstacle coordinates.
    - distance_field: 2D numpy array with distance values.
    - show_distance_field: Boolean to control whether to display the distance field.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if show_distance_field:
        # Plot distance field
        cmap = plt.cm.viridis
        cmap.set_bad(color='red')
        distance_plot = ax.imshow(distance_field.T, cmap=cmap, origin='lower', interpolation='nearest')
        fig.colorbar(distance_plot, ax=ax, label='Distance to Nearest Obstacle')
    else:
        # Plot obstacles
        obstacle_coords = np.array(kdtree.data)
        ax.scatter(obstacle_coords[:, 0], obstacle_coords[:, 1], c='gray', s=10, label='Obstacles')

    # Plot paths with adjusted colors
    x_initial, y_initial = zip(*path)
    ax.plot(x_initial, y_initial, 'r--', linewidth=1, label='Initial Path')

    x_opt, y_opt = zip(*optimized_path)
    ax.plot(x_opt, y_opt, 'b-', linewidth=2, label='Optimized Path')

    x_smooth, y_smooth = zip(*smoothed_path)
    ax.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='Smoothed Path')

    # Plot start and goal
    ax.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], marker='*', color='red', s=150, label='Goal')

    # Adjust legend position
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))

    ax.set_title(f'Final Smoothed Path - Test Case {case_number}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_aspect('equal', adjustable='box')

    # Save the figure as an image
    plt.savefig(f"case_{case_number}_final_result.png")
    plt.show()

# =========================
# Main Execution: Testing Multiple Cases
# =========================

def run_test_cases(num_cases=1):
    """
    Run the path planning and optimization algorithm for multiple test cases.

    Parameters:
    - num_cases: Number of test cases to run.
    """
    for case_number in range(1, num_cases + 1):
        print(f"\n=== Test Case {case_number} ===")

        # Adjust grid size and parameters if needed
        grid_size = (100, 100)  # Increased grid size for finer resolution
        grid = np.zeros(grid_size, dtype=int)

        # Randomly generate obstacles without a fixed seed
        generate_obstacles(grid, num_obstacles=500, min_size=2, max_size=5)

        # Calculate distance field using KD-Tree
        print("Calculating distance field using KD-Tree...")
        distance_field = calculate_distance_field_kdtree(grid)

        # Build obstacle KD-Tree
        obstacle_kdtree, obstacle_coords = build_kdtree(grid)

        # Randomly select start and goal positions that are connectable
        max_attempts = 1000  # Maximum attempts to find a connectable pair
        attempt = 0
        path_found = False
        while attempt < max_attempts:
            attempt += 1
            # Randomly select start and goal positions in free space
            free_space_coords = np.argwhere(grid == 0)
            if len(free_space_coords) < 2:
                print("Not enough free space for start and goal positions.")
                break
            start_idx = random.randint(0, len(free_space_coords) - 1)
            goal_idx = random.randint(0, len(free_space_coords) - 1)
            start = tuple(free_space_coords[start_idx])
            goal = tuple(free_space_coords[goal_idx])

            # Ensure start and goal are not the same
            if start == goal:
                continue

            grid[start[0], start[1]] = 0
            grid[goal[0], goal[1]] = 0

            # Run A* search
            path = a_star_search_with_potential_field(start, goal, grid, distance_field, influence_strength=5.0)
            if path:
                print(f"Found initial path in attempt {attempt}, length: {len(path)}")
                path_found = True
                break

        if not path_found:
            print("Could not find a connectable start and goal after many attempts.")
            continue

        # Optimize path
        print("Starting path optimization...")
        optimized_path, path_history = optimize_path_with_scipy(
            path, grid, obstacle_kdtree,
            obs_weight=2.0,  # Increase obstacle weight to avoid obstacles more aggressively
            smooth_weight=0.5,
            length_weight=0.1,
            max_iterations=10
        )
        print(f"Optimized path length: {len(optimized_path)}")

        # Animate optimization process with distance field
        animate_path_optimization(grid, path_history, start, goal, case_number, obstacle_kdtree, distance_field, show_distance_field=True)

        # Animate optimization process without distance field
        animate_path_optimization(grid, path_history, start, goal, case_number, obstacle_kdtree, distance_field, show_distance_field=False)

        # Smooth path
        print("Starting path smoothing...")
        smoothed_path = smooth_path_with_spline(optimized_path, smoothing_factor=1.0)
        print(f"Smoothed path length: {len(smoothed_path)}")

        # Plot final result with distance field
        plot_final_smoothed_result(grid, path, optimized_path, smoothed_path, start, goal, case_number, obstacle_kdtree, distance_field, show_distance_field=True)

        # Plot final result without distance field
        plot_final_smoothed_result(grid, path, optimized_path, smoothed_path, start, goal, case_number, obstacle_kdtree, distance_field, show_distance_field=False)

# Run the test cases
run_test_cases(num_cases=10)
