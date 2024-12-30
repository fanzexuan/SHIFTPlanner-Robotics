import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter

# =========================
# Path Planning and Optimization Functions
# =========================

def generate_obstacles(grid, num_obstacles=20, min_size=5, max_size=10):
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
        x = random.randint(0, grid.shape[0] - height - 1)
        y = random.randint(0, grid.shape[1] - width - 1)
        grid[x:x+height, y:y+width] = 1  # Mark obstacle cells with 1

def generate_dirty_area_map(grid_size, smooth_sigma=5):
    """
    Generate a dirty area distribution map.

    Parameters:
    - grid_size: Tuple representing the size of the grid.
    - smooth_sigma: Sigma value for Gaussian smoothing.

    Returns:
    - dirty_map: 2D numpy array representing the dirty area distribution.
    """
    np.random.seed(0)
    dirty_map = np.random.rand(*grid_size)
    dirty_map = gaussian_filter(dirty_map, sigma=smooth_sigma)
    # Normalize to 0-1
    dirty_map = (dirty_map - dirty_map.min()) / (dirty_map.max() - dirty_map.min())
    return dirty_map

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
                      options={'maxiter': max_iterations, 'disp': False}, callback=callback)

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

# Step 1: Generate the zigzag path
def generate_zigzag_path(grid_size, line_spacing=10):
    """
    Generate a zigzag path covering the grid.
    """
    path = []
    x_max, y_max = grid_size
    x_positions = range(0, x_max, line_spacing)
    direction = 1  # 1 for up, -1 for down
    for x in x_positions:
        y_positions = range(0, y_max) if direction == 1 else range(y_max - 1, -1, -1)
        for y in y_positions:
            # Ensure indices are within grid bounds
            if x < x_max and y < y_max:
                path.append((x, y))
        direction *= -1  # Change direction
    return path

# Step 3: Detect collisions
def detect_collisions(path, grid):
    """
    Detect where the path intersects obstacles.
    Returns a list of collision segments with their start and end indices in the path.
    """
    collision_segments = []
    in_obstacle = False
    start_idx = None
    for i, point in enumerate(path):
        x, y = int(point[0]), int(point[1])
        # Ensure indices are within grid bounds
        if x >= grid.shape[0] or y >= grid.shape[1]:
            continue
        if grid[x, y] == 1:
            if not in_obstacle:
                # Entering an obstacle
                in_obstacle = True
                start_idx = i - 1 if i > 0 else i
        else:
            if in_obstacle:
                # Exiting an obstacle
                in_obstacle = False
                end_idx = i
                collision_segments.append((start_idx, end_idx))
    # If path ends inside an obstacle
    if in_obstacle:
        collision_segments.append((start_idx, len(path) - 1))
    return collision_segments

# Main function to process the zigzag path
def process_zigzag_path(grid_size, grid, distance_field, obstacle_kdtree, dirty_map):
    # Generate the zigzag path
    path = generate_zigzag_path(grid_size, line_spacing=10)
    print(f"Generated zigzag path with {len(path)} points.")
    
    # Detect collisions
    collision_segments = detect_collisions(path, grid)
    print(f"Detected {len(collision_segments)} collision segments.")
    
    # Process each collision segment
    optimized_path = []
    idx = 0
    for segment in collision_segments:
        start_idx, end_idx = segment
        # Add path before the collision segment
        optimized_path.extend(path[idx:start_idx + 1])
        # Get entry and exit points
        start_point = path[start_idx]
        end_point = path[end_idx]
        # Use A* to find path around obstacle
        print(f"Finding A* path from {start_point} to {end_point}...")
        a_star_path = a_star_search_with_potential_field(
            (int(start_point[0]), int(start_point[1])),
            (int(end_point[0]), int(end_point[1])),
            grid,
            distance_field,
            influence_strength=5.0
        )
        if a_star_path:
            # Convert path points to float for optimization
            a_star_path = [(float(p[0]), float(p[1])) for p in a_star_path]
            print(f"A* path found with {len(a_star_path)} points.")
            # Optimize the A* path
            print("Optimizing A* path segment...")
            optimized_segment, _ = optimize_path_with_scipy(
                a_star_path, grid, obstacle_kdtree,
                obs_weight=2.0,
                smooth_weight=0.5,
                length_weight=0.1,
                max_iterations=10
            )
            # Smooth the optimized path segment
            print("Smoothing optimized path segment...")
            smoothed_segment = smooth_path_with_spline(optimized_segment, smoothing_factor=1.0)
            # Remove start point to avoid duplication
            if len(smoothed_segment) > 1:
                smoothed_segment = smoothed_segment[1:]
            # Add the smoothed segment to the optimized path
            optimized_path.extend(smoothed_segment)
        else:
            print("No path found around obstacle.")
            return None
        idx = end_idx  # Move index to end of collision segment
    # Add remaining path after the last collision segment
    optimized_path.extend(path[idx:])
    return path, optimized_path

def assign_speeds_along_path(path, dirty_map, min_speed=1.0, max_speed=5.0):
    """
    Assign speeds along the path based on the dirty area map.

    Parameters:
    - path: List of tuples representing the path.
    - dirty_map: 2D numpy array representing the dirty area distribution.
    - min_speed: Minimum speed.
    - max_speed: Maximum speed.

    Returns:
    - speeds: List of speeds corresponding to each point in the path.
    """
    speeds = []
    grid_size = dirty_map.shape
    for point in path:
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
            dirtiness = dirty_map[x, y]
            # Speed inversely proportional to dirtiness
            speed = max_speed - (max_speed - min_speed) * dirtiness
            speeds.append(speed)
        else:
            speeds.append(max_speed)  # Default speed
    return speeds

# Now, let's run the process
def run_zigzag_test_case():
    # Grid size
    grid_size = (100, 100)
    grid = np.zeros(grid_size, dtype=int)

    # Generate obstacles
    generate_obstacles(grid, num_obstacles=10, min_size=5, max_size=15)

    # Generate dirty area map
    print("Generating dirty area map...")
    dirty_map = generate_dirty_area_map(grid_size, smooth_sigma=5)

    # Calculate distance field
    print("Calculating distance field using KD-Tree...")
    distance_field = calculate_distance_field_kdtree(grid)

    # Build KD-Tree for obstacles
    obstacle_kdtree, obstacle_coords = build_kdtree(grid)

    # Process the zigzag path
    result = process_zigzag_path(grid_size, grid, distance_field, obstacle_kdtree, dirty_map)
    if result is None:
        print("Failed to process the zigzag path.")
        return
    original_path, optimized_path = result

    # Assign speeds along the optimized path based on dirty map
    speeds = assign_speeds_along_path(optimized_path, dirty_map)

    # Plot KDTree map
    plt.figure(figsize=(10, 8))
    plt.title('KDTree Map (Obstacles)')
    plt.imshow(grid.T, cmap='gray', origin='lower', interpolation='nearest')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Plot dirty area map
    plt.figure(figsize=(10, 8))
    plt.title('Dirty Area Distribution Map')
    plt.imshow(dirty_map.T, cmap='hot', origin='lower', interpolation='nearest')
    plt.colorbar(label='Dirtiness')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Plot speed allocation trajectory map
    plt.figure(figsize=(10, 8))
    plt.title('Speed Allocation Trajectory Map')
    plt.imshow(dirty_map.T, cmap='hot', origin='lower', interpolation='nearest')
    x_opt, y_opt = zip(*optimized_path)
    sc = plt.scatter(x_opt, y_opt, c=speeds, cmap='jet', s=5)
    plt.colorbar(sc, label='Speed')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Plot final coverage map
    plt.figure(figsize=(10, 8))
    plt.title('Final Coverage Map')
    plt.imshow(grid.T, cmap='gray', origin='lower', interpolation='nearest')
    plt.plot(x_opt, y_opt, 'g-', linewidth=2, label='Optimized Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Combined display map
    plt.figure(figsize=(10, 8))
    plt.title('Combined Display Map')
    plt.imshow(dirty_map.T, cmap='hot', origin='lower', interpolation='nearest', alpha=0.5)
    plt.imshow(grid.T, cmap='gray', origin='lower', interpolation='nearest', alpha=0.5)
    sc = plt.scatter(x_opt, y_opt, c=speeds, cmap='jet', s=5, label='Optimized Path')
    plt.colorbar(sc, label='Speed')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Run the test case
run_zigzag_test_case()

