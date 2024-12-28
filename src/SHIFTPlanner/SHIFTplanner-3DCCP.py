import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import RegularGridInterpolator, splprep, splev, NearestNDInterpolator
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
import heapq

# =========================
# Terrain and Obstacle Generation Functions
# =========================

def generate_random_terrain_points(n_points, x_range, y_range):
    """
    Generate random terrain points simulating hilly terrain using sine functions.
    """
    x = np.random.uniform(*x_range, n_points)
    y = np.random.uniform(*y_range, n_points)
    z = np.sin(0.5 * x) + np.sin(0.5 * y) + 0.5 * np.sin(0.2 * x) * np.cos(0.3 * y)
    return np.column_stack((x, y, z))

def create_tin(points):
    """
    Create a TIN (Triangulated Irregular Network) from terrain points.
    """
    return Delaunay(points[:, :2])

def plot_terrain(points, tri, obstacles=None, title='Terrain Surface'):
    """
    Plot the terrain surface and display obstacles.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                           triangles=tri.simplices, cmap='terrain', edgecolor='none', alpha=0.8)
    if obstacles is not None:
        for obs in obstacles:
            x_min, x_max, y_min, y_max, z_min, z_max = obs
            ax.bar3d(x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min,
                     color='gray', alpha=0.5)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Elevation')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation')
    plt.show()

def generate_aridity_map(x_range, y_range, resolution):
    """
    Generate an aridity map over the terrain with smoother variations.
    """
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    # Use a smoother function to simulate aridity distribution
    aridity = 0.5 + 0.5 * np.sin(0.2 * X) * np.cos(0.2 * Y)
    aridity = 1 - (aridity - np.min(aridity)) / (np.max(aridity) - np.min(aridity))  # Normalize and invert
    return x, y, aridity  # Return x and y axes

def interpolate_aridity_map(x, y, aridity_map):
    """
    Create an interpolation function for the aridity map.
    """
    interp_func = RegularGridInterpolator((x, y), aridity_map.T, method='linear', bounds_error=False, fill_value=np.mean(aridity_map))
    interp_func.values = aridity_map  # Store original aridity_map for handling NaNs
    return interp_func

def generate_obstacles(points, num_obstacles=3, min_size=5, max_size=7):
    """
    Randomly generate cubic obstacles on the terrain and return their boundary list.
    """
    np.random.seed(42)  # For reproducibility
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    obstacles = []
    for _ in range(num_obstacles):
        width = np.random.uniform(min_size, max_size)
        depth = np.random.uniform(min_size, max_size)
        height = np.random.uniform(2, 5)  # Obstacle height

        x0 = np.random.uniform(x_min, x_max - width)
        y0 = np.random.uniform(y_min, y_max - depth)
        z0 = terrain_interpolator([[x0, y0]])[0]  # Obstacle base level with terrain surface
        if np.isnan(z0):
            z0 = np.mean(points[:, 2])  # Set to average terrain elevation if NaN

        obstacles.append((x0, x0 + width, y0, y0 + depth, z0, z0 + height))
    return obstacles

def is_point_in_obstacle(point, obstacles):
    """
    Check if a given point is inside any obstacle.
    """
    x, y, z = point
    for obs in obstacles:
        x_min, x_max, y_min, y_max, z_min, z_max = obs
        if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
            return True
    return False

def heuristic(a, b):
    """
    Compute the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

# =========================
# A* Search with Potential Field
# =========================

def a_star_search_with_potential_field(start, goal, obstacles, terrain_interpolator, x_range, y_range, altitude, influence_radius=1.0):
    """
    Perform A* search on the terrain, avoiding obstacles using a potential field.
    """
    # Define possible movement directions (8-connected grid)
    movements = [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5),
                 (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Build obstacle KD-tree for potential field
    obstacle_points = []
    for obs in obstacles:
        x_min_obs, x_max_obs, y_min_obs, y_max_obs, z_min_obs, z_max_obs = obs
        xs = np.linspace(x_min_obs, x_max_obs, num=3)
        ys = np.linspace(y_min_obs, y_max_obs, num=3)
        zs = np.linspace(z_min_obs, z_max_obs, num=3)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        obstacle_points.extend(grid)
    obstacle_points = np.array(obstacle_points)
    obstacles_kdtree = cKDTree(obstacle_points)

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            data.reverse()
            # Get z coordinates
            path_points = []
            for point in data:
                z = terrain_interpolator([point])[0] + altitude
                if np.isnan(z):
                    z = altitude + np.mean(points[:, 2])
                path_points.append((point[0], point[1], z))
            return path_points

        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (x_min <= neighbor[0] <= x_max and y_min <= neighbor[1] <= y_max):
                continue
            z = terrain_interpolator([neighbor])[0] + altitude
            if np.isnan(z):
                z = altitude + np.mean(points[:, 2])
            neighbor_3d = (neighbor[0], neighbor[1], z)
            if is_point_in_obstacle(neighbor_3d, obstacles):
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            # Potential field to keep away from obstacles
            dist, _ = obstacles_kdtree.query([neighbor_3d], k=1)
            potential = influence_radius / (dist[0] + 1e-5)  # Avoid division by zero

            tentative_f_score = tentative_g_score + heuristic(neighbor, goal) + potential

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_f_score, neighbor))

    return False

# =========================
# Path Optimization Functions
# =========================

def build_kdtree_3d(obstacles):
    """
    Build a KD-tree of obstacle coordinates for efficient nearest neighbor queries.
    """
    obstacle_coords = []
    for obs in obstacles:
        x_min, x_max, y_min, y_max, z_min, z_max = obs
        xs = np.linspace(x_min, x_max, num=5)
        ys = np.linspace(y_min, y_max, num=5)
        zs = np.linspace(z_min, z_max, num=5)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1,3)
        obstacle_coords.extend(grid)
    obstacle_coords = np.array(obstacle_coords)
    kdtree = cKDTree(obstacle_coords)
    return kdtree

def optimize_path_with_scipy_3d(path, obstacles_kdtree, obs_weight=0.1, smooth_weight=0.3, length_weight=0.5, max_iterations=20):
    """
    Optimize a 3D path using Scipy optimizer, avoiding obstacles and improving smoothness.
    """
    path = np.array(path)
    num_points = len(path)

    if num_points < 2:
        print("Path is too short to optimize.")
        return path

    # Initial variables (excluding start and end points)
    initial_variables = path[1:-1].flatten()

    # Bounds for variables
    bounds = []
    for _ in range(num_points - 2):
        bounds.extend([(None, None), (None, None), (None, None)])  # x, y, z bounds

    # Objective function
    def objective(variables):
        x = variables.reshape(-1, 3)
        full_path = np.vstack((path[0], x, path[-1]))
        cost = 0.0
        for i in range(1, num_points):
            # Path length cost
            cost += length_weight * np.linalg.norm(full_path[i] - full_path[i - 1])
            # Smoothness cost
            if i < num_points - 1:
                cost += smooth_weight * np.linalg.norm(full_path[i - 1] - 2 * full_path[i] + full_path[i + 1])
            # Obstacle avoidance cost
            dist, _ = obstacles_kdtree.query(full_path[i], k=1)
            if dist < 1e-3:
                dist = 1e-3  # Avoid division by zero
            cost += obs_weight * (1.0 / dist)
        return cost

    result = minimize(objective, initial_variables, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iterations})

    if not result.success:
        print("Optimization did not converge:", result.message)

    optimized_path = np.vstack((path[0], result.x.reshape(-1, 3), path[-1]))
    return optimized_path

def bspline_smoothing(path, smoothing_factor=0):
    """
    Smooth the path using B-spline interpolation.
    """
    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]
    z = path[:, 2]

    # Remove duplicate points
    _, idx = np.unique(path[:, :2], axis=0, return_index=True)
    x = x[np.sort(idx)]
    y = y[np.sort(idx)]
    z = z[np.sort(idx)]

    if len(x) < 4:
        print("Not enough points to fit a B-spline.")
        return path

    # Parameterize the path
    try:
        tck, u = splprep([x, y, z], s=smoothing_factor, k=3)
    except Exception as e:
        print("B-spline fitting failed:", e)
        return path

    # Generate new points
    unew = np.linspace(0, 1, num=len(x) * 5)
    out = splev(unew, tck)

    smoothed_path = np.column_stack((out[0], out[1], out[2]))
    return smoothed_path

# =========================
# Path Generation Function
# =========================

def generate_zigzag_waypoints_with_obstacle_avoidance(x_range, y_range, line_spacing, obstacles, terrain_interpolator, x_step, altitude):
    """
    Generate zigzag waypoints, use A* for obstacle avoidance, optimize and smooth the path segments, maintain path continuity.
    """
    y_start, y_end = y_range
    x_start, x_end = x_range

    y_lines = np.arange(y_start, y_end + line_spacing, line_spacing)
    full_path = []
    obstacles_kdtree = build_kdtree_3d(obstacles)

    previous_line_end = None  # To connect the lines

    for i, y in enumerate(y_lines):
        if i % 2 == 0:
            x_line = np.arange(x_start, x_end + x_step, x_step)
        else:
            x_line = np.arange(x_end, x_start - x_step, -x_step)

        path_line = []
        in_obstacle = False
        prev_point = None
        idx = 0
        while idx < len(x_line):
            x = x_line[idx]
            z = terrain_interpolator([[x, y]])[0] + altitude
            if np.isnan(z):
                z = altitude + np.mean(points[:, 2])
            point = (x, y, z)
            if is_point_in_obstacle(point, obstacles):
                if not in_obstacle and prev_point is not None:
                    in_obstacle = True
                    # Record the point before obstacle
                    start_point = (prev_point[0], prev_point[1])
                    start_point_3d = prev_point
                    # Find the next point after obstacle
                    idx_after_obstacle = idx
                    while idx_after_obstacle < len(x_line):
                        x_after = x_line[idx_after_obstacle]
                        z_after = terrain_interpolator([[x_after, y]])[0] + altitude
                        if np.isnan(z_after):
                            z_after = altitude + np.mean(points[:, 2])
                        point_after = (x_after, y, z_after)
                        if not is_point_in_obstacle(point_after, obstacles):
                            end_point = (x_after, y)
                            end_point_3d = point_after
                            break
                        idx_after_obstacle += 1
                    else:
                        print("Cannot find point after obstacle, skipping this line.")
                        break  # Cannot find a point after obstacle

                    # Use A* to connect start and end points
                    astar_path = a_star_search_with_potential_field(start_point, end_point,
                                                                    obstacles, terrain_interpolator, x_range, y_range, altitude)
                    if astar_path:
                        # Optimize the A* path segment
                        optimized_astar_path = optimize_path_with_scipy_3d(astar_path, obstacles_kdtree)
                        # Smooth the optimized path
                        smoothed_astar_path = bspline_smoothing(optimized_astar_path)
                        # Remove start point to avoid duplication
                        if len(smoothed_astar_path) > 1:
                            smoothed_astar_path = smoothed_astar_path[1:]
                        path_line.extend(smoothed_astar_path)
                        in_obstacle = False
                        idx = idx_after_obstacle  # Move index to point after obstacle
                        prev_point = (x_after, y, z_after)
                        continue
                    else:
                        print("Cannot find path around obstacle, skipping this line.")
                        break
            else:
                if not in_obstacle:
                    path_line.append(point)
                prev_point = point
            idx += 1

        # Connect lines if necessary
        if previous_line_end is not None and path_line:
            start_point = (previous_line_end[0], previous_line_end[1])
            end_point = (path_line[0][0], path_line[0][1])
            astar_path = a_star_search_with_potential_field(start_point, end_point,
                                                            obstacles, terrain_interpolator, x_range, y_range, altitude)
            if astar_path:
                optimized_astar_path = optimize_path_with_scipy_3d(astar_path, obstacles_kdtree)
                smoothed_astar_path = bspline_smoothing(optimized_astar_path)
                # Remove start point to avoid duplication
                if len(smoothed_astar_path) > 1:
                    smoothed_astar_path = smoothed_astar_path[1:]
                full_path.extend(smoothed_astar_path)
            else:
                print("Cannot connect lines, skipping connection.")

        full_path.extend(path_line)
        if path_line:
            previous_line_end = path_line[-1]
        else:
            previous_line_end = None

    return full_path

# =========================
# Speed Adjustment Function
# =========================

def adjust_speeds_along_path(path_points, aridity_interp):
    """
    Adjust speeds along the path based on aridity.
    """
    speeds = []
    for point in path_points:
        x, y, _ = point
        aridity_value = aridity_interp([(x, y)])[0]
        if np.isnan(aridity_value):
            aridity_value = np.mean(aridity_interp.values)  # Replace NaN with mean aridity
        speed = 1 + aridity_value * 4  # Slower speed in arid areas
        speeds.append(speed)
    return np.array(speeds)

# =========================
# Coverage Calculation Functions
# =========================

def exponential_coverage(t, lambda_):
    return 1 - np.exp(-lambda_ * t)

def gaussian_2d(d, sigma):
    return np.exp(-d ** 2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

def combined_coverage(t, d, lambda_, sigma):
    return exponential_coverage(t, lambda_) * gaussian_2d(d, sigma)

# =========================
# Coverage Map Plotting Function
# =========================

def plot_coverage_map(points, path_points, speeds, x_range, y_range, obstacles):
    """
    Generate and plot a coverage map adjusted for aridity, show obstacles.
    """
    # Parameters
    lambda_ = 0.1
    sigma = 1.0  # Adjust based on terrain scale
    resolution = 100  # Coverage map resolution

    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    coverage = np.zeros_like(X)

    # Simulate coverage accumulation
    for idx, (pos, speed) in enumerate(zip(path_points, speeds)):
        x0, y0 = pos[0], pos[1]

        # Determine stay time based on speed (inverse relationship)
        stay_time = int((max(speeds) - speed + 1) * 2)  # Adjust multiplier as needed

        for _ in range(stay_time):
            # Create a mask for the drone position
            drone_mask = np.zeros_like(X, dtype=bool)
            xi = np.abs(x - x0).argmin()
            yi = np.abs(y - y0).argmin()
            drone_mask[yi, xi] = True

            # Compute distance from drone position to all points
            distance_field = distance_transform_edt(~drone_mask) * ((x_range[1] - x_range[0]) / resolution)

            # Update coverage
            coverage += combined_coverage(1, distance_field, lambda_, sigma)

    # Normalize coverage to [0,1]
    coverage = coverage / np.max(coverage)

    # Create an interpolator for the coverage map
    coverage_interp = RegularGridInterpolator((x, y), coverage.T)

    # Get coverage values along the path
    path_coverage = coverage_interp([p[:2] for p in path_points])

    # Plot the 3D coverage map
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, coverage.T, cmap='Blues', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Coverage')
    ax.set_title('Drone Coverage Map Adjusted for Aridity (3D)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Coverage')

    # Plot obstacles
    for obs in obstacles:
        x_min, x_max, y_min, y_max, z_min, z_max = obs
        ax.bar3d(x_min, y_min, coverage.max() + 0.1, x_max - x_min, y_max - y_min, 0.5,
                 color='gray', alpha=0.8)

    # Plot the drone path
    ax.plot([p[0] for p in path_points], [p[1] for p in path_points], path_coverage + 0.1, 'r-', linewidth=1, label='Drone Path')
    ax.legend()
    plt.show()

# =========================
# Terrain and Path Plotting Function
# =========================

def plot_terrain_and_path(points, tri, path_points, speeds, obstacles, title='Drone Irrigation Path with Speed Variation (Obstacle Avoidance)'):
    """
    Plot the irrigation path on the terrain, color represents speed variations, show obstacles.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot terrain surface
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                    triangles=tri.simplices, cmap='terrain', edgecolor='none', alpha=0.8)

    # Plot obstacles
    for obs in obstacles:
        x_min, x_max, y_min, y_max, z_min, z_max = obs
        ax.bar3d(x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min,
                 color='gray', alpha=0.5)

    # Plot irrigation path with speed variation
    speeds_normalized = (speeds - speeds.min()) / (speeds.max() - speeds.min())
    p = ax.scatter([p[0] for p in path_points], [p[1] for p in path_points], [p[2] for p in path_points],
                   c=speeds_normalized, cmap='coolwarm', s=20, label='Irrigation Path')

    fig.colorbar(p, ax=ax, shrink=0.5, aspect=5, label='Speed (Time Units)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Elevation')
    ax.set_title(title)
    plt.show()

# =========================
# Animation Creation Function
# =========================
def create_animation(points, tri, path_points, speeds, obstacles, filename='drone_irrigation_final.gif'):
    """
    Create an animation showing the drone dynamically adjusting speed according to aridity, avoiding obstacles.
    """
    # Ensure path_points is a NumPy array
    path_points = np.array(path_points)
    if path_points.size == 0:
        print("No path points available for animation.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot terrain surface
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                    triangles=tri.simplices, cmap='terrain', edgecolor='none', alpha=0.8)

    # Plot obstacles
    for obs in obstacles:
        x_min, x_max, y_min, y_max, z_min, z_max = obs
        ax.bar3d(x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min,
                 color='gray', alpha=0.5)

    # Initialize drone marker
    drone_marker, = ax.plot([], [], [], 'ro', markersize=6, label='Drone')

    # Initialize traversed path line
    path_line, = ax.plot([], [], [], 'r-', linewidth=2, label='Traversed Path')

    # Set labels and title
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Elevation')
    ax.set_title('Drone Irrigation Animation with Speed Variation (Obstacle Avoidance)')
    ax.legend()

    # Set axis limits
    ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim(np.min(points[:, 1]), np.max(points[:, 1]))
    ax.set_zlim(np.min(points[:, 2]), np.max(points[:, 2]) + 5)

    # Normalize speeds for frame duration (higher speed, fewer frames)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds)
    if max_speed - min_speed == 0:
        normalized_speeds = np.ones_like(speeds) * 0.1  # Avoid division by zero
    else:
        normalized_speeds = (max_speed - speeds) / (max_speed - min_speed) + 0.1

    # Accumulate speeds to get frame indices
    frame_durations = (normalized_speeds * 5).astype(int)
    frame_durations = np.maximum(frame_durations, 1)  # Ensure at least one frame per point
    frames = np.repeat(np.arange(len(path_points)), frame_durations)

    if len(frames) == 0:
        print("No frames to animate.")
        return

    # Prepare animation
    def update(num):
        idx = frames[num]
        if idx >= len(path_points):
            idx = len(path_points) - 1  # Prevent index out of range

        # Update drone position
        drone_marker.set_data(path_points[idx, 0], path_points[idx, 1])
        drone_marker.set_3d_properties(path_points[idx, 2])

        # Update traversed path
        xs = path_points[:idx+1, 0]
        ys = path_points[:idx+1, 1]
        zs = path_points[:idx+1, 2]
        path_line.set_data(xs, ys)
        path_line.set_3d_properties(zs)

        return drone_marker, path_line

    ani = FuncAnimation(fig, update, frames=len(frames), interval=5, blit=False)

    # Save animation as GIF
    try:
        writer = PillowWriter(fps=20)
        ani.save(filename, writer=writer)
        print(f"Animation '{filename}' has been successfully created.")
    except Exception as e:
        print(f"Failed to save animation: {e}")

    plt.close(fig)

# =========================
# Aridity Map Plotting Function
# =========================

def plot_aridity_map(x, y, aridity_map):
    """
    Plot the 3D aridity field.
    """
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, aridity_map.T, cmap='hot', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Aridity')
    ax.set_title('3D Aridity Field Map')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Aridity')
    plt.show()

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    # Generate random terrain points
    n_points = 1000
    x_range = (-10, 10)
    y_range = (-10, 10)
    terrain_points = generate_random_terrain_points(n_points, x_range, y_range)

    # Create TIN
    tin = create_tin(terrain_points)

    # Create terrain height interpolator
    # Use NearestNDInterpolator to avoid NaNs outside convex hull
    terrain_interpolator = NearestNDInterpolator(terrain_points[:, :2], terrain_points[:, 2])

    # Generate aridity map
    resolution = 100  # Adjust resolution
    x_aridity, y_aridity, aridity_map = generate_aridity_map(x_range, y_range, resolution)

    # Create aridity interpolator
    aridity_interp = interpolate_aridity_map(x_aridity, y_aridity, aridity_map)

    # Plot 3D aridity map
    plot_aridity_map(x_aridity, y_aridity, aridity_map)

    # Generate obstacles (larger cubes)
    obstacles = generate_obstacles(terrain_points, num_obstacles=2, min_size=5, max_size=5)

    # Plot terrain surface and obstacles
    plot_terrain(terrain_points, tin, obstacles=obstacles, title='Terrain Surface (View 1)')

    # Plot terrain from a different angle
    plot_terrain(terrain_points, tin, obstacles=obstacles, title='Terrain Surface (View 2)')

    # Path planning parameters
    altitude = 1.0  # Height above terrain
    line_spacing = 2.0  # Line spacing for zigzag path
    x_step = 0.5     # Step size in x-direction

    # Generate zigzag waypoints with obstacle avoidance, optimize and smooth the path
    path_points = generate_zigzag_waypoints_with_obstacle_avoidance(
        x_range, y_range, line_spacing, obstacles, terrain_interpolator, x_step, altitude)

    # Convert path_points to NumPy array for consistent indexing
    path_points = np.array(path_points)

    if path_points.size == 0:
        print("No path points generated. Exiting.")
    else:
        # Adjust speeds based on aridity
        speeds = adjust_speeds_along_path(path_points, aridity_interp)

        # Plot irrigation path with speed variations and obstacles
        plot_terrain_and_path(terrain_points, tin, path_points, speeds, obstacles, title='Drone Irrigation Path with Speed Variation (View 1)')

        # Plot irrigation path from a different angle
        plot_terrain_and_path(terrain_points, tin, path_points, speeds, obstacles, title='Drone Irrigation Path with Speed Variation (View 2)')

        # Create animation of the drone irrigation process with speed variation and obstacle avoidance
        create_animation(terrain_points, tin, path_points, speeds, obstacles, filename='drone_irrigation.gif')

        # Plot the coverage map adjusted for aridity, showing obstacles
        plot_coverage_map(terrain_points, path_points, speeds, x_range, y_range, obstacles)
