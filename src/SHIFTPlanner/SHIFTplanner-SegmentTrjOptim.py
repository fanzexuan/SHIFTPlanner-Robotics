import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

# Set random seed for reproducibility (optional)
random.seed(42)
np.random.seed(42)

# =========================
# Environment Setup and Obstacle Generation
# =========================

# Define grid size
grid_size = (100, 100)
grid = np.zeros(grid_size)

def generate_obstacle(grid, min_width, max_width, min_height, max_height):
    """Generate random rectangular obstacles."""
    width = random.randint(min_width, max_width)
    height = random.randint(min_height, max_height)
    x = random.randint(0, grid.shape[0] - width - 1)
    y = random.randint(0, grid.shape[1] - height - 1)
    grid[x:x+width, y:y+height] = 1  # Mark obstacle

# Generate multiple random obstacles
for _ in range(50):
    generate_obstacle(grid, min_width=5, max_width=15, min_height=5, max_height=15)

# =========================
# Potential Field Calculation
# =========================

def calculate_potential_field(grid):
    """Calculate distance field representing the distance from each point to the nearest obstacle."""
    grid_size = grid.shape
    distance_field = np.full(grid_size, np.inf)
    obstacles = np.argwhere(grid == 1)
    free_space = np.argwhere(grid == 0)
    if len(obstacles) > 0:
        kdtree = KDTree(obstacles)
        distances, _ = kdtree.query(free_space)
        for idx, coord in enumerate(free_space):
            distance_field[tuple(coord)] = distances[idx]
    else:
        distance_field[free_space[:,0], free_space[:,1]] = np.inf
    return distance_field

distance_field = calculate_potential_field(grid)

# =========================
# A* Search Algorithm with Potential Field
# =========================

def heuristic(a, b):
    """Euclidean distance."""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star_search_with_potential_field(start, goal, grid, distance_field, influence_strength=5.0):
    """A* search algorithm using potential field."""
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-neighbors
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

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

            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            tentative_gscore = gscore[current] + heuristic(current, neighbor)
            if neighbor in close_set and tentative_gscore >= gscore.get(neighbor, 0):
                continue

            if tentative_gscore < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                potential = influence_strength / (distance_field[neighbor[0]][neighbor[1]] + 1e-5)
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal) + potential
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

# =========================
# Path Optimization Functions
# =========================

def min_distance_to_obstacle(point, grid):
    """Calculate the distance from a point to the nearest obstacle."""
    obstacles = np.argwhere(grid == 1)
    if len(obstacles) == 0:
        return np.inf
    kdtree = KDTree(obstacles)
    dist, _ = kdtree.query(point)
    return dist

def identify_segments_close_to_obstacles(path, grid, threshold):
    """Identify and merge adjacent segments close to obstacles."""
    indices_to_optimize = []
    for i, point in enumerate(path):
        dist = min_distance_to_obstacle(point, grid)
        if dist < threshold:
            indices_to_optimize.append(i)

    # Merge adjacent indices into continuous segments
    segments = []
    if not indices_to_optimize:
        return segments

    start_idx = indices_to_optimize[0]
    end_idx = indices_to_optimize[0]
    for idx in indices_to_optimize[1:]:
        if idx == end_idx + 1:
            end_idx = idx
        else:
            # Append the segment with extra points before and after for smoothness
            segment_start = max(start_idx - 5, 0)
            segment_end = min(end_idx + 5, len(path) - 1)
            segments.append((segment_start, path[segment_start:segment_end + 1]))
            start_idx = idx
            end_idx = idx
    # Add the last segment
    segment_start = max(start_idx - 5, 0)
    segment_end = min(end_idx + 5, len(path) - 1)
    segments.append((segment_start, path[segment_start:segment_end + 1]))
    return segments

def compute_total_cost(variables, num_points, start, goal, kdtree, obs_weight=1.0, smooth_weight=0.1, length_weight=0.1):
    """Compute the total cost function for optimization."""
    path_points = [np.array(start)]
    for i in range(num_points - 2):
        x = variables[2 * i]
        y = variables[2 * i + 1]
        path_points.append(np.array([x, y]))
    path_points.append(np.array(goal))

    total_cost = 0.0

    # Smoothness cost
    for i in range(1, num_points - 1):
        xi_prev = path_points[i - 1]
        xi = path_points[i]
        xi_next = path_points[i + 1]
        smoothness = np.linalg.norm(xi_prev - 2 * xi + xi_next) ** 2
        total_cost += smooth_weight * smoothness

    # Obstacle avoidance cost
    for i in range(1, num_points - 1):
        xi = path_points[i]
        dist, _ = kdtree.query(xi)
        obs_cost = obs_weight / (dist + 1e-5)
        total_cost += obs_cost

    # Path length cost
    for i in range(num_points - 1):
        xi = path_points[i]
        xi_next = path_points[i + 1]
        length = np.linalg.norm(xi_next - xi)
        total_cost += length_weight * length

    return total_cost

def optimize_path_segment_with_scipy(path_segment, grid, obs_weight=1.0, smooth_weight=0.1, length_weight=0.1, max_iterations=100):
    """Optimize a path segment."""
    num_points = len(path_segment)
    if num_points < 3:
        return path_segment, []  # No need to optimize

    # Build KDTree
    obstacles = np.argwhere(grid == 1)
    kdtree = KDTree(obstacles)

    initial_variables = []
    for i in range(1, num_points - 1):
        initial_variables.extend(path_segment[i])

    bounds = []
    for _ in range(num_points - 2):
        bounds.append((0, grid.shape[0] - 1))  # x range
        bounds.append((0, grid.shape[1] - 1))  # y range

    # Record optimization history for animation
    path_history = []

    def callback(variables):
        path_points = [path_segment[0]]
        for i in range(num_points - 2):
            x = variables[2 * i]
            y = variables[2 * i + 1]
            path_points.append((x, y))
        path_points.append(path_segment[-1])
        path_history.append(path_points.copy())

    result = minimize(
        compute_total_cost,
        initial_variables,
        args=(num_points, path_segment[0], path_segment[-1], kdtree, obs_weight, smooth_weight, length_weight),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations},
        callback=callback
    )

    optimized_segment = [path_segment[0]]
    optimized_variables = result.x
    for i in range(num_points - 2):
        x = optimized_variables[2 * i]
        y = optimized_variables[2 * i + 1]
        optimized_segment.append((x, y))
    optimized_segment.append(path_segment[-1])

    return optimized_segment, path_history

def integrate_path_segments(path, segments_with_indices):
    """Integrate optimized segments back into the path."""
    path = list(path)  # Ensure it's mutable
    # Sort segments in reverse order to handle overlapping segments
    segments_with_indices.sort(key=lambda x: x[0], reverse=True)
    for start_idx, optimized_segment, _ in segments_with_indices:
        end_idx = start_idx + len(optimized_segment)
        path[start_idx:end_idx] = optimized_segment
    return path

# =========================
# Path Smoothing
# =========================

def smooth_path_with_spline(path, smoothing_factor=0):
    """Smooth the path using B-splines."""
    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]

    # Remove duplicate points
    _, idx = np.unique(path, axis=0, return_index=True)
    path = path[np.sort(idx)]
    x = path[:, 0]
    y = path[:, 1]

    # Need at least 4 points to fit a B-spline
    if len(x) < 4:
        return path

    # Generate B-spline parameters and control points
    tck, u = splprep([x, y], s=smoothing_factor)

    # Generate smooth curve with higher sampling rate
    unew = np.linspace(0, 1, 500)
    smooth_x, smooth_y = splev(unew, tck)

    return list(zip(smooth_x, smooth_y))

# =========================
# Animation Function
# =========================

def animate_optimization(path_history, grid, start, goal, segment_idx):
    """Create an animation of the optimization process."""
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot obstacles
    obstacles = np.argwhere(grid == 1)
    if len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], c='lightgray', s=10)

    # Plot start and goal
    ax.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], marker='*', color='red', s=100, label='Goal')

    line, = ax.plot([], [], 'b-', linewidth=2, label='Optimizing Segment')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        path = path_history[i]
        x, y = zip(*path)
        line.set_data(x, y)
        ax.set_title(f'Segment {segment_idx} Optimization Step: {i+1}/{len(path_history)}')
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=len(path_history), init_func=init,
                                  blit=True, interval=200, repeat=False)
    plt.legend()
    plt.show()

# =========================
# Main Program
# =========================

def main():
    # Randomly select start and goal
    free_space = np.argwhere(grid == 0)
    start = tuple(free_space[np.random.choice(len(free_space))])
    goal = tuple(free_space[np.random.choice(len(free_space))])

    # Run A* search
    path = a_star_search_with_potential_field(start, goal, grid, distance_field, influence_strength=5.0)

    if not path:
        print("No path found")
        return

    # Identify segments to optimize
    distance_threshold = 3  # Adjust as needed
    segments_to_optimize = identify_segments_close_to_obstacles(path, grid, distance_threshold)

    # Optimize each segment and collect optimized segments
    segments_with_indices = []
    for idx, (start_idx, segment) in enumerate(segments_to_optimize):
        optimized_segment, path_history = optimize_path_segment_with_scipy(
            segment, grid, obs_weight=0.5, smooth_weight=1.5, length_weight=1.0, max_iterations=20)
        segments_with_indices.append((start_idx, optimized_segment, path_history))
        # Show animation for each optimized segment
        animate_optimization(path_history, grid, start, goal, idx+1)

    # Integrate optimized segments back into the path
    optimized_path = integrate_path_segments(path, segments_with_indices)

    # Smooth the overall path
    smoothed_path = smooth_path_with_spline(optimized_path, smoothing_factor=1.0)

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 10))
    obstacles = np.argwhere(grid == 1)
    if len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], c='lightgray', s=10)

    # Plot start and goal
    ax.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], marker='*', color='red', s=100, label='Goal')

    # Plot initial path
    x_initial, y_initial = zip(*path)
    ax.plot(x_initial, y_initial, 'r--', linewidth=1, label='Initial Path')

    # Plot optimized path, using different colors for optimized segments
    colors = ['blue', 'cyan', 'magenta', 'yellow', 'orange']
    for idx, (start_idx, optimized_segment, _) in enumerate(segments_with_indices):
        x_opt, y_opt = zip(*optimized_segment)
        color = colors[idx % len(colors)]
        ax.plot(x_opt, y_opt, color=color, linewidth=2, label=f'Optimized Segment {idx+1}')

    # Plot remaining unoptimized path segments
    optimized_indices = set()
    for start_idx, optimized_segment, _ in segments_with_indices:
        optimized_indices.update(range(start_idx, start_idx + len(optimized_segment)))

    remaining_indices = [i for i in range(len(optimized_path)) if i not in optimized_indices]
    if remaining_indices:
        remaining_points = [optimized_path[i] for i in remaining_indices]
        x_remain, y_remain = zip(*remaining_points)
        #ax.plot(x_remain, y_remain, 'b-', linewidth=2, label='Unoptimized Path')

    # Plot smoothed path
    x_smooth, y_smooth = zip(*smoothed_path)
    ax.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='Smoothed Path')

    # Optionally, draw tangent circles
    for point in path:
        dist = min_distance_to_obstacle(point, grid)
        if dist < distance_threshold:
            circle = plt.Circle((point[0], point[1]), dist, color='orange', fill=False, linestyle='--', linewidth=1)
            ax.add_patch(circle)

    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Path Optimization with Merged Segments')
    plt.show()

if __name__ == "__main__":
    main()
