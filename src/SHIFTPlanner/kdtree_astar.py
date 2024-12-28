import numpy as np
import heapq
from math import radians, cos, sin, atan2, degrees, hypot
import random
import matplotlib.pyplot as plt
from queue import Queue
from matplotlib import patches

# Grid dimensions (assuming each unit is 1 meter for simplicity)
grid_size = (100, 50)  # Corresponding to the room size of 100x50 meters
grid = np.zeros(grid_size)  # 0 represents free space

# Marking obstacles (furniture) on the grid
# Table at (-4, 0) with size (4, 2) -> grid coordinates (8, 5) to (12, 7)
grid[8:12, 5:7] = 1  # 1 represents an obstacle

# Sofa at (5, -3) with size (6, 2) -> grid coordinates (15, 3) to (20, 5)
grid[15:20, 3:5] = 1

def calculate_potential_field(grid):
    """
    Calculate the potential field where each cell contains the distance to the nearest obstacle.
    """
    distance_field = np.ones_like(grid, dtype=float) * np.inf
    visited = np.zeros_like(grid, dtype=bool)
    
    obstacles = np.argwhere(grid == 1)
    queue = Queue()
    
    for obstacle in obstacles:
        obstacle = tuple(obstacle)
        queue.put(obstacle)
        distance_field[obstacle] = 0.0
        visited[obstacle] = True
    
    while not queue.empty():
        current = queue.get()
        neighbors = get_neighbors(current, grid_size)
        
        for neighbor in neighbors:
            if not visited[tuple(neighbor)]:
                queue.put(tuple(neighbor))
                # Update distance based on Euclidean distance
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                distance = distance_field[current] + hypot(dx, dy)
                distance_field[tuple(neighbor)] = distance
                visited[tuple(neighbor)] = True
    
    return distance_field

def get_neighbors(point, grid_size):
    """
    Get all 8-connected neighbors of a point.
    """
    x, y = point
    neighbors = []
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue  # Skip the current point
            new_x, new_y = x + i, y + j
            if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1]:
                neighbors.append((new_x, new_y))
    
    return neighbors

def plot_potential_field(ax, grid, distance_field):
    """
    Plot the potential field.
    """
    # Display free space as white and obstacles as dark gray
    ax.imshow(grid.T, cmap='Greys', origin='lower')
    # Overlay the potential field with viridis colormap
    im = ax.imshow(distance_field.T, cmap='viridis', origin='lower', alpha=0.6)
    ax.set_title('Potential Field')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.colorbar(im, ax=ax, label='Distance to Nearest Obstacle (meters)')
    return ax

def heuristic(a, b):
    """Calculate the Euclidean distance between two points."""
    return hypot(b[0] - a[0], b[1] - a[1])

def a_star_search_with_potential_field(start, goal, distance_field, influence_strength=20):
    """
    Perform A* search with potential field influence using 8-way connectivity.
    """
    neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]  # 8-way connectivity
    close_set = set()
    came_from = {}
    gscore = {start: 0.0}
    fscore = {start: heuristic(start, goal) + influence_strength / max(1.0, distance_field[start])}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            # Reconstruct the path
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + hypot(i, j)  # Euclidean distance for diagonal moves
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, np.inf):
                continue

            if tentative_g_score < gscore.get(neighbor, np.inf) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal) + influence_strength / max(1.0, distance_field[neighbor])
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return False

def min_distance_to_obstacle(point, grid):
    """Calculate the minimum distance from a point to the nearest obstacle boundary in the grid."""
    px, py = point
    min_dist = np.inf
    obstacles = np.argwhere(grid == 1)
    for obstacle in obstacles:
        x, y = obstacle
        # Define the obstacle square boundaries
        left = x - 0.5
        right = x + 0.5
        bottom = y - 0.5
        top = y + 0.5

        # Compute distance from point to the square
        dx = max(left - px, 0, px - right)
        dy = max(bottom - py, 0, py - top)

        if dx > 0 and dy > 0:
            dist = hypot(dx, dy)
        else:
            dist = max(dx, dy)
        
        if dist < min_dist:
            min_dist = dist

    return min_dist

def generate_obstacle(grid, min_width, max_width, min_height, max_height):
    """
    Generate a random rectangular obstacle on the grid.
    """
    width = random.randint(min_width, max_width)
    height = random.randint(min_height, max_height)
    start_row = random.randint(0, grid.shape[0] - height - 1)
    start_col = random.randint(0, grid.shape[1] - width - 1)
    grid[start_row:start_row + height, start_col:start_col + width] = 1

def draw_waypoint_circles(path, grid, ax, threshold_distance=5):
    """
    Draw circles around each waypoint. Use a special color if the distance to the nearest obstacle is below the threshold.
    """
    for point in path:
        dist = min_distance_to_obstacle(point, grid)
        if dist < threshold_distance:
            circle_color = 'red'  # Close to obstacle
        else:
            circle_color = 'blue'  # Far from obstacle
        # Draw the tangent circle
        circle = plt.Circle((point[0], point[1]), dist, color=circle_color, fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circle)
        # Draw the center point
        ax.plot(point[0], point[1], marker='o', color='green', markersize=3)

def main():
    # Generate 100 random rectangular obstacles
    for _ in range(100):
        generate_obstacle(grid, min_width=1, max_width=4, min_height=2, max_height=2)

    # Define start and goal positions
    start = (1, 10)  # Start position (bottom left corner)
    goal = (90, 40)  # Goal position (near the top right corner)

    # Calculate the potential field
    distance_field = calculate_potential_field(grid)

    # Perform A* search with potential field influence
    path = a_star_search_with_potential_field(start, goal, distance_field, influence_strength=20)

    if not path:
        print("No path found!")
    else:
        # ==========================
        # Plot 1: Original Grid with Start and Goal
        # ==========================
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.imshow(grid.T, cmap='Greys', origin='lower')  # Obstacles as dark gray, free space as white
        ax1.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
        ax1.scatter(goal[0], goal[1], marker='x', color='blue', s=100, label='Goal')
        ax1.set_title('Original Grid with Start and Goal')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.legend()
        ax1.set_xlim(0, grid_size[0])
        ax1.set_ylim(0, grid_size[1])
        ax1.set_aspect('equal')
        plt.show()

        # ==========================
        # Plot 2: Potential Field
        # ==========================
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        plot_potential_field(ax2, grid, distance_field)
        ax2.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
        ax2.scatter(goal[0], goal[1], marker='x', color='blue', s=100, label='Goal')
        ax2.set_title('Potential Field')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.legend()
        ax2.set_xlim(0, grid_size[0])
        ax2.set_ylim(0, grid_size[1])
        ax2.set_aspect('equal')
        plt.show()

        # ==========================
        # Plot 3: A* Search Path
        # ==========================
        fig3, ax3 = plt.subplots(figsize=(10, 10))
        ax3.imshow(grid.T, cmap='Greys', origin='lower')  # Obstacles as dark gray, free space as white
        ax3.plot([p[0] for p in path], [p[1] for p in path], color='red', linewidth=2, label='A* Path')
        ax3.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
        ax3.scatter(goal[0], goal[1], marker='x', color='blue', s=100, label='Goal')
        ax3.set_title('A* Search Path')
        ax3.set_xlabel('X-axis')
        ax3.set_ylabel('Y-axis')
        ax3.legend()
        ax3.set_xlim(0, grid_size[0])
        ax3.set_ylim(0, grid_size[1])
        ax3.set_aspect('equal')
        plt.show()

        # ==========================
        # Plot 4: A* Path with Waypoint Circles
        # ==========================
        fig4, ax4 = plt.subplots(figsize=(10, 10))
        ax4.imshow(grid.T, cmap='Greys', origin='lower')  # Obstacles as dark gray, free space as white
        ax4.plot([p[0] for p in path], [p[1] for p in path], color='red', linewidth=2, label='A* Path')
        draw_waypoint_circles(path, grid, ax4, threshold_distance=5)
        ax4.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
        ax4.scatter(goal[0], goal[1], marker='x', color='blue', s=100, label='Goal')
        ax4.set_title('A* Path with Waypoint Circles')
        ax4.set_xlabel('X-axis')
        ax4.set_ylabel('Y-axis')
        ax4.legend()
        ax4.set_xlim(0, grid_size[0])
        ax4.set_ylim(0, grid_size[1])
        ax4.set_aspect('equal')
        plt.show()

        # ==========================
        # Plot 5: Waypoints Only (Yellow Stars)
        # ==========================
        fig5, ax5 = plt.subplots(figsize=(10, 10))
        ax5.imshow(grid.T, cmap='Greys', origin='lower')  # Obstacles as dark gray, free space as white
        # Plot waypoints as yellow stars
        waypoints_x = [p[0] for p in path]
        waypoints_y = [p[1] for p in path]
        ax5.scatter(waypoints_x, waypoints_y, marker='*', color='yellow', s=150, label='Waypoints')
        ax5.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
        ax5.scatter(goal[0], goal[1], marker='x', color='blue', s=100, label='Goal')
        ax5.set_title('Waypoints Only')
        ax5.set_xlabel('X-axis')
        ax5.set_ylabel('Y-axis')
        ax5.legend()
        ax5.set_xlim(0, grid_size[0])
        ax5.set_ylim(0, grid_size[1])
        ax5.set_aspect('equal')
        plt.show()

        # ==========================
        # Plot 6: Circles for Waypoints Near Obstacles
        # ==========================
        fig6, ax6 = plt.subplots(figsize=(10, 10))
        ax6.imshow(grid.T, cmap='Greys', origin='lower')  # Obstacles as dark gray, free space as white
        # Draw circles only for waypoints within a smaller threshold distance (e.g., 2 meters)
        smaller_threshold = 2
        for point in path:
            dist = min_distance_to_obstacle(point, grid)
            if dist < smaller_threshold:
                circle_color = 'blue'  # Within smaller threshold
                circle = plt.Circle((point[0], point[1]), dist, color=circle_color, fill=False, linestyle='--', linewidth=1)
                ax6.add_patch(circle)
                # Optionally, mark the waypoint
                ax6.plot(point[0], point[1], marker='o', color='yellow', markersize=5)
        ax6.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
        ax6.scatter(goal[0], goal[1], marker='x', color='blue', s=100, label='Goal')
        ax6.set_title('Circles Around Waypoints Near Obstacles')
        ax6.set_xlabel('X-axis')
        ax6.set_ylabel('Y-axis')
        ax6.legend()
        ax6.set_xlim(0, grid_size[0])
        ax6.set_ylim(0, grid_size[1])
        ax6.set_aspect('equal')
        plt.show()

if __name__ == "__main__":
    main()

