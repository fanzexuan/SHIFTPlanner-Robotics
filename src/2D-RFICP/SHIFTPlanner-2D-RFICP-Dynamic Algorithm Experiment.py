import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

##############################################################################
# 1. 障碍环境 & 路径规划 (A* + 优化 + B-Spline)
##############################################################################
def generate_obstacles(grid, num_obstacles=20, min_size=5, max_size=10):
    """随机生成矩形障碍，标记为1。"""
    for _ in range(num_obstacles):
        width = random.randint(min_size, max_size)
        height = random.randint(min_size, max_size)
        x = random.randint(0, grid.shape[0] - height - 1)
        y = random.randint(0, grid.shape[1] - width - 1)
        grid[x:x+height, y:y+width] = 1

def calculate_distance_field_kdtree(grid):
    """利用KDTree计算每个栅格到最近障碍的距离场。"""
    grid_size = grid.shape
    obstacle_coords = np.argwhere(grid == 1)
    all_coords = np.indices(grid_size).reshape(2, -1).T

    if len(obstacle_coords) > 0:
        kdtree = KDTree(obstacle_coords)
        distances, _ = kdtree.query(all_coords)
        distance_field = distances.reshape(grid_size)
    else:
        distance_field = np.full(grid_size, np.inf)
    return distance_field

def build_kdtree(grid):
    """为障碍构建 KD-Tree，用于后续最近邻查询。"""
    obstacle_coords = np.argwhere(grid == 1)
    if len(obstacle_coords) > 0:
        kdtree = KDTree(obstacle_coords)
    else:
        kdtree = None
    return kdtree, obstacle_coords

def heuristic(a, b):
    """欧几里得距离。"""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star_search_with_potential_field(start, goal, grid, distance_field, influence_strength=1.0):
    """
    A* + potential field：在代价函数中增加基于障碍距离的势场，使路径更倾向远离障碍。
    """
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
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
            # 重建路径
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
            if grid[neighbor[0], neighbor[1]] == 1:  # 障碍
                continue
            if neighbor in close_set:
                continue

            tentative_g_score = gscore[current] + np.hypot(i, j)
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                # 势场：与 distance_field 成反比
                potential = influence_strength / (distance_field[neighbor] + 1e-5)
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal) + potential
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False

def compute_total_cost(variables, num_points, start, goal, kdtree, grid,
                       obs_weight=1.0, smooth_weight=0.1, length_weight=0.1):
    """
    目标函数：平滑度 + 避障 + 路径长度。
    """
    path_points = [np.array(start)]
    for i in range(num_points - 2):
        x = variables[2 * i]
        y = variables[2 * i + 1]
        path_points.append(np.array([x, y]))
    path_points.append(np.array(goal))

    total_cost = 0.0
    grid_size = grid.shape

    # 平滑度 (二阶差分)
    for i in range(1, num_points - 1):
        xi_prev = path_points[i - 1]
        xi = path_points[i]
        xi_next = path_points[i + 1]
        smoothness = np.linalg.norm(xi_prev - 2 * xi + xi_next) ** 2
        total_cost += smooth_weight * smoothness

    # 避障：距离越近，cost 越高
    if kdtree is not None:
        for i in range(1, num_points - 1):
            xi = path_points[i]
            xi_int = (int(round(xi[0])), int(round(xi[1])))
            # 越界或在障碍上 => 大惩罚
            if not (0 <= xi_int[0] < grid_size[0] and 0 <= xi_int[1] < grid_size[1]):
                total_cost += 1e6
            elif grid[xi_int] == 1:
                total_cost += 1e6
            else:
                dist, _ = kdtree.query(xi)
                obs_cost = obs_weight * np.exp(-dist)
                total_cost += obs_cost

    # 路径长度
    for i in range(num_points - 1):
        xi = path_points[i]
        xi_next = path_points[i + 1]
        length = np.linalg.norm(xi_next - xi)
        total_cost += length_weight * length

    return total_cost

def optimize_path_with_scipy(path, grid, kdtree,
                             obs_weight=1.0, smooth_weight=0.1, length_weight=0.1,
                             max_iterations=100):
    """
    采用 L-BFGS-B 对离散路径点进行优化。
    """
    num_points = len(path)
    grid_size = grid.shape
    # 只优化中间点
    initial_variables = []
    for i in range(1, num_points - 1):
        initial_variables.extend(path[i])

    bounds = []
    for _ in range(num_points - 2):
        bounds.append((0, grid_size[0] - 1))
        bounds.append((0, grid_size[1] - 1))

    def objective(variables):
        return compute_total_cost(variables, num_points, path[0], path[-1],
                                  kdtree, grid,
                                  obs_weight, smooth_weight, length_weight)

    path_history = []
    def callback(variables):
        inter_path = [path[0]]
        for i in range(num_points - 2):
            x = variables[2*i]
            y = variables[2*i + 1]
            inter_path.append((x, y))
        inter_path.append(path[-1])
        path_history.append(inter_path.copy())

    result = minimize(objective, initial_variables, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': max_iterations, 'disp': True},
                      callback=callback)

    if not result.success:
        print("Optimization failed:", result.message)

    optimized_path = [path[0]]
    for i in range(num_points - 2):
        x = result.x[2*i]
        y = result.x[2*i + 1]
        optimized_path.append((x, y))
    optimized_path.append(path[-1])
    return optimized_path, path_history

def smooth_path_with_spline(path, smoothing_factor=0):
    """B-Spline 光滑。"""
    path = np.array(path)
    # 去重
    _, idx = np.unique(path, axis=0, return_index=True)
    path = path[np.sort(idx)]

    if len(path) < 4:
        print("Not enough points for B-spline.")
        return path

    x = path[:, 0]
    y = path[:, 1]
    tck, u = splprep([x, y], s=smoothing_factor)
    unew = np.linspace(0, 1, 300)  # 采样数可调
    smooth_x, smooth_y = splev(unew, tck)
    return list(zip(smooth_x, smooth_y))


def generate_zigzag_path(grid_size, line_spacing=10):
    """
    简单 Zigzag：x 每隔 line_spacing，y 从 0-> y_max 往返。
    确保包含最后一列以覆盖整个网格。
    """
    path = []
    x_max, y_max = grid_size
    x_positions = list(range(0, x_max, line_spacing))
    
    # 确保包含最后一列
    if (x_max - 1) not in x_positions:
        x_positions.append(x_max - 1)
    
    direction = 1
    for x in x_positions:
        if direction == 1:
            y_range = range(0, y_max)
        else:
            y_range = range(y_max-1, -1, -1)
        for y in y_range:
            if x < x_max and y < y_max:
                path.append((x, y))
        direction *= -1
    return path


def detect_collisions(path, grid):
    """检测 path 中穿过障碍的段落。"""
    collision_segments = []
    in_obstacle = False
    start_idx = None
    for i, (px, py) in enumerate(path):
        x, y = int(px), int(py)
        if x >= grid.shape[0] or y >= grid.shape[1]:
            continue
        if grid[x, y] == 1:
            if not in_obstacle:
                in_obstacle = True
                start_idx = i - 1 if i>0 else i
        else:
            if in_obstacle:
                in_obstacle = False
                end_idx = i
                collision_segments.append((start_idx, end_idx))
    if in_obstacle:
        collision_segments.append((start_idx, len(path)-1))
    return collision_segments

def process_zigzag_path(grid_size, grid, distance_field, obstacle_kdtree):
    """
    主流程：生成 zigzag -> 碰撞段 -> A* -> 优化 -> B-Spline -> 拼接
    """
    path = generate_zigzag_path(grid_size, line_spacing=10)
    collision_segments = detect_collisions(path, grid)

    optimized_path = []
    idx = 0
    for (start_idx, end_idx) in collision_segments:
        # 先保存无障碍部分
        optimized_path.extend(path[idx:start_idx+1])
        start_pt = path[start_idx]
        end_pt   = path[end_idx]

        # A* 绕障
        a_star_path = a_star_search_with_potential_field(
            (int(start_pt[0]), int(start_pt[1])),
            (int(end_pt[0]),   int(end_pt[1])),
            grid, distance_field,
            influence_strength=5.0
        )
        if a_star_path:
            a_star_path = [(float(p[0]), float(p[1])) for p in a_star_path]
            # 优化
            opt_seg, _ = optimize_path_with_scipy(a_star_path, grid, obstacle_kdtree,
                                                  obs_weight=2.0,
                                                  smooth_weight=0.5,
                                                  length_weight=0.1,
                                                  max_iterations=10)
            # B-Spline 光滑
            smoothed_seg = smooth_path_with_spline(opt_seg, smoothing_factor=1.0)
            if len(smoothed_seg) > 1:
                smoothed_seg = smoothed_seg[1:]
            optimized_path.extend(smoothed_seg)
        else:
            print("No path found around obstacle.")
            return None
        idx = end_idx
    optimized_path.extend(path[idx:])
    return path, optimized_path

##############################################################################
# 2. 脏污地图 & 动态清洁仿真
##############################################################################
def generate_dirt_map(size=100, max_dirt=100):
    """随机生成脏污地图。"""
    dirt_map = np.random.randint(0, max_dirt, size=(size, size)).astype(float)
    return dirt_map

def gaussian_influence(dirt_map, center, alpha=15, sigma=5.0):
    """
    对地图在 center 位置施加高斯“清洁”影响，使得该位置附近的脏污值被降低。
    """
    rows, cols = dirt_map.shape
    x0, y0 = center
    if not (0 <= x0 < rows and 0 <= y0 < cols):
        return dirt_map

    x_coords = np.arange(cols)
    y_coords = np.arange(rows)
    xx, yy = np.meshgrid(x_coords, y_coords)  # xx: col, yy: row

    dist_sq = (xx - y0)**2 + (yy - x0)**2
    influence = alpha * np.exp(-dist_sq / (2*sigma*sigma))

    # 更新脏污地图
    dirt_map -= influence
    dirt_map = np.maximum(dirt_map, 0)  # 不允许出现负值
    return dirt_map

def simulate_cleaning(dirt_map, path, alpha=15, sigma=5, skip_step=1, dirt_threshold=5):
    """
    逐点清洁仿真，每走一步调用 gaussian_influence 减少脏污。
    - skip_step: 可以让机器人在密集路径上每隔多少点才进行一次清洁（加快演示）
    - dirt_threshold: 如果全图脏污低于此值，则提前终止
    返回 (cleaned_map, snapshots)
    """
    snapshots = []
    current_map = dirt_map.copy()

    for i, (px, py) in enumerate(path):
        # 跳过某些点
        if i % skip_step != 0:
            continue
        # 清洁
        gaussian_influence(current_map, (int(px), int(py)), alpha=alpha, sigma=sigma)
        snapshots.append(current_map.copy())

        if np.all(current_map < dirt_threshold):
            print(f"在第 {i} 步清洁完毕，地图脏污已足够低。")
            break

    return current_map, snapshots

##############################################################################
# 3. 主函数：障碍Zigzag + 脏污清洁 + 动画可视化
##############################################################################
def run_zigzag_dirt_cleaning_demo():
    grid_size = (100, 100)
    # (1) 障碍地图
    obstacle_grid = np.zeros(grid_size, dtype=int)
    generate_obstacles(obstacle_grid, num_obstacles=5, min_size=5, max_size=10)

    # (2) 脏污地图 (与障碍无关，可并行存在)
    dirt_map = generate_dirt_map(size=100, max_dirt=80)

    # (3) 计算障碍距离场 & 构建KDTree
    distance_field = calculate_distance_field_kdtree(obstacle_grid)
    obstacle_kdtree, obstacle_coords = build_kdtree(obstacle_grid)

    # (4) 获取 Zigzag 路径(带避障优化)
    result = process_zigzag_path(grid_size, obstacle_grid, distance_field, obstacle_kdtree)
    if result is None:
        print("Zigzag path planning failed.")
        return
    original_path, optimized_path = result

    # (5) 清洁仿真 (对最终路径)
    cleaned_map, snapshots = simulate_cleaning(
        dirt_map, optimized_path,
        alpha=8,     # 清洁强度
        sigma=5,      # 影响范围
        skip_step=1,  # 是否跳点
        dirt_threshold=3
    )
    print(f"总共生成了 {len(snapshots)} 帧动画。")

    # ===========================
    # 可视化：左图 障碍+路径, 右图 动画播放脏污清洁
    # ===========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (A) 左图：障碍 & 路径
    ax0 = axes[0]
    ax0.set_title("Obstacle Map + Zigzag Path")
    # 障碍显示
    ax0.imshow(obstacle_grid.T, origin='lower', cmap='gray_r')
    if obstacle_coords is not None and len(obstacle_coords) > 0:
        ax0.scatter(obstacle_coords[:, 0], obstacle_coords[:, 1], c='k', s=10, label='Obstacles')
    # 原始 & 优化后路径
    ox, oy = zip(*original_path)
    ax0.plot(ox, oy, 'b--', linewidth=1, label='Original Zigzag')
    x_opt, y_opt = zip(*optimized_path)
    ax0.plot(x_opt, y_opt, 'g-', linewidth=2, label='Optimized Path')
    ax0.set_xlim(0, grid_size[0])
    ax0.set_ylim(0, grid_size[1])
    ax0.set_aspect('equal', adjustable='box')
    ax0.legend()

    # (B) 右图：脏污地图动态清洁过程
    ax1 = axes[1]
    ax1.set_title("Dirt Cleaning Simulation")

    # 初始化展示第一帧
    vmin, vmax = 0, 80  # 根据 max_dirt 或 np.max(dirt_map)
    im = ax1.imshow(snapshots[0].T, cmap='hot', origin='lower', vmin=vmin, vmax=vmax, animated=True)
    ax1.set_xlim(0, dirt_map.shape[0])
    ax1.set_ylim(0, dirt_map.shape[1])
    ax1.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Dirt Value')

    # 动画更新函数
    def update(frame_idx):
        ax1.clear()
        ax1.set_title(f"Dirt Cleaning - Step {frame_idx+1}/{len(snapshots)}")
        # 显示脏污地图
        im_ani = ax1.imshow(snapshots[frame_idx].T, cmap='hot', origin='lower',
                            vmin=vmin, vmax=vmax)
        ax1.set_xlim(0, dirt_map.shape[0])
        ax1.set_ylim(0, dirt_map.shape[1])
        ax1.set_aspect('equal', adjustable='box')
        return [im_ani]

    ani = FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=1,  # 控制动画速度
        blit=True,     # 是否使用 blit 加速
        repeat=False
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_zigzag_dirt_cleaning_demo()

