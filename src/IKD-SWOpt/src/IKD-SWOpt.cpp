#include <iostream>
#include <vector>
#include <cmath>
#include <ikd_Tree.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <unordered_map>
#include <random>
#include <chrono>

// Include lbfgs.hpp
#include "lbfgs.hpp"

// 使用命名空间
using PointType = ikdTree_PointType;
using PointVector = KD_TREE<PointType>::PointVector;

// 全局变量
KD_TREE<PointType> ikdTree(1.0);  // 设置分辨率为 1.0
PointVector obstacles;            // 存储障碍物点云

// 节点结构体
struct Node {
    int x, y;   // 节点坐标
    float g, h; // g 为从起点到当前节点的代价，h 为启发式代价（到目标的估计）
    Node* parent; // 父节点指针

    Node(int x, int y) : x(x), y(y), g(0), h(0), parent(nullptr) {}
};

// 创建随机地图的函数
std::vector<std::vector<int>> createRandomMap(int width, int height, float obstacleDensity) {
    // 初始化地图为全 0，0 表示空地，1 表示障碍物
    std::vector<std::vector<int>> grid(height, std::vector<int>(width, 0));

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> widthDis(1, width - 2);
    std::uniform_int_distribution<> heightDis(1, height - 2);

    // 添加墙壁（地图边界）
    for (int x = 0; x < width; ++x) {
        grid[0][x] = 1;
        grid[height - 1][x] = 1;
    }
    for (int y = 0; y < height; ++y) {
        grid[y][0] = 1;
        grid[y][width - 1] = 1;
    }

    // 添加房间
    int numRooms = static_cast<int>(width * height * obstacleDensity / 1000);
    for (int i = 0; i < numRooms; ++i) {
        int roomWidth = widthDis(gen);
        int roomHeight = heightDis(gen);
        int roomX = widthDis(gen);
        int roomY = heightDis(gen);

        for (int yy = roomY; yy < std::min(roomY + roomHeight, height - 1); ++yy) {
            for (int xx = roomX; xx < std::min(roomX + roomWidth, width - 1); ++xx) {
                grid[yy][xx] = 1; // 设置为障碍物
            }
        }
    }

    // 添加门
    int numDoors = static_cast<int>(numRooms * 1.5);
    for (int i = 0; i < numDoors; ++i) {
        int doorX = widthDis(gen);
        int doorY = heightDis(gen);
        if (grid[doorY][doorX] == 1) {
            grid[doorY][doorX] = 0; // 设置为空地
        }
    }

    // 添加家具
    int numFurniture = static_cast<int>(width * height * obstacleDensity / 100);
    for (int i = 0; i < numFurniture; ++i) {
        int furnitureWidth = widthDis(gen) / 10 + 1;
        int furnitureHeight = heightDis(gen) / 10 + 1;
        int furnitureX = widthDis(gen);
        int furnitureY = heightDis(gen);

        for (int yy = furnitureY; yy < std::min(furnitureY + furnitureHeight, height - 1); ++yy) {
            for (int xx = furnitureX; xx < std::min(furnitureX + furnitureWidth, width - 1); ++xx) {
                if (dis(gen) < 0.8) {
                    grid[yy][xx] = 1; // 设置为障碍物
                }
            }
        }
    }

    return grid;
}

// 构建 KD-Tree 的函数
void buildKDTree(const std::vector<std::vector<int>>& grid, float resolution) {
    int width = grid[0].size();
    int height = grid.size();

    obstacles.clear();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (grid[y][x] == 1) {
                PointType new_point;
                new_point.x = x * resolution;
                new_point.y = y * resolution;
                obstacles.emplace_back(new_point);
            }
        }
    }

    ikdTree.Build(obstacles); // 构建 KD-Tree
}

// 生成距离场的函数
std::vector<std::vector<float>> generateDistanceField(const std::vector<std::vector<int>>& grid, float resolution) {
    int width = grid[0].size();
    int height = grid.size();
    std::vector<std::vector<float>> distField(height, std::vector<float>(width, 0.0));

    // 遍历地图中的每个点，计算到最近障碍物的距离
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            PointType query{x * resolution, y * resolution};
            PointVector result(1);
            std::vector<float> dists;
            ikdTree.Nearest_Search(query, 1, result, dists);
            distField[y][x] = std::sqrt(dists[0]);  // 存储实际距离
        }
    }

    return distField;
}

// 可视化函数（增加了 int testIndex，用于保存文件时区分不同测试）
void visualize(const std::vector<std::vector<int>>& grid,
               const std::vector<std::vector<float>>& distField,
               const std::vector<Node*>& path = {},
               const std::vector<Node*>& interpolated_path = {},
               const std::vector<Node*>& optimized_path = {},
               const std::vector<Node*>& smoothed_path = {},
               int testIndex = 0)
{
    int width = grid[0].size();
    int height = grid.size();

    // 创建空白图像
    cv::Mat gridImg(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat distFieldImg(height, width, CV_8UC1);

    float maxDist = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (grid[y][x] == 1) {
                gridImg.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // 障碍物为黑色
            }
            maxDist = std::max(maxDist, distField[y][x]);
        }
    }

    // 生成距离场的可视化图像
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            distFieldImg.at<uchar>(y, x) =
                static_cast<uchar>(255 - (distField[y][x] / maxDist) * 255);
        }
    }

    cv::applyColorMap(distFieldImg, distFieldImg, cv::COLORMAP_JET);

    // 创建独立的图像来显示不同的路径
    cv::Mat pathImg = gridImg.clone();
    cv::Mat interpolatedImg = gridImg.clone();
    cv::Mat optimizedImg = gridImg.clone();
    cv::Mat smoothedImg = gridImg.clone();

    // 绘制原始 A* 路径
    for (const auto& node : path) {
        gridImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(0, 0, 255);   // 红色
        pathImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(0, 0, 255);  // 红色
    }

    // 绘制插值后的路径
    for (const auto& node : interpolated_path) {
        gridImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(0, 255, 255);   // 黄色
        interpolatedImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(0, 255, 255); // 黄色
    }

    // 绘制优化后的路径
    for (const auto& node : optimized_path) {
        gridImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(0, 255, 0);   // 绿色
        optimizedImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(0, 255, 0); // 绿色
    }

    // 绘制平滑后的路径
    for (const auto& node : smoothed_path) {
        gridImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(255, 0, 0);   // 蓝色
        smoothedImg.at<cv::Vec3b>(node->y, node->x) = cv::Vec3b(255, 0, 0); // 蓝色
    }

    // 用不同的符号标记起点和终点
    if (!path.empty()) {
        // 起点（方块）
        const Node* start_node = path.front();
        cv::rectangle(gridImg,
                      cv::Point(start_node->x - 3, start_node->y - 3),
                      cv::Point(start_node->x + 3, start_node->y + 3),
                      cv::Scalar(255, 255, 0), -1);
        cv::rectangle(pathImg,
                      cv::Point(start_node->x - 3, start_node->y - 3),
                      cv::Point(start_node->x + 3, start_node->y + 3),
                      cv::Scalar(255, 255, 0), -1);
        cv::rectangle(interpolatedImg,
                      cv::Point(start_node->x - 3, start_node->y - 3),
                      cv::Point(start_node->x + 3, start_node->y + 3),
                      cv::Scalar(255, 255, 0), -1);
        cv::rectangle(optimizedImg,
                      cv::Point(start_node->x - 3, start_node->y - 3),
                      cv::Point(start_node->x + 3, start_node->y + 3),
                      cv::Scalar(255, 255, 0), -1);
        cv::rectangle(smoothedImg,
                      cv::Point(start_node->x - 3, start_node->y - 3),
                      cv::Point(start_node->x + 3, start_node->y + 3),
                      cv::Scalar(255, 255, 0), -1);

        // 终点（三角形）
        const Node* goal_node = path.back();
        cv::Point pts[1][3];
        pts[0][0] = cv::Point(goal_node->x, goal_node->y - 4);
        pts[0][1] = cv::Point(goal_node->x - 3, goal_node->y + 3);
        pts[0][2] = cv::Point(goal_node->x + 3, goal_node->y + 3);
        const cv::Point* ppt[1] = { pts[0] };
        int npt[] = { 3 };
        cv::fillPoly(gridImg, ppt, npt, 1, cv::Scalar(255, 0, 255));
        cv::fillPoly(pathImg, ppt, npt, 1, cv::Scalar(255, 0, 255));
        cv::fillPoly(interpolatedImg, ppt, npt, 1, cv::Scalar(255, 0, 255));
        cv::fillPoly(optimizedImg, ppt, npt, 1, cv::Scalar(255, 0, 255));
        cv::fillPoly(smoothedImg, ppt, npt, 1, cv::Scalar(255, 0, 255));
    }

    // 显示图像
    cv::imshow("Combined Paths", gridImg);
    cv::imshow("Original A* Path", pathImg);
    cv::imshow("Interpolated Path", interpolatedImg);
    cv::imshow("Optimized Path", optimizedImg);
    cv::imshow("Smoothed Path", smoothedImg);
    cv::imshow("Distance Field", distFieldImg);

    // —— 以下是保存图片的新增代码 ——
    // 根据 testIndex 拼接不同的文件名前缀，避免覆盖
    std::string prefix = "test" + std::to_string(testIndex) + "_";

    cv::imwrite(prefix + "CombinedPaths.png", gridImg);
    cv::imwrite(prefix + "OriginalAStarPath.png", pathImg);
    cv::imwrite(prefix + "InterpolatedPath.png", interpolatedImg);
    cv::imwrite(prefix + "OptimizedPath.png", optimizedImg);
    cv::imwrite(prefix + "SmoothedPath.png", smoothedImg);
    cv::imwrite(prefix + "DistanceField.png", distFieldImg);

    // 这里可根据需要调 waitKey 参数，0 表示等待按键，1 表示稍作停留即可
    cv::waitKey(1);
}

// 清理函数，用于释放内存
void cleanUp(std::vector<std::vector<Node*>>& nodeMap) {
    for (auto& row : nodeMap) {
        for (auto& node : row) {
            if (node != nullptr) {
                delete node;
                node = nullptr;
            }
        }
    }
}

// A* 算法实现
std::vector<Node*> aStar(const std::vector<std::vector<int>>& grid,
                         const std::vector<std::vector<float>>& distField,
                         int startX, int startY,
                         int goalX, int goalY,
                         float influenceStrength = 0.5,
                         float distanceWeight = 1.0)
{
    int width = grid[0].size();
    int height = grid.size();

    // 初始化关闭列表和打开列表
    std::vector<std::vector<bool>> closed(height, std::vector<bool>(width, false));
    std::vector<std::vector<bool>> open(height, std::vector<bool>(width, false));
    std::vector<std::vector<Node*>> nodeMap(height, std::vector<Node*>(width, nullptr));

    // 优先队列，按照 f = g + h 的值进行排序
    auto cmp = [](Node* a, Node* b) { return a->g + a->h > b->g + b->h; };
    std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> openQueue(cmp);

    // 创建起点和终点节点
    Node* startNode = new Node(startX, startY);
    Node* goalNode = new Node(goalX, goalY);

    // 将起点加入打开列表
    openQueue.push(startNode);
    open[startY][startX] = true;
    nodeMap[startY][startX] = startNode;

    while (!openQueue.empty()) {
        Node* currentNode = openQueue.top();
        openQueue.pop();
        closed[currentNode->y][currentNode->x] = true;

        // 如果到达目标节点，生成路径
        if (currentNode->x == goalX && currentNode->y == goalY) {
            std::vector<Node*> path;
            while (currentNode != nullptr) {
                path.push_back(currentNode);
                currentNode = currentNode->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // 遍历邻居节点
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;

                int newX = currentNode->x + i;
                int newY = currentNode->y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height &&
                    grid[newY][newX] == 0 && !closed[newY][newX])
                {
                    // 原始走路代价
                    float move_cost = std::hypot(i, j);

                    // ---- 使用距离场，离障碍越近，代价越大 ----
                    float dist_to_obs = distField[newY][newX];
                    float epsilon = 1e-3;
                    float dist_cost = 1.f / (dist_to_obs + epsilon);

                    float g = currentNode->g + move_cost + distanceWeight * dist_cost;
                    float h = std::hypot(newX - goalX, newY - goalY);

                    if (!open[newY][newX]) {
                        Node* newNode = new Node(newX, newY);
                        newNode->g = g;
                        newNode->h = h;
                        newNode->parent = currentNode;
                        openQueue.push(newNode);
                        open[newY][newX] = true;
                        nodeMap[newY][newX] = newNode;
                    } else {
                        Node* oldNode = nodeMap[newY][newX];
                        if (g < oldNode->g) {
                            oldNode->g = g;
                            oldNode->h = h;
                            oldNode->parent = currentNode;
                        }
                    }
                }
            }
        }
    }

    cleanUp(nodeMap);
    return {};
}

// 在路径点之间进行插值的函数
std::vector<Node*> interpolatePath(const std::vector<Node*>& path, int num_points) {
    std::vector<Node*> interpolatedPath;

    double total_length = 0.0;
    std::vector<double> lengths;
    lengths.push_back(0.0);

    // 计算路径的总长度
    for (size_t i = 1; i < path.size(); ++i) {
        double dx = path[i]->x - path[i - 1]->x;
        double dy = path[i]->y - path[i - 1]->y;
        double dist = std::sqrt(dx * dx + dy * dy);
        total_length += dist;
        lengths.push_back(total_length);
    }

    // 在路径上进行均匀插值
    for (int i = 0; i <= num_points; ++i) {
        double target_length = i * total_length / num_points;
        size_t idx = 0;
        while (idx < lengths.size() - 1 && lengths[idx + 1] < target_length) {
            ++idx;
        }
        double ratio = (target_length - lengths[idx]) /
                       (lengths[idx + 1] - lengths[idx]);
        double x = path[idx]->x +
                   ratio * (path[idx + 1]->x - path[idx]->x);
        double y = path[idx]->y +
                   ratio * (path[idx + 1]->y - path[idx]->y);

        Node* node = new Node(static_cast<int>(x), static_cast<int>(y));
        interpolatedPath.push_back(node);
    }

    return interpolatedPath;
}

// 优化数据结构体
struct OptimizationData {
    std::vector<std::pair<double, double>> path_points; // 路径点坐标
    int num_points; // 路径点数量
    int width;
    int height;
    KD_TREE<PointType>* ikdTreePtr;
    const std::vector<std::vector<int>>* grid;
    double obs_weight;
    double smooth_weight;
    double length_weight;
};

// 评价函数，用于 LBFGS 优化
double evaluate(void* instance, const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    OptimizationData* data = reinterpret_cast<OptimizationData*>(instance);
    const int num_points = data->num_points;
    const int n = x.size() / 2;

    // 重建路径，包括起点和终点
    std::vector<std::pair<double, double>> path_points;
    path_points.push_back(data->path_points.front()); // 起点
    for (int i = 0; i < n; ++i) {
        double xi = x(2 * i);
        double yi = x(2 * i + 1);
        path_points.emplace_back(xi, yi);
    }
    path_points.push_back(data->path_points.back()); // 终点

    double total_cost = 0.0;
    g.setZero(x.size());

    // 获取地图尺寸
    int grid_width = data->grid->front().size();
    int grid_height = data->grid->size();

    // 平滑度代价
    for (int i = 1; i < num_points - 1; ++i) {
        const auto& xi_prev = path_points[i - 1];
        const auto& xi = path_points[i];
        const auto& xi_next = path_points[i + 1];
        double dx = xi_prev.first - 2 * xi.first + xi_next.first;
        double dy = xi_prev.second - 2 * xi.second + xi_next.second;
        double smoothness = dx * dx + dy * dy;
        total_cost += data->smooth_weight * smoothness;

        // 梯度计算
        if (i - 1 > 0) {
            int idx_prev = (i - 2);
            g(2 * idx_prev) += 2 * data->smooth_weight * dx;
            g(2 * idx_prev + 1) += 2 * data->smooth_weight * dy;
        }
        // 更新 xi 的梯度
        int idx = (i - 1);
        g(2 * idx) += -4 * data->smooth_weight * dx;
        g(2 * idx + 1) += -4 * data->smooth_weight * dy;

        if (i < num_points - 1) {
            int idx_next = (i < n) ? i : n - 1;
            if (idx_next >= 0 && idx_next < n) {
                g(2 * idx_next) += 2 * data->smooth_weight * dx;
                g(2 * idx_next + 1) += 2 * data->smooth_weight * dy;
            }
        }
    }

    // 避障代价
    for (int i = 1; i < num_points - 1; ++i) {
        const auto& xi = path_points[i];

        // 检查 xi 是否在地图范围内
        int xi_int_x = static_cast<int>(std::round(xi.first));
        int xi_int_y = static_cast<int>(std::round(xi.second));
        if (xi_int_x < 0 || xi_int_x >= grid_width ||
            xi_int_y < 0 || xi_int_y >= grid_height)
        {
            // 超出地图范围，给予高惩罚
            total_cost += 1e6;
            continue;
        }

        // 检查 xi 是否在障碍物内
        if ((*data->grid)[xi_int_y][xi_int_x] == 1) {
            // 在障碍物内，给予高惩罚
            total_cost += 1e6;
            continue;
        }

        // 使用 KD-Tree 计算到最近障碍物的距离
        PointType query{static_cast<float>(xi.first), static_cast<float>(xi.second)};
        PointVector result(1);
        std::vector<float> dists;
        data->ikdTreePtr->Nearest_Search(query, 1, result, dists);
        double dist = std::sqrt(dists[0]) + 1e-6; // 避免除零

        // 避障代价，使用指数衰减
        double obs_cost = data->obs_weight * std::exp(-dist);
        total_cost += obs_cost;

        // 梯度计算
        int idx = (i - 1); // x 和 g 中的索引
        double coeff = -data->obs_weight * std::exp(-dist) / dist;
        double dx = xi.first - result[0].x;
        double dy = xi.second - result[0].y;
        g(2 * idx)     += coeff * dx / dist;
        g(2 * idx + 1) += coeff * dy / dist;
    }

    // 路径长度代价
    for (int i = 1; i < num_points; ++i) {
        const auto& xi_prev = path_points[i - 1];
        const auto& xi = path_points[i];
        double dx = xi.first - xi_prev.first;
        double dy = xi.second - xi_prev.second;
        double length = std::sqrt(dx * dx + dy * dy) + 1e-6; // 避免除零
        total_cost += data->length_weight * length;

        // 梯度计算
        if (i - 1 > 0) {
            int idx_prev = (i - 2);
            double coeff = data->length_weight / length;
            g(2 * idx_prev)     += -coeff * dx;
            g(2 * idx_prev + 1) += -coeff * dy;
        }
        if (i - 1 < n) {
            int idx = i - 1;
            double coeff = data->length_weight / length;
            g(2 * idx)     +=  coeff * dx;
            g(2 * idx + 1) +=  coeff * dy;
        }
    }

    return total_cost;
}

// 使用 De Boor 算法的 B 样条曲线拟合函数
std::vector<std::pair<double, double>> bsplineFit(
    const std::vector<std::pair<double, double>>& controlPoints,
    int degree, int num_samples)
{
    int n = static_cast<int>(controlPoints.size()) - 1;

    // 生成节点向量（clamped）
    int m = n + degree + 1;
    std::vector<double> knotVector(m);

    for (int i = 0; i <= degree; ++i) {
        knotVector[i] = 0.0;
    }
    for (int i = degree + 1; i <= n; ++i) {
        knotVector[i] = static_cast<double>(i - degree) /
                        (n - degree + 1);
    }
    for (int i = n + 1; i < m; ++i) {
        knotVector[i] = 1.0;
    }

    // 生成参数 u
    std::vector<double> u_list(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        u_list[i] = static_cast<double>(i) / (num_samples - 1);
    }

    // 使用 De Boor 算法计算 B 样条曲线上的点
    std::vector<std::pair<double, double>> bspline_points;
    for (double u : u_list) {
        // 寻找满足 knotVector[k] <= u < knotVector[k+1] 的 k
        int k;
        if (u == 1.0) {
            k = n;
        } else {
            for (k = degree; k < n + 1; ++k) {
                if (u >= knotVector[k] && u < knotVector[k + 1]) {
                    break;
                }
            }
        }

        // 初始化 d[i] = controlPoints[k - degree + i]
        std::vector<std::pair<double, double>> d(degree + 1);
        for (int i = 0; i <= degree; ++i) {
            d[i] = controlPoints[k - degree + i];
        }

        // 递归计算
        for (int r = 1; r <= degree; ++r) {
            for (int i = degree; i >= r; --i) {
                double denom = knotVector[k + i - r + 1] -
                               knotVector[k - degree + i];
                double alpha = (u - knotVector[k - degree + i]) / denom;
                d[i].first =
                    (1.0 - alpha) * d[i - 1].first + alpha * d[i].first;
                d[i].second =
                    (1.0 - alpha) * d[i - 1].second + alpha * d[i].second;
            }
        }

        bspline_points.push_back(d[degree]);
    }

    return bspline_points;
}

// 主函数
int main() {
    int width = 800;
    int height = 800;
    float resolution = 1.0;
    float obstacleDensity = 0.01;
    int numTests = 3;  // 这里可以修改测试次数

    // 随机数生成器，用于生成起点和终点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> widthDis(0, width - 1);
    std::uniform_int_distribution<> heightDis(0, height - 1);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numTests; ++i) {
        // 创建随机地图
        auto grid = createRandomMap(width, height, obstacleDensity);
        // 构建 KD-Tree
        buildKDTree(grid, resolution);
        // 生成距离场
        auto distField = generateDistanceField(grid, resolution);

        // 随机生成起点和终点，确保不在障碍物内
        int startX = widthDis(gen), startY = heightDis(gen);
        int goalX = widthDis(gen), goalY = heightDis(gen);
        while (grid[startY][startX] == 1 || grid[goalY][goalX] == 1) {
            startX = widthDis(gen);
            startY = heightDis(gen);
            goalX = widthDis(gen);
            goalY = heightDis(gen);
        }

        // 使用 A* 算法寻找初始路径
        auto path = aStar(grid, distField, startX, startY, goalX, goalY, 1.0, 1.0);

        // 如果找到路径，继续处理
        if (!path.empty()) {
            // 在路径点之间进行插值
            int interpolated_points_count = 200; // 插值后的点数
            auto interpolated_path = interpolatePath(path, interpolated_points_count);

            // 提取插值路径的坐标
            std::vector<std::pair<double, double>> path_points;
            for (const auto& node : interpolated_path) {
                path_points.emplace_back(node->x, node->y);
            }

            // 优化变量（去除起点和终点）
            int num_points = static_cast<int>(path_points.size());
            Eigen::VectorXd x((num_points - 2) * 2);
            for (int j = 1; j < num_points - 1; ++j) {
                x(2 * (j - 1)) = path_points[j].first;
                x(2 * (j - 1) + 1) = path_points[j].second;
            }

            // 设置优化数据
            OptimizationData data;
            data.path_points = path_points;
            data.num_points = num_points;
            data.width = width;
            data.height = height;
            data.ikdTreePtr = &ikdTree;
            data.grid = &grid;
            data.obs_weight = 1000.0;    // 避障权重
            data.smooth_weight = 0.8;    // 平滑权重
            data.length_weight = 1.0;    // 路径长度权重

            // 设置 L-BFGS 参数
            lbfgs::lbfgs_parameter_t param;
            param.mem_size = 32;
            param.max_iterations = 50;
            param.g_epsilon = 1e-6;

            // 运行优化器
            double fx;
            int ret = lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &data, param);

            // 检查结果
            if (ret == lbfgs::LBFGS_CONVERGENCE || ret == lbfgs::LBFGS_STOP) {
                std::cout << "Optimization succeeded. Final cost: " << fx << std::endl;
            } else {
                std::cout << "Optimization failed with status: "
                          << lbfgs::lbfgs_strerror(ret) << std::endl;
            }

            // 更新 path_points
            for (int j = 1; j < num_points - 1; ++j) {
                path_points[j].first = x(2 * (j - 1));
                path_points[j].second = x(2 * (j - 1) + 1);
            }

            // 更新优化后的路径节点
            std::vector<Node*> optimized_path;
            optimized_path.reserve(path_points.size());
            for (auto& p : path_points) {
                int x_coord = static_cast<int>(std::round(p.first));
                int y_coord = static_cast<int>(std::round(p.second));
                Node* node = new Node(x_coord, y_coord);
                optimized_path.push_back(node);
            }

            // 使用 De Boor 算法的 B 样条对优化后的路径进行平滑
            int bspline_degree = 3;     // 三次 B 样条
            int bspline_samples = 3000; // 采样点数
            auto bspline_points = bsplineFit(path_points, bspline_degree, bspline_samples);

            // 创建平滑后的路径节点
            std::vector<Node*> smoothed_path;
            smoothed_path.reserve(bspline_points.size());
            for (const auto& point : bspline_points) {
                int x_coord = static_cast<int>(std::round(point.first));
                int y_coord = static_cast<int>(std::round(point.second));
                // 确保坐标在地图范围内
                if (x_coord >= 0 && x_coord < width &&
                    y_coord >= 0 && y_coord < height)
                {
                    Node* node = new Node(x_coord, y_coord);
                    smoothed_path.push_back(node);
                }
            }

            // 可视化所有路径，并保存图片(带 i 做区分)
            visualize(grid, distField, path, interpolated_path, optimized_path, smoothed_path, i);

            // 等待按键以关闭窗口（可视化函数里也有 waitKey(1)，所以可根据需要调整）
            cv::waitKey(0);

            // 清理内存
            for (auto node : smoothed_path) {
                delete node;
            }
            for (auto node : optimized_path) {
                delete node;
            }
            for (auto node : interpolated_path) {
                delete node;
            }
            for (auto node : path) {
                delete node;
            }
        } else {
            std::cout << "No path found." << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Average time per test: "
              << duration.count() / static_cast<double>(numTests)
              << " ms" << std::endl;

    return 0;
}
