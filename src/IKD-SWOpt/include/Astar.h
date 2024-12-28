// AStar.h
#pragma once

#include "DistanceField.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct Node2D {
    int x, y;
    float g, h;
    Node2D* parent;

    Node2D(int nx, int ny);
};

class AStar {
public:
    AStar(const std::vector<std::vector<int>>& grid, float res);

    std::vector<cv::Point> findPath(int startX, int startY, int goalX, int goalY);

private:
    std::vector<std::vector<int>> grid;
    int width, height;
    float resolution;
    DistanceField distanceField;

    float heuristic(int x1, int y1, int x2, int y2);
    std::vector<Node2D*> getNeighbors(Node2D* node);
};