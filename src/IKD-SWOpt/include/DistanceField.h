// DistanceField.h
#pragma once

#include <ikd_Tree.h>
#include <vector>

using PointType = ikdTree_PointType;
using PointVector = KD_TREE<PointType>::PointVector;

class DistanceField {
public:
    DistanceField(const std::vector<std::vector<int>>& grid, float res);

    float getDistance(int x, int y) const;
    size_t getMemoryUsage() ;

private:
    std::vector<std::vector<float>> field;
    int width, height;
    float resolution;
    KD_TREE<PointType> kdTree;
};