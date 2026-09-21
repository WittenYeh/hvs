#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace hvs::detail {

inline constexpr int codewords = 256;
inline constexpr int start_centroids = 16;
inline constexpr int merged_books = 4;

struct RankedPoint {
    int id;
    float distance;
    bool operator<(const RankedPoint &other) const {
        if (distance < other.distance)
            return true;
        if (distance > other.distance)
            return false;
        return id < other.id;
    }
};

// Keep the original accumulation order and tie handling during training.
inline float squared_l2(const float *left, const float *right, int dim) {
    float distance = 0;
    for (int dim_idx = 0; dim_idx < dim; ++dim_idx)
        distance += (left[dim_idx] - right[dim_idx]) * (left[dim_idx] - right[dim_idx]);
    return distance;
}

inline int nearest(const float *point, const cv::Mat1f &centers) {
    int best = 0;
    float distance = squared_l2(point, centers[0], centers.cols);
    for (int center = 1; center < centers.rows; ++center) {
        const float candidate = squared_l2(point, centers[center], centers.cols);
        if (candidate < distance) {
            best = center;
            distance = candidate;
        }
    }
    return best;
}

// Extracted from sift_1b.cpp's kmeans_/kmeans0_/sub_kmeans_ routines.
// Both the original file-based demo and the in-memory index use these functions.
void kmeans(const cv::Mat1f &samples, cv::Mat1f &centers, int count = codewords);
void merge_quantizers(const cv::Mat1f &candidates, cv::Mat1f &centers, unsigned char *merge,
                      const cv::Mat1f &samples);
void build_start_merges(const std::vector<cv::Mat1f> &books, int block, unsigned char *merge,
                        const cv::Mat1f &samples);

} // namespace hvs::detail
