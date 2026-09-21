#include "quantizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace hvs::detail {

void kmeans(const cv::Mat1f &samples, cv::Mat1f &centers, int count) {
    const int num_samples = samples.rows, dim = samples.cols;
    centers.create(count, dim);
    cv::Mat1f sums(count, dim);
    std::vector<int> sizes(count);
    for (int center = 0; center < count; ++center)
        samples.row(int((num_samples - 1) * (double(center) / count))).copyTo(centers.row(center));
    // Native evenly spaced initialization, five Lloyd steps, unchanged empty centers.
    for (int round = 0; round < 5; ++round) {
        sums.setTo(0);
        std::fill(sizes.begin(), sizes.end(), 0);
        for (int row = 0; row < num_samples; ++row) {
            const int center = nearest(samples[row], centers);
            for (int dim_idx = 0; dim_idx < dim; ++dim_idx)
                sums(center, dim_idx) += samples(row, dim_idx);
            ++sizes[center];
        }
        for (int center = 0; center < count; ++center)
            if (sizes[center])
                for (int dim_idx = 0; dim_idx < dim; ++dim_idx)
                    centers(center, dim_idx) = sums(center, dim_idx) / sizes[center];
    }
}

void merge_quantizers(const cv::Mat1f &candidates, cv::Mat1f &centers, unsigned char *merge,
                      const cv::Mat1f &samples) {
    const int dim = samples.cols, half = dim / 2;
    constexpr int pairs = codewords * codewords;
    cv::Mat1f left(codewords, half), right(codewords, half);
    for (int center = 0; center < codewords; ++center) {
        candidates.row(center * codewords + center).colRange(0, half).copyTo(left.row(center));
        candidates.row(center * codewords + center).colRange(half, dim).copyTo(right.row(center));
    }
    std::vector<int> weight(pairs), assigned(samples.rows), selected(codewords);
    for (int row = 0; row < samples.rows; ++row)
        ++weight[nearest(samples[row], left) * codewords + nearest(samples[row] + half, right)];
    kmeans(samples, centers);
    for (int row = 0; row < samples.rows; ++row)
        assigned[row] = nearest(samples[row], centers);
    std::vector<RankedPoint> ranked(pairs);
    for (int center = 0; center < codewords; ++center) {
        for (int pair = 0; pair < pairs; ++pair)
            ranked[pair] = {pair, squared_l2(candidates[pair], centers[center], dim)};
        // The native selection examines only the five nearest Cartesian-product pairs.
        std::partial_sort(ranked.begin(), ranked.begin() + 5, ranked.end());
        float best_error = 0;
        for (int rank = 0; rank < 5; ++rank) {
            const int candidate = ranked[rank].id;
            float error = 0;
            for (int row = 0; row < samples.rows; ++row)
                if (assigned[row] == center)
                    for (int dim_idx = 0; dim_idx < dim; ++dim_idx)
                        error += (candidates(candidate, dim_idx) - samples(row, dim_idx)) *
                                 (candidates(candidate, dim_idx) - samples(row, dim_idx));
            if (rank == 0 || error < best_error) {
                selected[center] = candidate;
                best_error = error;
            }
        }
        candidates.row(selected[center]).copyTo(centers.row(center));
    }
    std::sort(selected.begin(), selected.end());
    std::vector<float> mean(dim), probability(pairs);
    int occupied = 0;
    for (int pair = 0; pair < pairs; ++pair) {
        if (!weight[pair])
            continue;
        for (int dim_idx = 0; dim_idx < dim; ++dim_idx)
            mean[dim_idx] += candidates(pair, dim_idx);
        ++occupied;
    }
    for (float &value : mean)
        value /= occupied;
    float total = 0;
    for (int pair = 0; pair < pairs; ++pair) {
        for (int dim_idx = 0; dim_idx < dim; ++dim_idx)
            probability[pair] += weight[pair] * (mean[dim_idx] - candidates(pair, dim_idx)) *
                                 (mean[dim_idx] - candidates(pair, dim_idx));
        total += probability[pair];
    }
    if (!(total > 0) || !std::isfinite(total))
        throw std::runtime_error("HVS needs non-degenerate training samples to merge quantizers");
    // Preserve the native integer uniform term and importance-sampling RNG order.
    for (float &value : probability)
        value = 1 / pairs / 2.0f + value / total / 2.0f;
    for (int pair = 1; pair < pairs; ++pair)
        probability[pair] += probability[pair - 1];
    for (int center = 1; center < codewords; ++center)
        for (int earlier = 0; earlier < center; ++earlier)
            if (selected[center] == selected[earlier]) {
                const double draw = std::rand() / double(RAND_MAX);
                for (int pair = 0; pair < pairs; ++pair)
                    if (draw <= probability[pair]) {
                        selected[center] = pair;
                        --center;
                        break;
                    }
                break;
            }
    std::sort(selected.begin(), selected.end());
    for (int center = 0; center < codewords; ++center) {
        merge[2 * center] = selected[center] / codewords;
        merge[2 * center + 1] = selected[center] % codewords;
    }
}

void build_start_merges(const std::vector<cv::Mat1f> &books, int block, unsigned char *merge,
                        const cv::Mat1f &samples) {
    cv::Mat1f centers;
    kmeans(samples, centers, start_centroids);
    const int subdim = books[block * merged_books].cols;
    for (int center = 0; center < start_centroids; ++center)
        for (int part = 0; part < merged_books; ++part)
            merge[center * merged_books + part] =
                nearest(centers[center] + part * subdim, books[block * merged_books + part]);
}

} // namespace hvs::detail
