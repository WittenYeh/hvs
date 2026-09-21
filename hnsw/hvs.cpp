#include "hvs.hpp"
#include "quantizer.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <omp.h>
#include "hnswlib/hnswlib.h"

#undef L
#undef OFF
#undef min_book
#undef cen
#undef nnum
#undef fan
#undef KK
#undef size_n

namespace hvs {
namespace {
using namespace detail;
using Graph = hnswlib::HierarchicalNSW<float>;
constexpr int start_books = 4, entry_points = 10, offset = 3;

// Non-owning row pointers are only needed at the original graph API boundary.
template <class T> std::vector<T *> row_pointers(cv::Mat_<T> &matrix) {
    std::vector<T *> rows(matrix.rows);
    for (int row = 0; row < matrix.rows; ++row)
        rows[row] = matrix[row];
    return rows;
}

struct Representative {
    const unsigned char *code;
    int id;
    float error;
};
} // namespace

struct Index::Impl {
    const int n, dimension, levels, width;
    std::vector<int> lengths, subdimensions, counts;
    std::vector<std::vector<unsigned char>> merges;
    std::vector<std::vector<char>> flags;
    std::vector<std::vector<unsigned>> transitions;
    std::array<unsigned char, start_books * start_centroids * merged_books> start_merges{};
    std::vector<unsigned> connections;
    std::vector<float> packed_codebooks;
    hnswlib::L2Space space;
    std::unique_ptr<Graph> graph;

    Impl(int size, int dim, int depth)
        : n(size), dimension(dim), levels(depth),
          width(((dim + (1 << (depth + offset)) - 1) >> (depth + offset)) << (depth + offset)),
          lengths(depth), subdimensions(depth), counts(depth), merges(depth), flags(depth),
          transitions(depth),
          connections(start_centroids * start_centroids * start_centroids * start_centroids * entry_points),
          space(dim, width) {
        for (int level = 0; level < levels; ++level) {
            lengths[level] = 1 << (levels - level + offset);
            subdimensions[level] = width / lengths[level];
            merges[level].resize(lengths[level] * 2 * codewords);
        }
    }

    void build(const float *base, const BuildConfig &config);
};

void Index::Impl::build(const float *base, const BuildConfig &config) {
    const int samples = std::min(config.training_samples, n);
    const int blocks = lengths[0], subdim = subdimensions[0];
    if (samples < width)
        throw std::invalid_argument("HVS needs at least as many training samples as padded dimensions");
    std::srand(config.seed);
    cv::setRNGSeed(config.seed);
    graph =
        std::make_unique<Graph>(levels, &space, n, dimension, config.M, config.ef_construction, config.seed);
    cv::Mat1f training(samples, width, 0.0f), data(n, width, 0.0f);
    for (int row = 0; row < samples; ++row)
        std::copy_n(base + size_t(row) * (n / samples) * dimension, dimension, training[row]);
    for (int row = 0; row < n; ++row)
        std::copy_n(base + size_t(row) * dimension, dimension, data[row]);
    std::vector<std::vector<cv::Mat1f>> books(levels);
    std::vector<cv::Mat1b> codes(levels);
    for (int level = 0; level < levels; ++level) {
        books[level].resize(lengths[level]);
        codes[level].create(n, lengths[level]);
    }

    graph->addPoint(base, 0, nullptr, -1, true);
#pragma omp parallel for
    for (int row = 1; row < n; ++row)
        graph->addPoint(base + size_t(row) * dimension, row, nullptr, -1, false);
    std::vector<float> density(n);
    cv::Mat1i labels(levels + 1, n);
    auto label_rows = row_pointers(labels);
    graph->est_density(density.data(), n, dimension);
    graph->permutation(label_rows.data(), n, levels, nullptr, -1);
    graph->finishLayer(-1);

    // Native PCA initialization balances eigenvalue products across subspaces.
    cv::Mat1f raw = training.t();
    cv::PCA pca(raw, cv::noArray(), cv::PCA::DATA_AS_COL, width);
    cv::Mat1f initial_rotation(width, width), rotation, rotated;
    std::vector<RankedPoint> products(blocks);
    std::vector<int> used(blocks), order(width);
    for (int dim_idx = 0; dim_idx < width; ++dim_idx) {
        const float value = pca.eigenvalues.at<float>(dim_idx);
        if (dim_idx < blocks) {
            products[dim_idx] = {dim_idx, value};
            used[dim_idx] = 1;
            order[dim_idx] = dim_idx * subdim;
        } else {
            std::sort(products.begin(), products.end());
            for (auto &product : products)
                if (used[product.id] < subdim) {
                    order[dim_idx] = product.id * subdim + used[product.id]++;
                    product.distance *= value;
                    break;
                }
        }
    }
    for (int row = 0; row < width; ++row) {
        const float *eigenvector = pca.eigenvectors.ptr<float>(row);
        float norm = 0;
        for (int dim_idx = 0; dim_idx < width; ++dim_idx)
            norm += eigenvector[dim_idx] * eigenvector[dim_idx];
        for (int dim_idx = 0; dim_idx < width; ++dim_idx)
            initial_rotation(order[row], dim_idx) = eigenvector[dim_idx] / std::sqrt(norm);
    }
    cv::gemm(initial_rotation, raw, 1, cv::noArray(), 0, rotated);
    cv::transpose(rotated, training);
    auto train_bottom = [&] {
        for (int block = 0; block < blocks; ++block)
            kmeans(training.colRange(block * subdim, (block + 1) * subdim), books[0][block]);
    };
    train_bottom();

    // Two native OPQ passes, ten centroid updates per pass and one SVD rotation.
    cv::Mat1f encoded(width, samples);
    std::vector<int> assigned(samples), cluster_sizes(codewords);
    for (int pass = 0; pass < 2; ++pass) {
        for (int block = 0; block < blocks; ++block) {
            auto &centers = books[0][block];
            for (int round = 0; round < 10; ++round) {
                for (int row = 0; row < samples; ++row)
                    assigned[row] = nearest(training[row] + block * subdim, centers);
                for (int row = 0; row < samples; ++row)
                    std::fill_n(centers[assigned[row]], subdim, 0);
                std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
                for (int row = 0; row < samples; ++row) {
                    for (int dim_idx = 0; dim_idx < subdim; ++dim_idx)
                        centers(assigned[row], dim_idx) += training(row, block * subdim + dim_idx);
                    ++cluster_sizes[assigned[row]];
                }
                for (int center = 0; center < codewords; ++center)
                    if (cluster_sizes[center])
                        for (int dim_idx = 0; dim_idx < subdim; ++dim_idx)
                            centers(center, dim_idx) /= cluster_sizes[center];
            }
            for (int row = 0; row < samples; ++row)
                for (int dim_idx = 0; dim_idx < subdim; ++dim_idx)
                    encoded(block * subdim + dim_idx, row) = centers(assigned[row], dim_idx);
        }
        if (pass == 1)
            break;
        cv::Mat1f cross, step;
        cv::gemm(rotated, encoded, 1, cv::noArray(), 0, cross, cv::GEMM_2_T);
        cv::SVD svd(cross, cv::SVD::FULL_UV);
        cv::gemm(svd.u, svd.vt, 1, cv::noArray(), 0, step);
        cv::gemm(step, initial_rotation, 1, cv::noArray(), 0, rotation, cv::GEMM_1_T);
        cv::gemm(rotation, raw, 1, cv::noArray(), 0, rotated);
        cv::transpose(rotated, training);
    }

    auto merge_level = [&](int level) {
        const int half = subdimensions[level - 1], dim = subdimensions[level];
        cv::Mat1f candidates(codewords * codewords, dim);
        for (int block = 0; block < lengths[level]; ++block) {
            for (int left = 0; left < codewords; ++left)
                for (int right = 0; right < codewords; ++right) {
                    std::copy_n(books[level - 1][2 * block][left], half,
                                candidates[left * codewords + right]);
                    std::copy_n(books[level - 1][2 * block + 1][right], half,
                                candidates[left * codewords + right] + half);
                }
            auto *merge = merges[level].data() + block * 2 * codewords;
            merge_quantizers(candidates, books[level][block], merge,
                             training.colRange(block * dim, (block + 1) * dim));
            for (int center = 0; center < codewords; ++center)
                candidates.row(merge[2 * center] * codewords + merge[2 * center + 1])
                    .copyTo(books[level][block].row(center));
        }
    };

    // Pair correlated subspaces. A row permutation avoids a dense permutation matrix.
    for (int level = 1; level < levels; ++level) {
        const int count = lengths[level - 1], dim = subdimensions[level - 1];
        cv::Mat1i assignment(count, samples), sizes(count, codewords, 0), joint(codewords, codewords);
        std::vector<unsigned char> paired(count);
        std::vector<RankedPoint> ranked(count * count, {-1, 0});
        for (int block = 0; block < count; ++block)
            for (int row = 0; row < samples; ++row) {
                const int id = nearest(training[row] + block * dim, books[level - 1][block]);
                assignment(block, row) = id;
                ++sizes(block, id);
            }
        for (int left = 0; left < count; ++left)
            for (int right = left + 1; right < count; ++right) {
                joint.setTo(0);
                for (int row = 0; row < samples; ++row)
                    ++joint(assignment(left, row), assignment(right, row));
                float score = 0;
                for (int first = 0; first < codewords; ++first) {
                    if (!sizes(left, first))
                        continue;
                    for (int second = 0; second < codewords; ++second)
                        if (joint(first, second) && sizes(right, second))
                            score += joint(first, second) *
                                     std::log(double(samples) * joint(first, second) / sizes(left, first) /
                                              sizes(right, second)) /
                                     samples;
                }
                ranked[left * count + right] = {left * count + right, score};
            }
        std::sort(ranked.begin(), ranked.end());
        cv::Mat1f reordered_rotation(width, width);
        std::vector<cv::Mat1f> reordered;
        for (auto point = ranked.rbegin(); point != ranked.rend(); ++point) {
            if (point->id < 0)
                continue;
            const int left = point->id / count, right = point->id % count;
            if (paired[left] || paired[right])
                continue;
            paired[left] = paired[right] = 1;
            for (int child : {left, right}) {
                const int next = int(reordered.size());
                rotation.rowRange(child * dim, (child + 1) * dim)
                    .copyTo(reordered_rotation.rowRange(next * dim, (next + 1) * dim));
                reordered.push_back(books[level - 1][child]);
            }
        }
        rotation = std::move(reordered_rotation);
        cv::gemm(rotation, raw, 1, cv::noArray(), 0, rotated);
        cv::transpose(rotated, training);
        if (level == levels - 1)
            break;
        books[level - 1] = std::move(reordered);
        merge_level(level);
    }
    train_bottom();
    for (int level = 1; level < levels; ++level)
        merge_level(level);
    for (int block = 0; block < start_books; ++block)
        build_start_merges(books.back(), block, start_merges.data() + block * start_centroids * merged_books,
                           training.colRange(block * width / start_books, (block + 1) * width / start_books));
    auto data_rows = row_pointers(data);
    graph->rotation_(n, width, data_rows.data(), nullptr, rotation[0]);

    // The density order is identical at every layer.
    std::vector<RankedPoint> ranked(n);
    for (int id = 0; id < n; ++id)
        ranked[id] = {id, density[id]};
    std::sort(ranked.begin(), ranked.end());
    cv::Mat1b selected(levels, n, uchar(0)), eligible(levels, n, uchar(0));
    eligible.row(levels - 1).setTo(1);
    const size_t buckets = size_t(levels) * codewords * codewords;
    std::vector<std::vector<Representative>> representatives(buckets);
    std::vector<std::mutex> locks(buckets);
    auto bucket = [](int level, const unsigned char *code) {
        return level * codewords * codewords + code[0] * codewords + code[1];
    };
    int cut = n;
    for (int level = levels - 1; level >= 0; --level) {
        std::vector<float> errors(n);
#pragma omp parallel for
        for (int id = 0; id < n; ++id)
            for (int block = 0; block < lengths[level]; ++block) {
                const auto *point = data[id] + block * subdimensions[level];
                const int center = nearest(point, books[level][block]);
                codes[level](id, block) = center;
                errors[id] += squared_l2(point, books[level][block][center], subdimensions[level]);
            }
        if (level < levels - 1) {
            int kept = 0;
            for (auto point = ranked.rbegin(); point != ranked.rend(); ++point)
                if (eligible(level + 1, point->id)) {
                    eligible(level, point->id) = 1;
                    if (++kept >= cut)
                        break;
                }
        }
        cut = config.delta * cut;
#pragma omp parallel for
        for (int id = 0; id < n; ++id) {
            if (!eligible(level, id))
                continue;
            const auto *code = codes[level][id];
            const int slot = bucket(level, code);
            std::lock_guard<std::mutex> lock(locks[slot]);
            auto &group = representatives[slot];
            auto found = std::find_if(group.begin(), group.end(), [&](const Representative &other) {
                return std::equal(code + 2, code + lengths[level], other.code + 2);
            });
            if (found == group.end()) {
                group.push_back({code, id, errors[id]});
                selected(level, id) = 1;
            } else if (errors[id] < found->error) {
                selected(level, found->id) = 0;
                *found = {code, id, errors[id]};
                selected(level, id) = 1;
            }
        }
        counts[level] = cv::countNonZero(selected.row(level));
        if (counts[level] < entry_points)
            throw std::runtime_error(
                "HVS needs at least ten representatives per layer; increase delta or reduce T");
    }

    cv::Mat1i targets(levels, n, 0);
    for (int level = levels - 1; level > 0; --level)
        for (int id = 0; id < n; ++id) {
            if (!selected(level, id))
                continue;
            targets(level, id) = id;
            if (!eligible(level - 1, id))
                continue;
            const auto *code = codes[level - 1][id];
            for (const auto &candidate : representatives[bucket(level - 1, code)])
                if (std::equal(code + 2, code + lengths[level - 1], candidate.code + 2)) {
                    targets(level, id) = candidate.id;
                    break;
                }
        }
    cv::Mat1b next_is_quantized(levels, n, uchar(0)), compact_flags(levels, n, uchar(0));
    std::vector<int> compact_ids(n);
    for (int level = 0; level < levels; ++level) {
        int compact = 0;
        for (int id = 0; id < n; ++id)
            if (selected(level, id)) {
                compact_ids[id] = compact;
                if (level == 0)
                    targets(0, compact) = id;
                ++compact;
            }
        if (level + 1 < levels)
            for (int id = 0; id < n; ++id)
                if (selected(level + 1, id) && eligible(level, id)) {
                    targets(level + 1, id) = compact_ids[targets(level + 1, id)];
                    next_is_quantized(level + 1, id) = 1;
                }
        if (level > 0) {
            compact = 0;
            for (int id = 0; id < n; ++id)
                if (selected(level, id)) {
                    compact_flags(level, compact) = next_is_quantized(level, id);
                    targets(level, compact++) = targets(level, id);
                }
        }
    }

    // Only one layer's pairwise distance tables need to be resident at a time.
    std::vector<cv::Mat1f> distances;
    std::vector<std::vector<float *>> distance_rows;
    std::vector<float **> distance_books;
    for (int level = 0; level < levels; ++level) {
        distances.resize(lengths[level]);
        distance_rows.resize(lengths[level]);
        distance_books.resize(lengths[level]);
        for (int block = 0; block < lengths[level]; ++block) {
            distances[block].create(codewords, codewords);
            for (int left = 0; left < codewords; ++left)
                for (int right = 0; right < codewords; ++right)
                    distances[block](left, right) = squared_l2(
                        books[level][block][left], books[level][block][right], subdimensions[level]);
            distance_rows[block] = row_pointers(distances[block]);
            distance_books[block] = distance_rows[block].data();
        }
        std::vector<int> ids;
        for (int id = 0; id < n; ++id)
            if (selected(level, id))
                ids.push_back(id);
        graph->addPoint(codes[level][ids[0]], 0, distance_books.data(), level, true);
#pragma omp parallel for
        for (int id = 1; id < int(ids.size()); ++id)
            graph->addPoint(codes[level][ids[id]], id, distance_books.data(), level, false);
        if (level + 1 < levels)
            graph->finishLayer(level);
    }
    graph->permutation(label_rows.data(), n, levels, counts.data(), 0);
    for (int level = 0; level < levels; ++level) {
        flags[level].resize(counts[level]);
        transitions[level].resize(counts[level]);
        for (int id = 0; id < counts[level]; ++id) {
            const int internal = labels(level + 1, id);
            const bool quantized = compact_flags(level, id);
            flags[level][internal] = quantized;
            transitions[level][internal] = labels(quantized ? level : 0, targets(level, id));
        }
    }

    // connect() only reads the completed graph; each cell owns its route and output.
    graph->setEf(200);
#pragma omp parallel for schedule(static)
    for (size_t cell = 0; cell < connections.size() / entry_points; ++cell) {
        std::array<unsigned char, start_books * merged_books> route;
        size_t remainder = cell;
        for (int block = start_books - 1; block >= 0; --block) {
            const size_t centroid = remainder % start_centroids;
            remainder /= start_centroids;
            std::copy_n(start_merges.data() + (block * start_centroids + centroid) * merged_books,
                        merged_books, route.data() + block * merged_books);
        }
        graph->connect(route.data(), distance_books.data(), connections.data() + cell * entry_points,
                       levels - 1, false);
    }
    graph->finishLayer(levels - 1);

    packed_codebooks.resize(size_t(width) * (width + codewords));
    for (int block = 0; block < blocks; ++block) {
        float *out = packed_codebooks.data() + size_t(block) * subdim * (width + codewords);
        std::copy_n(rotation[block * subdim], size_t(subdim) * width, out);
        std::copy_n(books[0][block][0], codewords * subdim, out + size_t(subdim) * width);
    }
}

struct Index::QueryScratch::Impl {
    const Index::Impl *owner;
    const size_t capacity;
    std::vector<float> query, rotated;
    std::vector<cv::Mat1f> books;
    std::vector<std::vector<float *>> rows;
    std::array<float, start_books * start_centroids> start{};
    std::vector<unsigned> entries;
    std::vector<std::vector<elem>> intermediate;
    std::vector<Graph::Neighbor> candidates;
    hnswlib::VisitedListPool visited;

    Impl(const Index::Impl &index, size_t queue)
        : owner(&index), capacity(queue), query(index.width, 0), rotated(index.subdimensions[0]),
          books(index.levels), rows(index.levels), entries(queue), intermediate(index.levels),
          candidates(queue + 1), visited(1, index.n) {
        for (int level = 0; level < index.levels; ++level) {
            books[level].create(index.lengths[level], codewords);
            rows[level] = row_pointers(books[level]);
            intermediate[level].resize(queue);
        }
    }
};

Index::Index(const float *base, size_t size, size_t dim, const BuildConfig &config)
    : n(size), dimension(dim) {
    if (!base || size < 257 || size > INT_MAX || dim == 0 || dim > INT_MAX - 128 || config.T < 1 ||
        config.T > 4 || !(config.delta > 0 && config.delta <= 1) || config.M < 2 || config.M >= int(size) ||
        config.ef_construction < config.M || config.training_samples < 257)
        throw std::invalid_argument("Invalid HVS dimensions/count or build parameters");
    impl_ = std::make_unique<Impl>(int(size), int(dim), config.T);
    impl_->build(base, config);
}
Index::~Index() = default;
size_t Index::max_search_queue() const {
    return *std::min_element(impl_->counts.begin(), impl_->counts.end());
}
Index::QueryScratch::QueryScratch(const Index &index, size_t max_queue) {
    if (max_queue == 0 || max_queue > index.max_search_queue())
        throw std::invalid_argument("HVS query scratch capacity exceeds a representative layer");
    impl_ = std::make_unique<Impl>(*index.impl_, max_queue);
}
Index::QueryScratch::~QueryScratch() = default;

void Index::search(const float *query, size_t topk, size_t queue, unsigned *output,
                   QueryScratch &work) const {
    auto &scratch = *work.impl_;
    if (!query || !output || scratch.owner != impl_.get() || topk == 0 || topk > queue ||
        queue > scratch.capacity)
        throw std::invalid_argument("Invalid HVS query, topk, queue or scratch owner");
    const auto &index = *impl_;
    auto &graph = *index.graph;
    const int subdim = index.subdimensions[0];
    std::copy_n(query, dimension, scratch.query.data());
    for (int block = 0; block < index.lengths[0]; ++block)
        graph.computeDistanceTable(scratch.query.data(),
                                   index.packed_codebooks.data() +
                                       size_t(block) * subdim * (index.width + codewords),
                                   scratch.rows[0][block], subdim, index.width, scratch.rotated.data());
    for (int level = 1; level < index.levels; ++level)
        for (int block = 0; block < index.lengths[level]; ++block)
            for (int center = 0; center < codewords; ++center) {
                const auto *merge = index.merges[level].data() + block * 2 * codewords;
                scratch.books[level](block, center) =
                    scratch.books[level - 1](2 * block, merge[2 * center]) +
                    scratch.books[level - 1](2 * block + 1, merge[2 * center + 1]);
            }
    scratch.start.fill(0);
    unsigned cell = 0;
    for (int block = 0; block < start_books; ++block) {
        for (int center = 0; center < start_centroids; ++center)
            for (int part = 0; part < merged_books; ++part)
                scratch.start[block * start_centroids + center] += scratch.books.back()(
                    block * merged_books + part,
                    index.start_merges[(block * start_centroids + center) * merged_books + part]);
        const auto begin = scratch.start.begin() + block * start_centroids;
        cell = cell * start_centroids + unsigned(std::min_element(begin, begin + start_centroids) - begin);
    }
    const auto *points = index.connections.data() + cell * entry_points;
    if (index.levels == 1) {
        graph.SearchWithsingleGraph(scratch.rows[0].data(), index.lengths[0], points, scratch.entries.data(),
                                    index.transitions[0].data(), queue, &scratch.visited,
                                    &scratch.candidates);
    } else {
        graph.SearchWithquanGraph(scratch.rows.back().data(), index.lengths.back(), points,
                                  scratch.intermediate[index.levels - 2].data(),
                                  index.transitions.back().data(), queue, index.flags.back().data(),
                                  index.levels - 1, &scratch.visited, &scratch.candidates);
        for (int level = index.levels - 2; level >= 1; --level)
            graph.SearchWithquanGraph3(
                scratch.rows[level].data(), index.lengths[level], scratch.intermediate[level].data(),
                scratch.intermediate[level - 1].data(), index.transitions[level].data(), queue,
                index.flags[level].data(), level, &scratch.visited, &scratch.candidates);
        graph.SearchWithquanGraph2(scratch.rows[0].data(), index.lengths[0], scratch.intermediate[0].data(),
                                   scratch.entries.data(), index.transitions[0].data(), queue,
                                   &scratch.visited, &scratch.candidates);
    }
    graph.SearchWithOptGraph(query, topk, queue, output, scratch.entries.data(), &scratch.visited,
                             &scratch.candidates);
}

} // namespace hvs
