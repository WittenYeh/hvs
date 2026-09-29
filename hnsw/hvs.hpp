#pragma once
#include <cstddef>
#include <memory>

namespace hvs {

struct BuildConfig {
    // Paper's finest quantization level, in [4, 7]. There are T - 3
    // quantized graph layers, with 2^T sub-codebooks at the finest level.
    int T = 4;
    float delta = 0.5f;
    int M = 16;
    int ef_construction = 500;
    int training_samples = 10000;
    int seed = 100;
};

// Builds and owns the original-vector graph, quantized layers and routing tables.
// Input vectors need only remain valid during construction. No files are accessed.
class Index {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    const std::size_t n, dimension;
    Index(const float *base, std::size_t size, std::size_t dim, const BuildConfig &config = {});
    ~Index();
    std::size_t max_search_queue() const;

    // Allocate once per worker; reuse for all queries of this index. Different
    // workers can search the same index concurrently with distinct scratch objects.
    class QueryScratch {
        friend class Index;
        struct Impl;
        std::unique_ptr<Impl> impl_;

      public:
        QueryScratch(const Index &index, std::size_t max_queue);
        ~QueryScratch();
    };

    // Require 0 < topk <= queue <= min(max_search_queue(), scratch capacity).
    void search(const float *query, std::size_t topk, std::size_t queue, unsigned *output,
                QueryScratch &scratch) const;
};

} // namespace hvs
