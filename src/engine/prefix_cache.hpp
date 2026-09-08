#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <functional>
#include <chrono>
#include "paged_kv_cache.hpp"

namespace barskuy::engine {

// Prefix Cache - enables KV cache reuse for common prompt prefixes
// Uses rolling hash for efficient prefix matching

class PrefixCache {
public:
    struct Config {
        size_t max_entries = 1024;           // Max cached prefixes
        size_t max_prefix_tokens = 2048;     // Max tokens in a cached prefix
        size_t min_prefix_tokens = 4;        // Min tokens to consider caching
        double similarity_threshold = 0.95;  // Min similarity for reuse
        int64_t ttl_seconds = 3600;          // Time-to-live for cache entries (seconds)
    };

    struct PrefixEntry {
        std::vector<int> token_ids;          // Token sequence
        std::vector<int> block_ids;          // KV cache block IDs
        size_t token_count = 0;
        int64_t created_at = 0;
        int64_t last_access = 0;
        int hit_count = 0;
        size_t hash = 0;
    };

    using MatchCallback = std::function<void(const PrefixEntry&, int matched_tokens)>;

    explicit PrefixCache(const Config& config, PagedKVCacheManager* kv_cache);
    ~PrefixCache();

    // Find matching prefix for token sequence
    // Returns number of matched tokens, populates matched_blocks if provided
    int find_prefix(const std::vector<int>& tokens, 
                    std::vector<int>* matched_blocks = nullptr,
                    MatchCallback callback = nullptr);

    // Store prefix in cache after prefill
    void store_prefix(const std::vector<int>& tokens, const std::vector<int>& block_ids);

    // Invalidate entries (e.g., on model reload)
    void invalidate();

    // Get cache statistics
    struct Stats {
        size_t total_entries = 0;
        size_t total_tokens_cached = 0;
        int64_t total_hits = 0;
        int64_t total_misses = 0;
        double hit_rate = 0.0;
    };
    Stats get_stats() const;

    // Cleanup expired entries
    void cleanup_expired();

private:
    Config config_;
    PagedKVCacheManager* kv_cache_;
    
    struct HashEntry {
        size_t hash;
        std::shared_ptr<PrefixEntry> entry;
    };
    
    // Hash -> entries (for collision handling)
    std::unordered_map<size_t, std::vector<std::shared_ptr<PrefixEntry>>> cache_;
    mutable std::mutex mutex_;
    
    int64_t access_counter_ = 0;
    
    // Rolling hash for prefix matching
    static constexpr uint64_t BASE = 131;
    static constexpr uint64_t MOD = 18446744073709551557ULL;  // 2^64 - 59
    
    size_t compute_hash(const std::vector<int>& tokens, size_t len) const;
    size_t compute_rolling_hash(const std::vector<int>& tokens, size_t start, size_t len) const;
    
    // Find longest common prefix
    int find_lcp(const std::vector<int>& a, const std::vector<int>& b) const;
};

// Hash specialization for vector<int>
struct VectorIntHash {
    size_t operator()(const std::vector<int>& v) const noexcept {
        size_t hash = 0;
        for (int x : v) {
            hash = hash * 1315423911u + static_cast<size_t>(x);
        }
        return hash;
    }
};

} // namespace barskuy::engine