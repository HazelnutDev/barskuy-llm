#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <memory>
#include <optional>
#include <ggml.h>

namespace barskuy::engine {

// Paged KV Cache - manages KV cache in fixed-size blocks
// Enables continuous batching by allowing dynamic allocation/reuse of KV blocks

struct KVBlock {
    int block_id = -1;
    struct ggml_tensor* k_cache = nullptr;  // [n_heads, block_size, head_dim]
    struct ggml_tensor* v_cache = nullptr;  // [n_heads, block_size, head_dim]
    int ref_count = 0;
    bool in_use = false;
};

struct KVBlockAllocation {
    std::vector<int> block_ids;  // Block IDs allocated to this sequence
    int num_tokens = 0;          // Total tokens in this allocation
    int64_t last_access = 0;     // For LRU eviction
};

class PagedKVCacheManager {
public:
    struct Config {
        int n_layers = 32;
        int n_heads = 32;
        int n_kv_heads = 32;       // For GQA
        int head_dim = 128;
        int block_size = 16;       // Tokens per block (typically 16 or 32)
        int max_blocks = 8192;     // Maximum number of blocks (VRAM dependent)
        enum ggml_type dtype = GGML_TYPE_F16;
        struct ggml_context* ctx = nullptr;
    };

    explicit PagedKVCacheManager(const Config& config);
    ~PagedKVCacheManager();

    // Initialize the block pool
    bool initialize();

    // Allocate blocks for a new sequence (prefill)
    // Returns block IDs allocated, or empty if OOM
    std::vector<int> allocate_blocks(int num_tokens);

    // Append tokens to existing sequence (decode step)
    // Returns new block IDs if expansion needed, or empty
    std::vector<int> append_blocks(int sequence_id, int num_new_tokens);

    // Free all blocks for a sequence (sequence complete)
    void free_sequence(int sequence_id);

    // Get block pointers for a sequence (for attention computation)
    std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> get_sequence_blocks(int sequence_id) const;

    // Get block pointers for specific layer
    std::pair<struct ggml_tensor*, struct ggml_tensor*> get_layer_blocks(int sequence_id, int layer) const;

    // Get number of tokens currently stored for a sequence
    int get_sequence_length(int sequence_id) const;

    // Update last access time for LRU
    void touch_sequence(int sequence_id);

    // Get statistics
    int get_free_blocks() const { return free_blocks_.size(); }
    int get_used_blocks() const { return static_cast<int>(allocations_.size()); }
    int get_total_blocks() const { return config_.max_blocks; }
    float get_usage_ratio() const { return static_cast<float>(get_used_blocks()) / config_.max_blocks; }

    // Evict LRU sequences if needed (for memory pressure)
    std::vector<int> evict_lru(int num_blocks_needed);

    // Check if we can allocate N blocks
    bool can_allocate(int num_blocks) const { return free_blocks_.size() >= static_cast<size_t>(num_blocks); }

    // Get statistics info
    struct BlockInfo {
        int total_blocks = 0;
        int free_blocks = 0;
        int used_blocks = 0;
        int active_sequences = 0;
    };
    BlockInfo get_info() const;

    // Get config (for prefix cache)
    const Config& get_config() const { return config_; }

private:
    Config config_;
    std::vector<KVBlock> blocks_;
    std::queue<int> free_blocks_;
    std::unordered_map<int, KVBlockAllocation> allocations_;
    int next_block_id_ = 0;
    int64_t access_counter_ = 0;
    mutable std::mutex mutex_;

    // Create KV tensors for a block
    bool create_block_tensors(KVBlock& block);

    // Calculate blocks needed for token count
    int calc_blocks_needed(int num_tokens) const { return (num_tokens + config_.block_size - 1) / config_.block_size; }

    // Internal free without locking (for use within locked methods)
    void free_sequence_unlocked(int sequence_id);
};

} // namespace barskuy::engine