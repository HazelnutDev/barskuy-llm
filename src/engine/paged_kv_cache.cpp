#include "paged_kv_cache.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <chrono>

namespace barskuy::engine {

PagedKVCacheManager::PagedKVCacheManager(const Config& config) : config_(config) {
    blocks_.reserve(config_.max_blocks);
}

PagedKVCacheManager::~PagedKVCacheManager() {
    // Tensors are owned by ggml context, just clear our structures
    blocks_.clear();
    while (!free_blocks_.empty()) free_blocks_.pop();
    allocations_.clear();
}

bool PagedKVCacheManager::initialize() {
    if (!config_.ctx) {
        core::Logger::error("PagedKVCacheManager: No ggml context provided");
        return false;
    }

    core::Logger::info("Initializing PagedKVCache: {} layers, {} heads, {} KV heads, {} dim, block_size={}, max_blocks={}",
        config_.n_layers, config_.n_heads, config_.n_kv_heads, config_.head_dim,
        config_.block_size, config_.max_blocks);

    // Pre-allocate all blocks
    blocks_.resize(config_.max_blocks);
    for (int i = 0; i < config_.max_blocks; ++i) {
        blocks_[i].block_id = i;
        if (!create_block_tensors(blocks_[i])) {
            core::Logger::error("Failed to create tensors for block {}", i);
            return false;
        }
        free_blocks_.push(i);
    }

    core::Logger::info("PagedKVCache initialized with {} blocks", config_.max_blocks);
    return true;
}

bool PagedKVCacheManager::create_block_tensors(KVBlock& block) {
    // K cache: [n_kv_heads, block_size, head_dim]
    int64_t k_ne[3] = {config_.head_dim, config_.block_size, config_.n_kv_heads};
    block.k_cache = ggml_new_tensor_3d(config_.ctx, config_.dtype, k_ne[0], k_ne[1], k_ne[2]);
    if (!block.k_cache) return false;
    
    char k_name[64];
    snprintf(k_name, sizeof(k_name), "kv_block_%d_k", block.block_id);
    ggml_set_name(block.k_cache, k_name);

    // V cache: [n_kv_heads, block_size, head_dim]
    int64_t v_ne[3] = {config_.head_dim, config_.block_size, config_.n_kv_heads};
    block.v_cache = ggml_new_tensor_3d(config_.ctx, config_.dtype, v_ne[0], v_ne[1], v_ne[2]);
    if (!block.v_cache) return false;
    
    char v_name[64];
    snprintf(v_name, sizeof(v_name), "kv_block_%d_v", block.block_id);
    ggml_set_name(block.v_cache, v_name);

    return true;
}

std::vector<int> PagedKVCacheManager::allocate_blocks(int num_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int blocks_needed = calc_blocks_needed(num_tokens);
    if (free_blocks_.size() < static_cast<size_t>(blocks_needed)) {
        core::Logger::warn("PagedKVCache: OOM - need {} blocks, only {} free", blocks_needed, free_blocks_.size());
        return {};
    }

    std::vector<int> allocated;
    allocated.reserve(blocks_needed);
    
    for (int i = 0; i < blocks_needed; ++i) {
        int block_id = free_blocks_.front();
        free_blocks_.pop();
        
        blocks_[block_id].in_use = true;
        blocks_[block_id].ref_count = 1;
        allocated.push_back(block_id);
    }

    // Create allocation record
    int sequence_id = next_block_id_++;
    KVBlockAllocation alloc;
    alloc.block_ids = allocated;
    alloc.num_tokens = num_tokens;
    alloc.last_access = access_counter_++;
    allocations_[sequence_id] = std::move(alloc);

    core::Logger::debug("Allocated {} blocks for sequence {} (tokens: {})", blocks_needed, sequence_id, num_tokens);
    // Return: [sequence_id, block_id1, block_id2, ...]
    std::vector<int> result;
    result.push_back(sequence_id);
    result.insert(result.end(), allocated.begin(), allocated.end());
    return result;
}

std::vector<int> PagedKVCacheManager::append_blocks(int sequence_id, int num_new_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(sequence_id);
    if (it == allocations_.end()) {
        core::Logger::warn("PagedKVCache: Sequence {} not found for append", sequence_id);
        return {};
    }

    KVBlockAllocation& alloc = it->second;
    int current_tokens = alloc.num_tokens;
    int new_total_tokens = current_tokens + num_new_tokens;
    
    int current_blocks = calc_blocks_needed(current_tokens);
    int needed_blocks = calc_blocks_needed(new_total_tokens);
    int additional_blocks = needed_blocks - current_blocks;

    if (additional_blocks <= 0) {
        alloc.num_tokens = new_total_tokens;
        alloc.last_access = access_counter_++;
        return {};  // No new blocks needed
    }

    if (free_blocks_.size() < static_cast<size_t>(additional_blocks)) {
        core::Logger::warn("PagedKVCache: OOM on append - need {} more blocks, only {} free", additional_blocks, free_blocks_.size());
        return {};
    }

    std::vector<int> new_blocks;
    new_blocks.reserve(additional_blocks);
    
    for (int i = 0; i < additional_blocks; ++i) {
        int block_id = free_blocks_.front();
        free_blocks_.pop();
        
        blocks_[block_id].in_use = true;
        blocks_[block_id].ref_count = 1;
        alloc.block_ids.push_back(block_id);
        new_blocks.push_back(block_id);
    }

    alloc.num_tokens = new_total_tokens;
    alloc.last_access = access_counter_++;

    core::Logger::debug("Appended {} blocks to sequence {} (total tokens: {})", additional_blocks, sequence_id, new_total_tokens);
    return new_blocks;
}

void PagedKVCacheManager::free_sequence(int sequence_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_sequence_unlocked(sequence_id);
}

std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> PagedKVCacheManager::get_sequence_blocks(int sequence_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(sequence_id);
    if (it == allocations_.end()) {
        return {};
    }

    const KVBlockAllocation& alloc = it->second;
    std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> result;
    result.reserve(alloc.block_ids.size() * config_.n_layers);

    for (int block_id : alloc.block_ids) {
        if (block_id >= 0 && block_id < config_.max_blocks) {
            const KVBlock& block = blocks_[block_id];
            for (int layer = 0; layer < config_.n_layers; ++layer) {
                // Note: In a full implementation, each layer would have its own block tensors
                // For now, we share blocks across layers (simplified)
                result.emplace_back(block.k_cache, block.v_cache);
            }
        }
    }
    return result;
}

std::pair<struct ggml_tensor*, struct ggml_tensor*> PagedKVCacheManager::get_layer_blocks(int sequence_id, int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(sequence_id);
    if (it == allocations_.end()) {
        return {nullptr, nullptr};
    }

    // For now, return first block's tensors (simplified - real impl would have per-layer blocks)
    const KVBlockAllocation& alloc = it->second;
    if (alloc.block_ids.empty()) return {nullptr, nullptr};

    int block_id = alloc.block_ids[0];
    if (block_id >= 0 && block_id < config_.max_blocks) {
        const KVBlock& block = blocks_[block_id];
        return {block.k_cache, block.v_cache};
    }
    return {nullptr, nullptr};
}

int PagedKVCacheManager::get_sequence_length(int sequence_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(sequence_id);
    if (it == allocations_.end()) return 0;
    return it->second.num_tokens;
}

void PagedKVCacheManager::free_sequence_unlocked(int sequence_id) {
    auto it = allocations_.find(sequence_id);
    if (it == allocations_.end()) {
        return;
    }

    KVBlockAllocation& alloc = it->second;
    
    for (int block_id : alloc.block_ids) {
        if (block_id >= 0 && block_id < config_.max_blocks) {
            blocks_[block_id].in_use = false;
            blocks_[block_id].ref_count = 0;
            free_blocks_.push(block_id);
        }
    }

    int freed = static_cast<int>(alloc.block_ids.size());
    allocations_.erase(it);
    
    core::Logger::debug("Freed {} blocks for sequence {}", freed, sequence_id);
}

void PagedKVCacheManager::touch_sequence(int sequence_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(sequence_id);
    if (it != allocations_.end()) {
        it->second.last_access = access_counter_++;
    }
}

std::vector<int> PagedKVCacheManager::evict_lru(int num_blocks_needed) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (free_blocks_.size() >= static_cast<size_t>(num_blocks_needed)) {
        return {};
    }

    // Sort sequences by last_access (oldest first)
    std::vector<std::pair<int, int64_t>> seq_access;
    seq_access.reserve(allocations_.size());
    for (const auto& [seq_id, alloc] : allocations_) {
        seq_access.emplace_back(seq_id, alloc.last_access);
    }
    std::sort(seq_access.begin(), seq_access.end(), 
        [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<int> evicted;
    int blocks_freed = 0;
    
    for (const auto& [seq_id, _] : seq_access) {
        if (blocks_freed >= num_blocks_needed) break;
        
        auto it = allocations_.find(seq_id);
        if (it != allocations_.end()) {
            int block_count = static_cast<int>(it->second.block_ids.size());
            free_sequence_unlocked(seq_id);
            evicted.push_back(seq_id);
            blocks_freed += block_count;
        }
    }

    core::Logger::info("Evicted {} LRU sequences, freed ~{} blocks", evicted.size(), blocks_freed);
    return evicted;
}

PagedKVCacheManager::BlockInfo PagedKVCacheManager::get_info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockInfo info;
    info.total_blocks = config_.max_blocks;
    info.free_blocks = static_cast<int>(free_blocks_.size());
    
    // Calculate total blocks used across all sequences
    int used = 0;
    for (const auto& [seq_id, alloc] : allocations_) {
        used += static_cast<int>(alloc.block_ids.size());
    }
    info.used_blocks = used;
    info.active_sequences = static_cast<int>(allocations_.size());
    return info;
}

} // namespace barskuy::engine