#include "prefix_cache.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <chrono>

namespace barskuy::engine {

PrefixCache::PrefixCache(const Config& config, PagedKVCacheManager* kv_cache)
    : config_(config), kv_cache_(kv_cache) {
    if (!kv_cache_) {
        core::Logger::warn("PrefixCache: No KV cache provided, prefix caching disabled");
    }
}

PrefixCache::~PrefixCache() {
    invalidate();
}

size_t PrefixCache::compute_hash(const std::vector<int>& tokens, size_t len) const {
    uint64_t hash = 0;
    size_t n = std::min(len, tokens.size());
    for (size_t i = 0; i < n; ++i) {
        hash = (hash * BASE + static_cast<uint64_t>(tokens[i] + 1)) % MOD;
    }
    return static_cast<size_t>(hash);
}

size_t PrefixCache::compute_rolling_hash(const std::vector<int>& tokens, size_t start, size_t len) const {
    return compute_hash(tokens, start + len);
}

int PrefixCache::find_lcp(const std::vector<int>& a, const std::vector<int>& b) const {
    size_t n = std::min(a.size(), b.size());
    int lcp = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] == b[i]) {
            lcp++;
        } else {
            break;
        }
    }
    return lcp;
}

int PrefixCache::find_prefix(const std::vector<int>& tokens,
                             std::vector<int>* matched_blocks,
                             MatchCallback callback) {
    if (!kv_cache_ || tokens.size() < config_.min_prefix_tokens) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    int best_match = 0;
    std::shared_ptr<PrefixEntry> best_entry;
    
    // Check all cached prefixes
    for (auto& [hash, entries] : cache_) {
        for (auto& entry : entries) {
            // Check TTL (based on creation time, not last access)
            if (now - entry->created_at > config_.ttl_seconds) {
                continue;
            }
            
            // Find longest common prefix
            int lcp = find_lcp(tokens, entry->token_ids);
            
            // Check if match is significant enough
            if (lcp >= static_cast<int>(config_.min_prefix_tokens) && 
                lcp > best_match &&
                static_cast<double>(lcp) / entry->token_ids.size() >= config_.similarity_threshold) {
                best_match = lcp;
                best_entry = entry;
            }
        }
    }
    
    if (best_entry && best_match > 0) {
        // Found a matching prefix
        best_entry->hit_count++;
        
        int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        best_entry->last_access = now;
        
        // Determine which blocks to reuse
        int blocks_needed = (best_match + kv_cache_->get_config().block_size - 1) / kv_cache_->get_config().block_size;
        blocks_needed = std::min(blocks_needed, static_cast<int>(best_entry->block_ids.size()));
        
        if (matched_blocks) {
            matched_blocks->assign(best_entry->block_ids.begin(), 
                                  best_entry->block_ids.begin() + blocks_needed);
        }
        
        if (callback) {
            callback(*best_entry, best_match);
        }
        
        core::Logger::debug("Prefix cache HIT: matched {} tokens, reusing {} blocks", 
            best_match, blocks_needed);
        return best_match;
    }
    
    core::Logger::debug("Prefix cache MISS: no matching prefix for {} tokens", tokens.size());
    return 0;
}

void PrefixCache::store_prefix(const std::vector<int>& tokens, const std::vector<int>& block_ids) {
    if (!kv_cache_ || tokens.size() < config_.min_prefix_tokens || block_ids.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    
    // Count total entries
    auto count_total_entries = [this]() {
        size_t total = 0;
        for (const auto& [hash, entries] : cache_) {
            total += entries.size();
        }
        return total;
    };
    
    // Limit cache size
    size_t current_count = count_total_entries();
    if (current_count >= config_.max_entries) {
        // Remove least recently used
        int64_t oldest = INT64_MAX;
        size_t oldest_hash = 0;
        std::shared_ptr<PrefixEntry> oldest_entry;
        
        for (auto& [hash, entries] : cache_) {
            for (auto& entry : entries) {
                if (entry->last_access < oldest) {
                    oldest = entry->last_access;
                    oldest_hash = hash;
                    oldest_entry = entry;
                }
            }
        }
        
        if (oldest_entry) {
            auto& vec = cache_[oldest_hash];
            vec.erase(std::remove(vec.begin(), vec.end(), oldest_entry), vec.end());
            if (vec.empty()) cache_.erase(oldest_hash);
        }
    }
    
    // Create new entry
    auto entry = std::make_shared<PrefixEntry>();
    entry->token_ids = tokens;
    entry->block_ids = block_ids;
    entry->token_count = tokens.size();
    entry->hash = compute_hash(tokens, tokens.size());
    
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    entry->created_at = now;
    entry->last_access = now;
    entry->hit_count = 0;
    
    cache_[entry->hash].push_back(entry);
    core::Logger::debug("Prefix cache STORED: {} tokens, {} blocks", tokens.size(), block_ids.size());
}

void PrefixCache::invalidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    core::Logger::info("Prefix cache invalidated");
}

PrefixCache::Stats PrefixCache::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Stats stats;
    stats.total_entries = 0;
    stats.total_tokens_cached = 0;
    stats.total_hits = 0;
    stats.total_misses = 0;
    
    for (const auto& [hash, entries] : cache_) {
        for (const auto& entry : entries) {
            stats.total_entries++;
            stats.total_tokens_cached += entry->token_count;
            stats.total_hits += entry->hit_count;
        }
    }
    
    // Approximate misses (simplified)
    stats.total_misses = stats.total_entries * 2;  // Rough estimate
    if (stats.total_hits + stats.total_misses > 0) {
        stats.hit_rate = static_cast<double>(stats.total_hits) / 
                        (stats.total_hits + stats.total_misses);
    }
    
    return stats;
}

void PrefixCache::cleanup_expired() {
    if (!kv_cache_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    int64_t ttl_seconds = config_.ttl_seconds;
    
    for (auto it = cache_.begin(); it != cache_.end(); ) {
        auto& entries = it->second;
        entries.erase(std::remove_if(entries.begin(), entries.end(),
            [now, ttl_seconds](const std::shared_ptr<PrefixEntry>& e) {
                return (now - e->last_access) > ttl_seconds;
            }), entries.end());
        
        if (entries.empty()) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace barskuy::engine