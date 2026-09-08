#include "continuous_batching.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <random>

namespace barskuy::engine {

ContinuousBatchingScheduler::ContinuousBatchingScheduler(const Config& config) : config_(config) {}

ContinuousBatchingScheduler::~ContinuousBatchingScheduler() {
    stop();
}

bool ContinuousBatchingScheduler::initialize(PagedKVCacheManager* kv_cache, TextEngine* text_engine) {
    kv_cache_ = kv_cache;
    text_engine_ = text_engine;
    
    if (!kv_cache_ || !text_engine_) {
        core::Logger::error("ContinuousBatchingScheduler: kv_cache and text_engine required");
        return false;
    }
    
    running_ = true;
    start_time_ = std::chrono::steady_clock::now();
    scheduler_thread_ = std::thread(&ContinuousBatchingScheduler::run_scheduler, this);
    
    core::Logger::info("ContinuousBatchingScheduler started (max_batch={}, max_tokens={})", 
        config_.max_batch_size, config_.max_tokens_per_batch);
    return true;
}

std::string ContinuousBatchingScheduler::submit_request(BatchRequest&& request) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (pending_queue_.size() >= static_cast<size_t>(config_.max_queue_size)) {
        core::Logger::warn("ContinuousBatchingScheduler: Queue full, rejecting request");
        return "";
    }
    
    std::string id = generate_request_id();
    request.id = id;
    request.enqueued_at = std::chrono::steady_clock::now();
    
    pending_queue_.push(std::move(request));
    queue_cv_.notify_one();
    
    return id;
}

std::string ContinuousBatchingScheduler::submit_stream_request(BatchRequest&& request) {
    request.stream = true;
    return submit_request(std::move(request));
}

void ContinuousBatchingScheduler::stop() {
    if (!running_.exchange(false)) return;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }
    
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
    
    core::Logger::info("ContinuousBatchingScheduler stopped");
}

ContinuousBatchingScheduler::Stats ContinuousBatchingScheduler::get_stats() const {
    Stats stats;
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats.pending_requests = static_cast<int>(pending_queue_.size());
    stats.active_sequences = static_cast<int>(active_requests_.size());
    stats.completed_requests = completed_count_.load();
    stats.failed_requests = failed_count_.load();
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time_).count();
    if (elapsed > 0) {
        stats.avg_latency_ms = static_cast<float>(elapsed) / std::max(1, stats.completed_requests);
        stats.throughput_tokens_per_sec = static_cast<float>(total_tokens_.load()) * 1000.0f / elapsed;
    }
    
    return stats;
}

void ContinuousBatchingScheduler::run_scheduler() {
    core::Logger::debug("Scheduler thread started");
    
    while (running_) {
        std::vector<BatchRequest*> batch;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for work or timeout
            queue_cv_.wait_for(lock, config_.batch_timeout, [this] {
                return !running_ || !pending_queue_.empty() || !active_requests_.empty();
            });
            
            if (!running_) break;
            
            batch = form_batch();
        }
        
        if (!batch.empty()) {
            process_batch(batch);
        }
        
        // Clean up old completed requests (keep last 1000)
        if (completed_requests_.size() > 1000) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            auto it = completed_requests_.begin();
            std::advance(it, completed_requests_.size() - 1000);
            completed_requests_.erase(completed_requests_.begin(), it);
        }
    }
    
    core::Logger::debug("Scheduler thread exited");
}

std::vector<BatchRequest*> ContinuousBatchingScheduler::form_batch() {
    std::vector<BatchRequest*> batch;
    batch.reserve(config_.max_batch_size);
    
    int total_tokens = 0;
    
    // First, add pending requests (prefill phase)
    while (!pending_queue_.empty() && 
           static_cast<int>(batch.size()) < config_.max_batch_size &&
           total_tokens < config_.max_tokens_per_batch) {
        
        BatchRequest* req = &pending_queue_.front();
        
        // Estimate token count (rough: ~4 chars per token)
        int est_tokens = 0;
        for (auto& [role, content] : req->messages) {
            est_tokens += static_cast<int>(content.size()) / 4;
        }
        est_tokens += req->max_tokens;
        
        if (total_tokens + est_tokens > config_.max_tokens_per_batch && !batch.empty()) {
            break;  // Would exceed batch token budget
        }
        
        // Allocate KV cache blocks for this sequence
        auto seq_ids = kv_cache_->allocate_blocks(est_tokens);
        if (seq_ids.empty()) {
            core::Logger::warn("KV cache OOM, waiting for space");
            break;  // Wait for KV cache space
        }
        
        req->sequence_id = seq_ids[0];
        req->started_at = std::chrono::steady_clock::now();
        
        // Move to active
        batch.push_back(req);
        active_requests_[req->id] = std::move(*req);
        pending_queue_.pop();
        
        total_tokens += est_tokens;
    }
    
    // Then, add active requests that need decode
    for (auto it = active_requests_.begin(); it != active_requests_.end(); ++it) {
        const std::string& id = it->first;
        BatchRequest& req = it->second;
        if (req.finished) continue;
        if (static_cast<int>(batch.size()) >= config_.max_batch_size) break;
        
        // Only add if not already in batch (from pending phase)
        if (std::find(batch.begin(), batch.end(), &req) == batch.end()) {
            batch.push_back(&req);
        }
    }
    
    return batch;
}

void ContinuousBatchingScheduler::process_batch(std::vector<BatchRequest*>& batch) {
    if (batch.empty()) return;
    
    // Separate into prefill (new) and decode (active)
    std::vector<BatchRequest*> prefill_batch;
    std::vector<BatchRequest*> decode_batch;
    
    for (auto* req : batch) {
        if (!req->prefill_done) {
            prefill_batch.push_back(req);
        } else {
            decode_batch.push_back(req);
        }
    }
    
    // Run prefill for new requests
    if (!prefill_batch.empty()) {
        run_prefill(prefill_batch);
    }
    
    // Run decode for all active requests
    if (!decode_batch.empty()) {
        run_decode(decode_batch);
    }
    
    // Check for finished requests
    for (auto* req : batch) {
        if (is_finished(*req)) {
            finish_request(*req, req->finish_reason);
        }
    }
}

void ContinuousBatchingScheduler::run_prefill(std::vector<BatchRequest*>& new_requests) {
    // For now, process sequentially
    // In a full implementation, this would batch multiple prefill requests
    for (auto* req : new_requests) {
        if (req->finished) continue;
        
        try {
            // Tokenize prompt
            // Note: This is simplified - real implementation would batch tokenize
            TextEngine::CompletionResult result = text_engine_->complete(
                req->model_id,
                req->messages,
                1,  // Just prefill - we'll do decode step by step
                req->temperature,
                req->top_p,
                req->stop
            );
            
            // For now, we treat this as full generation
            // Real continuous batching would only run prefill here
            req->prompt_tokens = result.prompt_tokens;
            req->accumulated_text = result.text;
            req->finish_reason = result.finish_reason;
            req->finished = true;
            req->prefill_done = true;
            
            // Call stream callback if streaming
            if (req->stream && req->callback && !result.text.empty()) {
                req->callback(result.text, true);
            }
            
        } catch (const std::exception& e) {
            core::Logger::error("Prefill failed for request {}: {}", req->id, e.what());
            req->finish_reason = "error";
            req->finished = true;
        }
    }
}

void ContinuousBatchingScheduler::run_decode(std::vector<BatchRequest*>& active) {
    // In a full implementation, this would:
    // 1. Batch all active sequences
    // 2. Run single forward pass
    // 3. Sample next token for each
    // 4. Update KV cache
    // 5. Check stop conditions
    
    // For now, mark as prefill_done to prevent re-prefill
    for (auto* req : active) {
        req->prefill_done = true;
    }
}

bool ContinuousBatchingScheduler::is_finished(BatchRequest& req) {
    if (req.finished) return true;
    if (req.generated_tokens >= req.max_tokens) {
        req.finish_reason = "length";
        return true;
    }
    return false;
}

void ContinuousBatchingScheduler::finish_request(BatchRequest& req, const std::string& finish_reason) {
    if (req.finished) return;
    
    req.finished = true;
    req.finish_reason = finish_reason;
    
    // Free KV cache blocks
    if (req.sequence_id >= 0) {
        kv_cache_->free_sequence(req.sequence_id);
    }
    
    // Call final callback for streaming
    if (req.stream && req.callback) {
        req.callback("", true);
    }
    
    // Move to completed
    completed_count_++;
    total_tokens_ += req.generated_tokens + req.prompt_tokens;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        active_requests_.erase(req.id);
        completed_requests_[req.id] = req;
    }
}

std::string ContinuousBatchingScheduler::generate_request_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 35);
    
    const char* chars = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::string id = "req_";
    for (int i = 0; i < 16; ++i) {
        id += chars[dis(gen)];
    }
    return id;
}

} // namespace barskuy::engine