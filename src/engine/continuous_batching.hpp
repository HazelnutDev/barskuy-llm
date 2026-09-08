#pragma once

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>
#include <optional>
#include <unordered_map>
#include "paged_kv_cache.hpp"
#include "text_engine.hpp"

namespace barskuy::engine {

// Request in the continuous batching queue
struct BatchRequest {
    std::string id;
    std::string model_id;
    std::vector<std::pair<std::string, std::string>> messages;
    int max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    std::vector<std::string> stop;
    bool stream = false;
    
    // Callback for streaming
    TextEngine::StreamCallback callback;
    
    // State
    int sequence_id = -1;  // PagedKVCache sequence ID
    int prompt_tokens = 0;
    int generated_tokens = 0;
    bool prefill_done = false;
    bool finished = false;
    std::string finish_reason;
    std::string accumulated_text;
    
    std::chrono::steady_clock::time_point enqueued_at;
    std::chrono::steady_clock::time_point started_at;
};

// Continuous Batching Scheduler
// Packs prefill and decode steps into single batches for maximum throughput
class ContinuousBatchingScheduler {
public:
    struct Config {
        int max_batch_size = 32;           // Max sequences in a batch
        int max_tokens_per_batch = 4096;   // Max total tokens in a batch
        int max_queue_size = 1024;         // Max pending requests
        std::chrono::milliseconds batch_timeout{10};  // Max wait to form batch
    };

    explicit ContinuousBatchingScheduler(const Config& config = {});
    ~ContinuousBatchingScheduler();

    // Initialize with PagedKVCache and TextEngine
    bool initialize(PagedKVCacheManager* kv_cache, TextEngine* text_engine);

    // Submit a new request (non-blocking)
    // Returns request ID or empty string if queue full
    std::string submit_request(BatchRequest&& request);

    // Submit streaming request
    std::string submit_stream_request(BatchRequest&& request);

    // Stop the scheduler
    void stop();

    // Get queue statistics
    struct Stats {
        int pending_requests = 0;
        int active_sequences = 0;
        int completed_requests = 0;
        int failed_requests = 0;
        float avg_latency_ms = 0.0f;
        float throughput_tokens_per_sec = 0.0f;
    };
    Stats get_stats() const;

private:
    Config config_;
    PagedKVCacheManager* kv_cache_ = nullptr;
    TextEngine* text_engine_ = nullptr;
    
    std::queue<BatchRequest> pending_queue_;
    std::unordered_map<std::string, BatchRequest> active_requests_;
    std::unordered_map<std::string, BatchRequest> completed_requests_;
    
    std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;  // For get_stats()
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{false};
    std::thread scheduler_thread_;
    
    std::atomic<int> request_counter_{0};
    std::atomic<int> completed_count_{0};
    std::atomic<int> failed_count_{0};
    std::atomic<long long> total_tokens_{0};
    std::chrono::steady_clock::time_point start_time_;
    
    // Main scheduler loop
    void run_scheduler();
    
    // Form a batch from pending + active requests
    std::vector<BatchRequest*> form_batch();
    
    // Process a batch (prefill for new, decode for active)
    void process_batch(std::vector<BatchRequest*>& batch);
    
    // Run prefill for new requests
    void run_prefill(std::vector<BatchRequest*>& new_requests);
    
    // Run decode step for active requests
    void run_decode(std::vector<BatchRequest*>& active);
    
    // Check if request is finished
    bool is_finished(BatchRequest& req);
    
    // Clean up finished request
    void finish_request(BatchRequest& req, const std::string& finish_reason);
    
    // Generate unique request ID
    std::string generate_request_id();
};

} // namespace barskuy::engine