#pragma once

#include <string>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include "registry/model_registry.hpp"

namespace barskuy::queue {

class JobQueue {
public:
    using JobHandler = std::function<void(const registry::GenerationJob&)>;

    JobQueue(registry::ModelRegistry& model_registry);
    ~JobQueue();

    bool initialize();
    void shutdown();

    void enqueue_image_job(const registry::GenerationJob& job, JobHandler handler);
    void enqueue_video_job(const registry::GenerationJob& job, JobHandler handler);

    size_t image_queue_size() const;
    size_t video_queue_size() const;

    void set_image_worker_count(size_t count);
    void set_video_worker_count(size_t count);

private:
    struct QueuedJob {
        registry::GenerationJob job;
        JobHandler handler;
    };

    void worker_loop(std::queue<QueuedJob>& queue,
                     std::mutex& mutex,
                     std::condition_variable& cv,
                     std::atomic<bool>& running);

    registry::ModelRegistry& model_registry_;

    std::queue<QueuedJob> image_queue_;
    std::queue<QueuedJob> video_queue_;
    mutable std::mutex image_mutex_;
    mutable std::mutex video_mutex_;
    std::condition_variable image_cv_;
    std::condition_variable video_cv_;

    std::atomic<bool> running_{false};
    std::vector<std::thread> image_workers_;
    std::vector<std::thread> video_workers_;
    size_t image_worker_count_ = 1;
    size_t video_worker_count_ = 1;
};

} // namespace barskuy::queue