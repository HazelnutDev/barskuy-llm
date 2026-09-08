#include "job_queue.hpp"
#include "core/logger.hpp"

namespace barskuy::queue {

JobQueue::JobQueue(registry::ModelRegistry& model_registry)
    : model_registry_(model_registry) {}

JobQueue::~JobQueue() {
    shutdown();
}

bool JobQueue::initialize() {
    running_ = true;

    for (size_t i = 0; i < image_worker_count_; ++i) {
        image_workers_.emplace_back(&JobQueue::worker_loop, this,
            std::ref(image_queue_), std::ref(image_mutex_), std::ref(image_cv_), std::ref(running_));
    }

    for (size_t i = 0; i < video_worker_count_; ++i) {
        video_workers_.emplace_back(&JobQueue::worker_loop, this,
            std::ref(video_queue_), std::ref(video_mutex_), std::ref(video_cv_), std::ref(running_));
    }

    core::Logger::info("Job queue initialized with {} image workers, {} video workers",
        image_worker_count_, video_worker_count_);
    return true;
}

void JobQueue::shutdown() {
    running_ = false;
    image_cv_.notify_all();
    video_cv_.notify_all();

    for (auto& t : image_workers_) {
        if (t.joinable()) t.join();
    }
    for (auto& t : video_workers_) {
        if (t.joinable()) t.join();
    }

    image_workers_.clear();
    video_workers_.clear();

    core::Logger::info("Job queue shutdown complete");
}

void JobQueue::enqueue_image_job(const registry::GenerationJob& job, JobHandler handler) {
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        image_queue_.push({job, handler});
    }
    image_cv_.notify_one();
    model_registry_.update_job_status(job.id, "queued", 0);
}

void JobQueue::enqueue_video_job(const registry::GenerationJob& job, JobHandler handler) {
    {
        std::lock_guard<std::mutex> lock(video_mutex_);
        video_queue_.push({job, handler});
    }
    video_cv_.notify_one();
    model_registry_.update_job_status(job.id, "queued", 0);
}

size_t JobQueue::image_queue_size() const {
    std::lock_guard<std::mutex> lock(image_mutex_);
    return image_queue_.size();
}

size_t JobQueue::video_queue_size() const {
    std::lock_guard<std::mutex> lock(video_mutex_);
    return video_queue_.size();
}

void JobQueue::set_image_worker_count(size_t count) {
    image_worker_count_ = count;
}

void JobQueue::set_video_worker_count(size_t count) {
    video_worker_count_ = count;
}

void JobQueue::worker_loop(std::queue<QueuedJob>& queue,
                           std::mutex& mutex,
                           std::condition_variable& cv,
                           std::atomic<bool>& running) {
    while (running) {
        QueuedJob job;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return !queue.empty() || !running; });

            if (!running && queue.empty()) {
                break;
            }

            if (queue.empty()) continue;

            job = std::move(queue.front());
            queue.pop();
        }

        model_registry_.update_job_status(job.job.id, "processing", 0);

        try {
            job.handler(job.job);
            model_registry_.complete_job(job.job.id, job.job.result_path.value_or(""));
        } catch (const std::exception& e) {
            core::Logger::error("Job {} failed: {}", job.job.id, e.what());
            model_registry_.fail_job(job.job.id, e.what());
        }
    }
}

} // namespace barskuy::queue