#include "logger.hpp"
#include <cstdlib>
#include <iostream>

namespace barskuy::core {

LogLevel Logger::current_level_ = LogLevel::Info;
std::unique_ptr<std::ofstream> Logger::log_file_;
std::mutex Logger::log_mutex_;
bool Logger::initialized_ = false;

void Logger::init(LogLevel level, const std::string& file_path) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    current_level_ = level;
    initialized_ = true;

    // F9-12: BARSKUY_LOG_FILE env allows daemon logging without a flag
    std::string log_path = file_path;
    if (log_path.empty()) {
        if (const char* env_log = std::getenv("BARSKUY_LOG_FILE")) log_path = env_log;
    }
    if (!log_path.empty()) {
        log_file_ = std::make_unique<std::ofstream>(log_path, std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Failed to open log file: " << log_path << std::endl;
        }
    }
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_) {
        log_file_->flush();
        log_file_->close();
        log_file_.reset();
    }
    initialized_ = false;
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "UNKNOWN";
}

std::string Logger::current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time_t);
#else
    localtime_r(&time_t, &tm);
#endif

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
    return fmt::format("{}.{:03d}", buffer, ms.count());
}

} // namespace barskuy::core