#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iostream>
#include <fmt/core.h>
#include <fmt/format.h>

namespace barskuy::core {

enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error
};

class Logger {
public:
    static void init(LogLevel level = LogLevel::Info, const std::string& file_path = "");
    static void shutdown();

    template<typename... Args>
    static void debug(const char* fmt, Args&&... args) {
        log(LogLevel::Debug, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void info(const char* fmt, Args&&... args) {
        log(LogLevel::Info, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void warn(const char* fmt, Args&&... args) {
        log(LogLevel::Warn, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void error(const char* fmt, Args&&... args) {
        log(LogLevel::Error, fmt, std::forward<Args>(args)...);
    }

private:
    template<typename... Args>
    static void log(LogLevel level, const char* fmt, Args&&... args);
    static std::string level_to_string(LogLevel level);
    static std::string current_timestamp();

    static LogLevel current_level_;
    static std::unique_ptr<std::ofstream> log_file_;
    static std::mutex log_mutex_;
    static bool initialized_;
};

template<typename... Args>
void Logger::log(LogLevel level, const char* fmt, Args&&... args) {
    if (level < current_level_ || !initialized_) return;

    std::lock_guard<std::mutex> lock(log_mutex_);

    std::string message = fmt::format(fmt, std::forward<Args>(args)...);
    std::string timestamp = current_timestamp();
    std::string level_str = level_to_string(level);

    std::string full_message = fmt::format("[{}] [{}] {}", timestamp, level_str, message);

    if (level >= LogLevel::Warn) {
        std::cerr << full_message << std::endl;
    } else {
        std::cout << full_message << std::endl;
    }

    if (log_file_ && log_file_->is_open()) {
        *log_file_ << full_message << std::endl;
        log_file_->flush();
    }
}

} // namespace barskuy::core