#include "api/server.hpp"
#include "core/config.hpp"
#include "core/logger.hpp"
#include "registry/model_registry.hpp"
#include "queue/job_queue.hpp"
#include "engine/text_engine.hpp"
#include "engine/image_engine.hpp"
#include "engine/video_engine.hpp"

#include <iostream>
#include <memory>
#include <signal.h>
#include <atomic>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#endif
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <openssl/sha.h>

static std::unique_ptr<barskuy::api::Server> g_server;
static std::unique_ptr<barskuy::registry::ModelRegistry> g_model_registry;
static std::unique_ptr<barskuy::queue::JobQueue> g_job_queue;
static std::unique_ptr<barskuy::engine::TextEngine> g_text_engine;
static std::unique_ptr<barskuy::engine::ImageEngine> g_image_engine;
static std::unique_ptr<barskuy::engine::VideoEngine> g_video_engine;
static std::atomic<bool> g_shutdown{false};
#ifdef _WIN32
static HANDLE g_port_lock = nullptr;
#else
static int g_port_lock_fd = -1;
#endif

// Single-instance guard per port (F9-12): on Windows SO_REUSEADDR lets two
// processes bind the same port, splitting traffic unpredictably. Hold an
// exclusive lock file so the second instance fails loud instead.
static bool acquire_port_lock(uint16_t port) {
    char path[256];
#ifdef _WIN32
    DWORD n = GetTempPathA(sizeof(path) - 32, path);
    if (n == 0 || n > sizeof(path) - 32) return false;
    snprintf(path + n, sizeof(path) - n, "barskuy-llm-%u.lock", (unsigned)port);
    g_port_lock = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (g_port_lock == INVALID_HANDLE_VALUE) return false;
    char pid[32];
    snprintf(pid, sizeof(pid), "%lu\n", (unsigned long)GetCurrentProcessId());
    DWORD written = 0;
    WriteFile(g_port_lock, pid, (DWORD)strlen(pid), &written, nullptr);
    FlushFileBuffers(g_port_lock);
    return true;
#else
    snprintf(path, sizeof(path), "/tmp/barskuy-llm-%u.lock", (unsigned)port);
    g_port_lock_fd = open(path, O_RDWR | O_CREAT, 0644);
    if (g_port_lock_fd < 0) return false;
    if (flock(g_port_lock_fd, LOCK_EX | LOCK_NB) != 0) {
        close(g_port_lock_fd);
        g_port_lock_fd = -1;
        return false;
    }
    return true;
#endif
}

void signal_handler(int sig) {
    barskuy::core::Logger::info("Received signal {}, shutting down...", sig);
    g_shutdown = true;
    if (g_server) {
        g_server->stop();
    }
}

int main(int argc, char* argv[]) {
    barskuy::core::Logger::init();

    barskuy::core::Config config;
    if (!config.load(argc, argv)) {
        barskuy::core::Logger::error("Failed to load configuration");
        return 1;
    }

    barskuy::core::Logger::info("Starting barskuy-llm v{}", BARSKUY_VERSION);

    g_model_registry = std::make_unique<barskuy::registry::ModelRegistry>(config.database_path());
    if (!g_model_registry->initialize()) {
        barskuy::core::Logger::error("Failed to initialize model registry");
        return 1;
    }

    // Bootstrap API key from env (F6-6): first-run provisioning without chicken-and-egg.
    if (const char* bootstrap_key = std::getenv("BARSKUY_API_KEY")) {
        std::string key_str(bootstrap_key);
        if (!key_str.empty()) {
            unsigned char hash[SHA256_DIGEST_LENGTH];
            SHA256(reinterpret_cast<const unsigned char*>(key_str.c_str()), key_str.length(), hash);
            std::stringstream ss;
            for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
                ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
            }
            if (!g_model_registry->get_api_key_by_hash(ss.str())) {
                barskuy::registry::ApiKey key;
                key.id = "key_bootstrap";
                key.key_hash = ss.str();
                key.label = "bootstrap";
                key.created_at = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                if (g_model_registry->create_api_key(key)) {
                    barskuy::core::Logger::info("Bootstrapped API key from BARSKUY_API_KEY env");
                } else {
                    barskuy::core::Logger::warn("Failed to bootstrap API key from env");
                }
            }
        }
    }

    g_job_queue = std::make_unique<barskuy::queue::JobQueue>(*g_model_registry);
    if (!g_job_queue->initialize()) {
        barskuy::core::Logger::error("Failed to initialize job queue");
        return 1;
    }

    g_text_engine = std::make_unique<barskuy::engine::TextEngine>();
    if (!g_text_engine->initialize(config.models_dir())) {
        barskuy::core::Logger::error("Failed to initialize text engine");
        return 1;
    }
    g_text_engine->set_max_loaded_models(config.max_loaded_models());
    if (!config.tensor_split().empty()) {
        g_text_engine->set_tensor_split(config.tensor_split());
    }
    g_text_engine->set_split_mode(config.split_mode());
    g_text_engine->set_n_batch(config.n_batch());
    g_text_engine->set_n_ubatch(config.n_ubatch());
    g_text_engine->set_n_threads_batch(config.n_cpu_moe());
    g_text_engine->set_cache_types(config.cache_type_k(), config.cache_type_v());
    g_text_engine->set_use_mmap(config.enable_mmap());
    g_text_engine->set_context_cap((int)config.max_context_size());
    barskuy::core::Logger::info("Context: batch={} ubatch={} n_cpu_moe={} cache_k={} cache_v={} flash_attn={} mmap={}",
        config.n_batch(), config.n_ubatch(), config.n_cpu_moe(),
        config.cache_type_k(), config.cache_type_v(), config.flash_attn(),
        config.enable_mmap() ? "on" : "off");
#ifdef _WIN32
    // F9-9 --prio: process priority class (0 idle .. 4 high)
    {
        DWORD cls = NORMAL_PRIORITY_CLASS;
        if (config.prio() <= 0) cls = IDLE_PRIORITY_CLASS;
        else if (config.prio() == 1) cls = BELOW_NORMAL_PRIORITY_CLASS;
        else if (config.prio() == 2) cls = NORMAL_PRIORITY_CLASS;
        else if (config.prio() == 3) cls = ABOVE_NORMAL_PRIORITY_CLASS;
        else cls = HIGH_PRIORITY_CLASS;
        SetPriorityClass(GetCurrentProcess(), cls);
    }
#endif

    // Fase 8 WebUI-style startup: --api-key seeds a fixed key (same store as env bootstrap).
    if (!config.api_key().empty()) {
        std::string key_str = config.api_key();
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(key_str.c_str()), key_str.length(), hash);
        std::stringstream ss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
        }
        if (!g_model_registry->get_api_key_by_hash(ss.str())) {
            barskuy::registry::ApiKey key;
            key.id = "key_flag";
            key.key_hash = ss.str();
            key.label = "flag";
            key.created_at = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (g_model_registry->create_api_key(key)) {
                barskuy::core::Logger::info("Seeded API key from --api-key flag");
            }
        }
    }

    // Fase 8: -m/--model preload (ala llama-server): register + load text models now.
    for (auto& mpath : config.preload_models()) {
        std::string stem = mpath;
        size_t slash = stem.find_last_of("/\\");
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        size_t dot = stem.find_last_of('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        if (!g_model_registry->get_model(stem)) {
            barskuy::registry::Model m;
            m.id = stem;
            m.path = mpath;
            m.format = "gguf";
            m.capabilities = {"text"};
            m.created_at = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            g_model_registry->register_model(m);
        }
        if (g_text_engine->load_model(stem, mpath)) {
            barskuy::core::Logger::info("Preloaded model: {} ({})", stem, mpath);
        } else {
            barskuy::core::Logger::warn("Preload failed (will lazy-load on request): {}", mpath);
        }
    }

    g_image_engine = std::make_unique<barskuy::engine::ImageEngine>();
    if (!g_image_engine->initialize()) {
        barskuy::core::Logger::error("Failed to initialize image engine");
        return 1;
    }
    if (config.image_cpu() > 0) g_image_engine->set_cpu_threads(config.image_cpu());
    // Preload image/video models if specified (manual path, separate from -m text)
    auto preload_one = [&](const std::string& path, const std::string& cap) {
        if (path.empty()) return;
        std::string stem = path;
        size_t slash = stem.find_last_of("/\\");
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        size_t dot = stem.find_last_of('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        if (!g_model_registry->get_model(stem)) {
            barskuy::registry::Model m;
            m.id = stem;
            m.path = path;
            m.format = "gguf";
            m.capabilities = {cap};
            m.created_at = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            g_model_registry->register_model(m);
        }
    };
    preload_one(config.image_model(), "image");
    preload_one(config.video_model(), "video");

    g_server = std::make_unique<barskuy::api::Server>(config, *g_model_registry, *g_job_queue, *g_text_engine, *g_image_engine, *g_video_engine);

    if (!acquire_port_lock(config.port())) {
        barskuy::core::Logger::error("FATAL: port {} is held by another barskuy-llm instance - bunuh proses lama dulu atau ganti --port", config.port());
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!g_server->run(config.host(), config.port())) {
        barskuy::core::Logger::error("Server failed to start (see FATAL above)");
        return 1;
    }

    barskuy::core::Logger::info("Server stopped gracefully");
    return 0;
}