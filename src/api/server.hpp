#pragma once

#include <string>
#include <memory>
#include <httplib.h>
#include "registry/model_registry.hpp"
#include "queue/job_queue.hpp"
#include "agent/agent_tools.hpp"
#include "agent/agent_mcp.hpp"
#include "agent/agent_loop.hpp"
#include "api/webui_embed.hpp"
#include "engine/text_engine.hpp"
#include "engine/image_engine.hpp"
#include "engine/video_engine.hpp"
#include "core/config.hpp"
#include <thread>
#include <atomic>
#include <mutex>

namespace barskuy::api {

class Server {
public:
    Server(const core::Config& config, registry::ModelRegistry& model_registry, queue::JobQueue& job_queue, 
           engine::TextEngine& text_engine, engine::ImageEngine& image_engine, engine::VideoEngine& video_engine);
    ~Server();

    bool start(const std::string& host, uint16_t port);
    void stop();

    // Run server in current thread (blocking). Returns false on bind failure.
    bool run(const std::string& host, uint16_t port);

private:
    void setup_routes();
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_models_list(const httplib::Request& req, httplib::Response& res);
    void handle_model_register(const httplib::Request& req, httplib::Response& res);
    void handle_model_unload(const httplib::Request& req, httplib::Response& res);
    void handle_props(const httplib::Request& req, httplib::Response& res);
    void handle_router_load(const httplib::Request& req, httplib::Response& res);
    void handle_router_unload(const httplib::Request& req, httplib::Response& res);
    void handle_complete(const httplib::Request& req, httplib::Response& res);
    void handle_embeddings(const httplib::Request& req, httplib::Response& res);
    void handle_generate_image(const httplib::Request& req, httplib::Response& res);
    void handle_generate_image_status(const httplib::Request& req, httplib::Response& res);
    void handle_generate_video(const httplib::Request& req, httplib::Response& res);
    void handle_generate_video_status(const httplib::Request& req, httplib::Response& res);
    void handle_jobs_list(const httplib::Request& req, httplib::Response& res);
    void handle_api_keys(const httplib::Request& req, httplib::Response& res);
    void handle_lora_load(const httplib::Request& req, httplib::Response& res);
    void handle_lora_unload(const httplib::Request& req, httplib::Response& res);
    void handle_lora_list(const httplib::Request& req, httplib::Response& res);
    void handle_tools_list(const httplib::Request& req, httplib::Response& res);
    void handle_tools_call(const httplib::Request& req, httplib::Response& res);
    void handle_webui(const httplib::Request& req, httplib::Response& res);
    void handle_streams_lookup(const httplib::Request& req, httplib::Response& res);
    void init_agent(); // F7: tools + MCP setup from config
    bool execute_tool(const std::string& name, const nlohmann::json& args,
                      std::string& out, const std::string& cwd = "");

    void handle_metrics(const httplib::Request& req, httplib::Response& res);

    bool authenticate(const httplib::Request& req, std::string& api_key_id);
    void send_error(httplib::Response& res, int status, const std::string& type, const std::string& message, const std::string& code);
    void send_sse_chunk(httplib::Response& res, const std::string& data);

    const core::Config& config_;
    registry::ModelRegistry& model_registry_;
    queue::JobQueue& job_queue_;
    engine::TextEngine& text_engine_;
    engine::ImageEngine& image_engine_;
    engine::VideoEngine& video_engine_;

    // Agent & tools (F7, default on; --no-tools / --no-agent / --no-mcp to disable)
    agent::AgentTools agent_tools_;
    agent::McpManager mcp_;
    agent::AgentRunner agent_runner_;
    std::once_flag agent_init_once_;

    // Fase 9: active model swap (dialog dropdown). /models/load sets it;
    // /props, /v1/models order and chat default follow it.
    mutable std::mutex active_mu_;
    std::string active_model_id_;
    void set_active_model(const std::string& id);
    std::string get_active_model() const;

    // Prometheus-style counters (F6-7 observability)
    mutable std::atomic<uint64_t> metric_complete_requests_{0};
    mutable std::atomic<uint64_t> metric_embeddings_requests_{0};
    mutable std::atomic<uint64_t> metric_image_requests_{0};
    mutable std::atomic<uint64_t> metric_video_requests_{0};
    mutable std::atomic<uint64_t> metric_models_requests_{0};
    mutable std::atomic<uint64_t> metric_tools_requests_{0};
    mutable std::atomic<uint64_t> metric_agent_steps_{0};
    mutable std::atomic<uint64_t> metric_prompt_tokens_{0};
    mutable std::atomic<uint64_t> metric_completion_tokens_{0};
    mutable std::atomic<uint64_t> metric_errors_total_{0};

    std::unique_ptr<httplib::Server> server_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};
};

} // namespace barskuy::api