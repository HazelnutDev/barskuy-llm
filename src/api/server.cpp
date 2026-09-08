#include "server.hpp"
#include "core/logger.hpp"
#include "utils/utils.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <thread>
#include <filesystem>
#include <unordered_set>
#include <deque>
#include <condition_variable>
#include <mutex>

#include "engine/text_engine.hpp"
#include "engine/image_engine.hpp"
#include "engine/video_engine.hpp"
#include "engine/preview_media.hpp"

namespace barskuy::api {

namespace {
// F9-10: thread-safe SSE line queue for true token streaming.
struct SseQueue {
    mutable std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool done = false;
    void push(const std::string& s) {
        std::lock_guard<std::mutex> lk(mu);
        q.push_back(s);
        cv.notify_one();
    }
    void finish() {
        std::lock_guard<std::mutex> lk(mu);
        done = true;
        cv.notify_all();
    }
    // true = got a line; false = drained after finish
    bool pop(std::string& out, int wait_ms = 5000) {
        std::unique_lock<std::mutex> lk(mu);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
        while (q.empty() && !done) {
            if (cv.wait_until(lk, deadline) == std::cv_status::timeout) return false;
        }
        if (!q.empty()) { out = std::move(q.front()); q.pop_front(); return true; }
        return false;
    }
    bool finished() const {
        std::lock_guard<std::mutex> lk(mu);
        return done && q.empty();
    }
};

// F9-10: split <think> blocks into reasoning deltas live (collapsible UI).
// Tags may straddle token pieces; up to 7 trailing chars stay unclassified.
// F9-10 BUG-049: prompt Qwen3 diakhiri prefill <think> -> model tak emit
// opener, langsung isi think. Splitter harus bisa mulai dalam mode think
// (in_think=true dari awal) + buang opener bila model echo ulang.
struct ThinkSplitter {
    bool in_think = false;
    bool first = true;
    std::string buf;
    struct Parts { std::string content; std::string reasoning; };
    Parts push(const std::string& s) {
        buf += s;
        if (first && in_think) {
            if (buf.rfind("<think>", 0) == 0) buf.erase(0, 7);
            first = false;
        }
        Parts o;
        for (;;) {
            if (!in_think) {
                size_t p = buf.find("<think>");
                if (p == std::string::npos) {
                    size_t keep = 0;
                    for (size_t k = 1; k <= 6 && k < buf.size(); ++k) {
                        if (buf.compare(buf.size() - k, k, "<think>", k) == 0) { keep = k; break; }
                    }
                    o.content += buf.substr(0, buf.size() - keep);
                    buf = buf.substr(buf.size() - keep);
                    break;
                }
                o.content += buf.substr(0, p);
                buf = buf.substr(p + 7);
                in_think = true;
            } else {
                size_t p = buf.find("</think>");
                if (p == std::string::npos) {
                    size_t keep = 0;
                    for (size_t k = 1; k <= 7 && k < buf.size(); ++k) {
                        if (buf.compare(buf.size() - k, k, "</think>", k) == 0) { keep = k; break; }
                    }
                    o.reasoning += buf.substr(0, buf.size() - keep);
                    buf = buf.substr(buf.size() - keep);
                    break;
                }
                o.reasoning += buf.substr(0, p);
                buf = buf.substr(p + 8);
                in_think = false;
            }
        }
        return o;
    }
    // remainder at end of stream (unclosed think -> reasoning)
    Parts flush() {
        Parts o;
        if (in_think) o.reasoning = std::move(buf);
        else o.content = std::move(buf);
        buf.clear();
        return o;
    }
};

// Split a finished text into {content, reasoning} (non-stream paths)
// F9-10 BUG-046: Qwen3 sometimes emits an empty think (<think></think>).
// Whitespace-only reasoning must count as absent, or the UI renders an
// empty "Reasoning" block (reasoningContent "\n\n" is truthy client-side).
inline bool is_blank(const std::string& s) {
    for (unsigned char c : s) if (c!=' '&&c!='\t'&&c!='\n'&&c!='\r') return false;
    return true;
}
inline std::pair<std::string, std::string> split_thinking(const std::string& text,
                                                          bool prefilled = false) {
    if (!prefilled) {
        // auto-detect prefilled think: </think> tanpa opener sebelumnya
        size_t c = text.find("</think>");
        if (c != std::string::npos) {
            size_t o = text.find("<think>");
            if (o == std::string::npos || o > c) prefilled = true;
        }
    }
    ThinkSplitter ts;
    ts.in_think = prefilled;
    auto p = ts.push(text);
    auto f = ts.flush();
    return {p.content + f.content, p.reasoning + f.reasoning};
}

inline nlohmann::json timings_json(int prompt_n, double prompt_ms, int pred_n, double pred_ms) {
    nlohmann::json t;
    t["prompt_n"] = prompt_n;
    t["prompt_ms"] = prompt_ms;
    t["predicted_n"] = pred_n;
    t["predicted_ms"] = pred_ms;
    return t;
}

// Unified perf line: same numbers go to terminal log and WebUI (F9-10 sync)
inline void log_perf(const std::string& model, int prompt_n, double prompt_ms,
                     int pred_n, double pred_ms) {
    double pre_s = prompt_n > 0 && prompt_ms > 0 ? 1000.0 * prompt_n / prompt_ms : 0.0;
    double dec_s = pred_n > 0 && pred_ms > 0 ? 1000.0 * pred_n / pred_ms : 0.0;
    char buf[256];
    snprintf(buf, sizeof(buf),
        "[perf] %s prefill %dtok/%.0fms (%.1f tok/s) decode %dtok/%.0fms (%.1f tok/s)",
        model.c_str(), prompt_n, prompt_ms, pre_s, pred_n, pred_ms, dec_s);
    core::Logger::info("{}", buf);
}

// Write RGBA frames as BMP files under ./files, return public urls.
// Used by image/video preview output (Fase 8) until real diffusion lands.
std::vector<std::string> save_preview_frames(const std::string& job_id,
    const std::vector<std::vector<uint8_t>>& frames, const std::vector<int>& widths,
    const std::vector<int>& heights) {
    std::vector<std::string> urls;
    std::error_code ec;
    std::filesystem::create_directories("./image", ec);
    for (size_t i = 0; i < frames.size(); ++i) {
        int w = i < widths.size() ? widths[i] : 0;
        int h = i < heights.size() ? heights[i] : 0;
        if (w <= 0 || h <= 0) continue;
        std::string fname = job_id + "_f" + std::to_string(i) + ".png";
        bool ok = engine::preview::write_png("./image/" + fname, frames[i], w, h);
        if (!ok) ok = engine::preview::write_bmp("./image/" + fname + ".bmp", frames[i], w, h);
        if (ok) urls.push_back("/image/" + fname);
    }
    return urls;
}
}

Server::Server(const core::Config& config, registry::ModelRegistry& model_registry, queue::JobQueue& job_queue,
               engine::TextEngine& text_engine, engine::ImageEngine& image_engine, engine::VideoEngine& video_engine)
    : config_(config), model_registry_(model_registry), job_queue_(job_queue),
      text_engine_(text_engine), image_engine_(image_engine), video_engine_(video_engine),
      agent_tools_(config.tools_root()), agent_runner_(text_engine_, agent_tools_, mcp_) {
    // F7 tools selection: "all" (default) or comma list; "" disables
    if (!config.tools_mode().empty() && config.tools_mode() != "all") {
        std::vector<std::string> names;
        std::string cur;
        for (char c : config.tools_mode() + ",") {
            if (c == ',') { if (!cur.empty()) names.push_back(cur); cur.clear(); }
            else if (c != ' ') cur += c;
        }
        agent_tools_.set_enabled(names);
    }
    if (config.tools_mode().empty()) agent_tools_.set_enabled_all(false);
    server_ = std::make_unique<httplib::Server>();
    setup_routes();
}

void Server::init_agent() {
    if (!config_.tools_mode().empty()) {
        core::Logger::info("Agent tools enabled ({} tools, root={}) - EXPERIMENTAL, trusted env only",
            agent_tools_.tool_names().size(), agent_tools_.root());
    } else {
        core::Logger::info("Agent tools disabled (enable with --tools all)");
    }
    agent_runner_.set_max_steps(config_.max_agent_steps());
    if (config_.mcp_enabled()) {
        std::string path = config_.mcp_config().empty() ? "./mcp.json" : config_.mcp_config();
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            std::string err;
            if (mcp_.load_config_file(path, err)) {
                int n = mcp_.discover_all();
                core::Logger::info("MCP: {} tool(s) from {} server(s) [{}]", n, mcp_.server_names().size(), path);
            } else {
                core::Logger::warn("MCP config ignored: {}", err);
            }
        } else if (!config_.mcp_config().empty()) {
            core::Logger::warn("MCP config not found: {}", path);
        } else {
            core::Logger::info("MCP: no ./mcp.json, MCP disabled (see mcp.json.example)");
        }
    } else {
        core::Logger::info("MCP disabled (enable with --mcp)");
    }
    if (!config_.agent_enabled()) core::Logger::info("Agent loop disabled (enable with --agent)");
}

void Server::set_active_model(const std::string& id) {
    std::lock_guard<std::mutex> lk(active_mu_);
    active_model_id_ = id;
}

std::string Server::get_active_model() const {
    std::lock_guard<std::mutex> lk(active_mu_);
    return active_model_id_;
}

bool Server::execute_tool(const std::string& name, const nlohmann::json& args, std::string& out,
                          const std::string& cwd) {
    // F9-10 BUG-053: arguments ala OpenAI adalah STRING JSON; parse dulu.
    // Tanpa ini semua argumen string dibuang -> MCP tolak (undefined).
    nlohmann::json a = args;
    if (a.is_string()) {
        try { a = nlohmann::json::parse(a.get<std::string>()); }
        catch (...) { a = nlohmann::json::object(); }
    }
    if (!a.is_object()) a = nlohmann::json::object();
    if (agent_tools_.has_tool(name)) { out = agent_tools_.execute(name, a, cwd); return true; }
    if (mcp_.has_tool(name)) { out = mcp_.execute(name, a); return true; }
    return false;
}

Server::~Server() {
    stop();
}

bool Server::start(const std::string& host, uint16_t port) {
    core::Logger::info("Starting HTTP server on {}:{}", host, port);
    std::call_once(agent_init_once_, [this]() { init_agent(); }); // F7: tools + MCP before serving

    server_->set_logger([](const auto& req, const auto& res) {
        core::Logger::info("Request: {} {} -> {}", req.method, req.path, res.status);
    });

    server_->set_error_handler([](const auto& req, auto& res) {
        nlohmann::json error = {
            {"error", {
                {"type", "internal_error"},
                {"message", "Internal server error"},
                {"code", "INTERNAL_ERROR"}
            }}
        };
        res.set_content(error.dump(), "application/json");
    });

    running_ = true;
    server_thread_ = std::thread([this, host, port]() {
        this->run(host, port);
    });

    return true;
}

bool Server::run(const std::string& host, uint16_t port) {
    std::call_once(agent_init_once_, [this]() { init_agent(); }); // main() calls run() directly
    // Auth mode visible at startup (Fase 9): fresh DB = OPEN like llama-server
    // without --api-key; stale keys in DB (old sessions) mean REQUIRED.
    size_t nkeys = model_registry_.list_api_keys().size();
    if (nkeys == 0) {
        core::Logger::info("Auth: OPEN (no keys configured) - WebUI works without key; set --api-key to require auth");
    } else {
        core::Logger::info("Auth: REQUIRED ({} key(s) in store) - clients must send a valid Bearer key", nkeys);
    }
    core::Logger::info("Server listening on {}:{}", host, port);
    bool ok = server_->listen(host.c_str(), port);
    if (!ok) {
        core::Logger::error("FATAL: cannot bind {}:{} - port dipakai proses lain/firewall. Bunuh proses lama (taskkill) atau ganti --port", host, port);
        return false;
    }
    core::Logger::info("Server stopped");
    return true;
}

void Server::stop() {
    if (running_) {
        core::Logger::info("Stopping HTTP server...");
        server_->stop();
        running_ = false;
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }
}

void Server::setup_routes() {
    // F9-10: true-streaming providers block up to minutes; raise timeouts.
    // (here, not start(), because main() calls run() directly)
    server_->set_read_timeout(3600, 0);
    server_->set_write_timeout(3600, 0);
    server_->Get("/health", [this](const auto& req, auto& res) { handle_health(req, res); });
    server_->Get("/v1/models", [this](const auto& req, auto& res) { handle_models_list(req, res); });
    server_->Post("/v1/models/register", [this](const auto& req, auto& res) { handle_model_register(req, res); });
    server_->Post("/v1/models/unload", [this](const auto& req, auto& res) { handle_model_unload(req, res); });
    // llama.cpp WebUI compat (Fase 9): /props, router /models/load|unload
    server_->Get("/props", [this](const auto& req, auto& res) { handle_props(req, res); });
    server_->Post("/models/load", [this](const auto& req, auto& res) { handle_router_load(req, res); });
    server_->Post("/models/unload", [this](const auto& req, auto& res) { handle_router_unload(req, res); });
    // UI stream-resume probe (F9-10): we don't track conversation streams
    // server-side, so there is never anything to reattach to.
    server_->Post("/v1/streams/lookup", [this](const auto& req, auto& res) { handle_streams_lookup(req, res); });
    server_->Post("/v1/complete", [this](const auto& req, auto& res) { handle_complete(req, res); });
    server_->Post("/v1/chat/completions", [this](const auto& req, auto& res) { handle_complete(req, res); });
    server_->Post("/v1/completions", [this](const auto& req, auto& res) { handle_complete(req, res); });
    server_->Post("/v1/embeddings", [this](const auto& req, auto& res) { handle_embeddings(req, res); });
    server_->Post("/v1/generate/image", [this](const auto& req, auto& res) { handle_generate_image(req, res); });
    server_->Get(R"(/v1/generate/image/(.*))", [this](const auto& req, auto& res) { handle_generate_image_status(req, res); });
    server_->Post("/v1/generate/video", [this](const auto& req, auto& res) { handle_generate_video(req, res); });
    server_->Get(R"(/v1/generate/video/(.*))", [this](const auto& req, auto& res) { handle_generate_video_status(req, res); });
    server_->Get("/v1/jobs", [this](const auto& req, auto& res) { handle_jobs_list(req, res); });
    server_->Post("/v1/api-keys", [this](const auto& req, auto& res) { handle_api_keys(req, res); });
    server_->Post("/v1/models/lora/load", [this](const auto& req, auto& res) { handle_lora_load(req, res); });
    server_->Post("/v1/models/lora/unload", [this](const auto& req, auto& res) { handle_lora_unload(req, res); });
    server_->Get("/v1/models/lora", [this](const auto& req, auto& res) { handle_lora_list(req, res); });
    server_->Get("/tools", [this](const auto& req, auto& res) { handle_tools_list(req, res); });
    server_->Post("/tools", [this](const auto& req, auto& res) { handle_tools_call(req, res); });
    server_->Get("/metrics", [this](const auto& req, auto& res) { handle_metrics(req, res); });

    // Serve generated media: PNG di ./image / ./video saja (files dihapus)
    {
        std::error_code ec;
        std::filesystem::create_directories("./image", ec);
        std::filesystem::create_directories("./video", ec);
        if (std::filesystem::exists("./files")) std::filesystem::remove_all("./files", ec);
        server_->set_mount_point("/image", "./image");
        server_->set_mount_point("/video", "./video");
    }
    // Fase 9: embedded llama.cpp WebUI lives INSIDE the binary (no frontend dir).
    // Hash-router SPA: only / and bundled assets exist; API routes above win.
    core::Logger::info("Embedded WebUI: {} file(s) in binary", webui::count());
    server_->Get(R"(/(?!v1|health|metrics|tools|props|models|files|image|video|slots).*)",
        [this](const auto& req, auto& res) { handle_webui(req, res); });
}

void Server::handle_webui(const httplib::Request& req, httplib::Response& res) {
    // No-op service worker: kills the console SW-registration error. No
    // offline precache (upstream workbox bundle is intentionally excluded).
    if (req.path == "/sw.js") {
        res.set_content("self.addEventListener('fetch', () => {});\n",
                        "text/javascript");
        return;
    }
    const webui::Entry* e = webui::find(req.path);
    if (!e) {
        // Icon aliases -> embedded favicon.svg (quiet the 404 noise)
        if (req.path == "/favicon.ico" || req.path == "/apple-touch-icon-180x180.png" ||
            req.path == "/favicon-dark.svg" || req.path == "/favicon-dark.ico") {
            e = webui::find("/favicon.svg");
        }
        if (!e) {
            res.status = 404;
            res.set_content("not found", "text/plain");
            return;
        }
    }
    res.set_content(reinterpret_cast<const char*>(e->data), e->size, e->mime);
}

void Server::handle_tools_list(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    // Upstream shape: bare JSON ARRAY of ServerToolInfo (Fase 9-9)
    nlohmann::json out = nlohmann::json::array();
    for (auto& n : agent_tools_.tool_names()) {
        nlohmann::json info = agent_tools_.server_info(n);
        if (!info.is_null()) out.push_back(info);
    }
    for (auto& t : mcp_.definitions_openai()) {
        auto& f = t["function"];
        std::string full = f["name"].get<std::string>();
        out.push_back({{"display_name", full}, {"tool", full}, {"type", "server"},
            {"permissions", {{"write", false}}}, {"uses_cwd", false},
            {"definition", {{"type", "function"}, {"function", f}}}});
    }
    res.set_content(out.dump(), "application/json");
}

void Server::handle_tools_call(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    try {
        auto json = nlohmann::json::parse(req.body);
        std::string tool = json.value("tool", "");
        nlohmann::json params = json.contains("params") ? json["params"]
            : json.value("arguments", nlohmann::json::object());
        if (tool.empty()) {
            send_error(res, 400, "invalid_request", "tool is required", "MISSING_TOOL");
            return;
        }
        std::string out;
        auto t0 = std::chrono::steady_clock::now();
        // F9-10: hormati header x-tool-cwd UI (working directory picker).
        // Tanpa ini folder pilihan tak berpengaruh sama sekali.
        std::string cwd = req.get_header_value("x-tool-cwd");
        bool known = execute_tool(tool, params, out, cwd);
        auto tms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        // F9-10 agentic visibility: same event the WebUI shows per turn
        core::Logger::info("tool exec {} {}ms{}", tool.c_str(), (long long)tms,
            known ? "" : " (UNKNOWN TOOL)");
        if (!known) {
            send_error(res, 404, "not_found", "Unknown or disabled tool: " + tool, "TOOL_NOT_FOUND");
            return;
        }
        metric_tools_requests_++;
        bool stream = json.value("stream", false);
        if (stream) {
            std::string payload = "data: " + nlohmann::json({{"chunk", out}}).dump() + "\n\n";
            payload += "data: " + nlohmann::json({{"done", true}}).dump() + "\n\n";
            auto shared = std::make_shared<std::string>(std::move(payload));
            res.set_content_provider(
                shared->size(), "text/event-stream",
                [shared](size_t offset, size_t length, httplib::DataSink& sink) -> bool {
                    if (offset >= shared->size()) return false;
                    size_t n = std::min(length, shared->size() - offset);
                    sink.write(shared->data() + offset, n);
                    return true;
                });
            return;
        }
        nlohmann::json response;
        response["plain_text_response"] = out;
        // F9-10 BUG-057b: samakan upstream - tool yang outputnya JSON object
        // (glob: entries/base) diekspos top-level agar UI bisa baca langsung.
        try {
            nlohmann::json inner = nlohmann::json::parse(out);
            if (inner.is_object()) {
                for (auto& kv : inner.items()) {
                    if (!response.contains(kv.key())) response[kv.key()] = kv.value();
                }
            }
        } catch (...) {}
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_props(const httplib::Request& req, httplib::Response& res) {
    // llama.cpp WebUI compat (Fase 9): server props in MODEL-mode shape.
    // Extra barskuy fields (capabilities) are additive; the UI ignores unknowns.
    std::string wanted = req.get_param_value("model");
    auto models = model_registry_.list_models();
    std::unordered_set<std::string> resident;
    for (auto& lm : text_engine_.list_loaded_models()) {
        if (lm.loaded) resident.insert(lm.model_id);
    }
    const registry::Model* pick = nullptr;
    if (!wanted.empty()) {
        for (auto& m : models) if (m.id == wanted) { pick = &m; break; }
    }
    if (!pick) {
        std::string active = get_active_model();
        if (!active.empty()) {
            for (auto& m : models) if (m.id == active) { pick = &m; break; }
        }
    }
    if (!pick) {
        for (auto& m : models) if (resident.count(m.id)) { pick = &m; break; }
    }
    if (!pick && !models.empty()) pick = &models[0];

    std::vector<std::string> caps = pick ? pick->capabilities : std::vector<std::string>{"text"};
    auto has = [&](const std::string& c) {
        return std::find(caps.begin(), caps.end(), c) != caps.end();
    };
    int n_ctx = 8192;
    if (pick) {
        for (auto& lm : text_engine_.list_loaded_models()) {
            if (lm.model_id == pick->id && lm.config.n_ctx > 0) { n_ctx = lm.config.n_ctx; break; }
        }
    }
    nlohmann::json params = {
        {"n_predict", -1}, {"seed", 0}, {"temperature", 0.6},
        {"dynatemp_range", 0.0}, {"dynatemp_exponent", 1.0},
        {"top_k", 20}, {"top_p", 0.9}, {"min_p", 0.05}, {"top_n_sigma", -1.0},
        {"xtc_probability", 0.0}, {"xtc_threshold", 0.1}, {"typ_p", 1.0},
        {"repeat_last_n", 64}, {"repeat_penalty", 1.1},
        {"presence_penalty", 0.0}, {"frequency_penalty", 0.0},
        {"dry_multiplier", 0.0}, {"dry_base", 1.75}, {"dry_allowed_length", 2},
        {"dry_penalty_last_n", -1}, {"dry_sequence_breakers", nlohmann::json::array()},
        {"mirostat", 0}, {"mirostat_tau", 5.0}, {"mirostat_eta", 0.1},
        {"stop", nlohmann::json::array()}, {"max_tokens", -1},
        {"n_keep", 0}, {"n_discard", 0}, {"ignore_eos", false},
        {"stream", true}, {"logit_bias", nlohmann::json::array()},
        {"n_probs", 0}, {"min_keep", 0},
        {"grammar", ""}, {"grammar_lazy", false},
        {"grammar_triggers", nlohmann::json::array()},
        {"preserved_tokens", nlohmann::json::array()},
        {"chat_format", ""}, {"reasoning_format", ""}, {"reasoning_in_content", false},
        {"generation_prompt", ""}, {"samplers", nlohmann::json::array()},
        {"backend_sampling", true},
        {"speculative.n_max", 16}, {"speculative.n_min", 5}, {"speculative.p_min", 0.75},
        {"timings_per_token", false}, {"post_sampling_probs", false},
        {"lora", nlohmann::json::array()}
    };
    nlohmann::json response;
    response["default_generation_settings"] = {
        {"id", 0}, {"id_task", 0}, {"n_ctx", n_ctx},
        {"speculative", false}, {"is_processing", false},
        {"params", params}, {"prompt", ""},
        {"next_token", {{"has_next_token", false}, {"has_new_line", false},
                        {"n_remain", 0}, {"n_decoded", 0}, {"stopping_word", ""}}}
    };
    response["total_slots"] = 1;
    response["model_path"] = pick ? pick->path : "";
    response["role"] = "server";
    response["modalities"] = {
        {"vision", has("vision")}, {"audio", false}, {"video", has("video")}
    };
    response["chat_template"] = "";
    response["bos_token"] = "";
    response["eos_token"] = "";
    response["build_info"] = std::string("barskuy-llm ") + BARSKUY_VERSION;
    response["capabilities"] = caps;
    // F9-10 Barskuy detail block for the Model Information dialog
    {
        nlohmann::json b;
        b["capabilities"] = caps;
        std::string quant;
        int vocab = 0, embd = 0, layers = 0, heads = 0, n_gpu = 0;
        int ctx_train = 0, ctx_effective = n_ctx;
        size_t file_mb = 0;
        for (auto& lm : text_engine_.list_loaded_models()) {
            if (pick && lm.model_id == pick->id && lm.loaded) {
                quant = lm.config.quantization;
                vocab = lm.config.n_vocab;
                embd = lm.config.n_embd;
                layers = lm.config.n_layer;
                heads = lm.config.n_head;
                n_gpu = lm.n_gpu_layers_used;
                ctx_train = lm.config.n_ctx_train > 0 ? lm.config.n_ctx_train : lm.config.n_ctx;
                ctx_effective = lm.config.n_ctx;
                file_mb = lm.mem_size / (1024 * 1024);
                break;
            }
        }
        b["quantization"] = quant;
        b["vocab"] = vocab;
        b["embed"] = embd;
        b["layers"] = layers;
        b["heads"] = heads;
        b["file_size_mb"] = file_mb;
        b["backend"] = n_gpu > 0 ? "GPU+CPU" : "CPU";
        b["n_gpu_layers"] = n_gpu;
        b["ctx_train"] = ctx_train;
        b["ctx_effective"] = ctx_effective;
        b["tools"] = (int)agent_tools_.tool_names().size();
        b["mcp_servers"] = mcp_.server_names();
        response["barskuy"] = b;
    }
    res.set_content(response.dump(), "application/json");
}

void Server::handle_router_load(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    try {
        auto json = nlohmann::json::parse(req.body);
        std::string model_id = json.value("model", "");
        nlohmann::json response;
        if (model_id.empty()) { response["success"] = false; response["error"] = "model is required"; }
        else {
            auto m = model_registry_.get_model(model_id);
            if (!m) { response["success"] = false; response["error"] = "Model not found"; }
            else if (text_engine_.is_loaded(model_id)) { response["success"] = true; }
            else {
                bool text_cap = false;
                for (auto& c : m->capabilities) {
                    if (c == "text" || c == "vision" || c == "embeddings") { text_cap = true; break; }
                }
                if (text_cap) response["success"] = text_engine_.load_model(model_id, m->path);
                else response["success"] = true; // image/video engines load lazily
                if (response["success"].get<bool>()) set_active_model(model_id);
                else response["error"] = "load failed";
            }
        }
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_router_unload(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    try {
        auto json = nlohmann::json::parse(req.body);
        std::string model_id = json.value("model", "");
        nlohmann::json response;
        response["success"] = model_id.empty() ? false : text_engine_.unload_model(model_id);
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_streams_lookup(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    // One entry per matching LIVE session keyed by conversation_id.
    // Barskuy streams are request-scoped (no resume); upstream UI expects
    // an ARRAY here (object form breaks its probe with "non-array response").
    nlohmann::json response = nlohmann::json::array();
    res.set_content(response.dump(), "application/json");
}

void Server::handle_metrics(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    std::ostringstream out;
    out << "# HELP barskuy_requests_total Total HTTP requests per endpoint\n";
    out << "# TYPE barskuy_requests_total counter\n";
    out << "barskuy_requests_total{endpoint=\"complete\"} " << metric_complete_requests_.load() << "\n";
    out << "barskuy_requests_total{endpoint=\"embeddings\"} " << metric_embeddings_requests_.load() << "\n";
    out << "barskuy_requests_total{endpoint=\"generate_image\"} " << metric_image_requests_.load() << "\n";
    out << "barskuy_requests_total{endpoint=\"generate_video\"} " << metric_video_requests_.load() << "\n";
    out << "barskuy_requests_total{endpoint=\"models\"} " << metric_models_requests_.load() << "\n";
    out << "barskuy_requests_total{endpoint=\"tools\"} " << metric_tools_requests_.load() << "\n";
    out << "barskuy_agent_steps_total " << metric_agent_steps_.load() << "\n";
    out << "# HELP barskuy_tokens_total Total tokens processed\n";
    out << "# TYPE barskuy_tokens_total counter\n";
    out << "barskuy_tokens_total{kind=\"prompt\"} " << metric_prompt_tokens_.load() << "\n";
    out << "barskuy_tokens_total{kind=\"completion\"} " << metric_completion_tokens_.load() << "\n";
    out << "# HELP barskuy_errors_total Total error responses\n";
    out << "# TYPE barskuy_errors_total counter\n";
    out << "barskuy_errors_total " << metric_errors_total_.load() << "\n";
    res.set_content(out.str(), "text/plain; version=0.0.4");
}

void Server::handle_health(const httplib::Request& req, httplib::Response& res) {
    core::Logger::debug("Handling health request");
    nlohmann::json response = {
        {"status", "ok"},
        {"version", BARSKUY_VERSION},
        {"build", std::string(__DATE__) + " " + __TIME__}
    };
    res.set_content(response.dump(), "application/json");
    core::Logger::debug("Health response sent");
}

void Server::handle_models_list(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    metric_models_requests_++;
    auto models = model_registry_.list_models();
    // Active model first so MODEL-mode UIs (models[0]) follow swaps (Fase 9)
    {
        std::string active = get_active_model();
        if (!active.empty()) {
            std::stable_partition(models.begin(), models.end(),
                [&](const registry::Model& m) { return m.id == active; });
        }
    }
    // Source of truth for resident state is the engine (hot-swap aware, F1-8)
    std::unordered_set<std::string> resident;
    for (auto& lm : text_engine_.list_loaded_models()) {
        if (lm.loaded) resident.insert(lm.model_id);
    }
    nlohmann::json data = nlohmann::json::array();

    for (const auto& model : models) {
        bool is_resident = resident.count(model.id) > 0;
        nlohmann::json model_json;
        model_json["id"] = model.id;
        model_json["object"] = "model";
        model_json["created"] = model.created_at;
        model_json["format"] = model.format;
        model_json["quantization"] = model.quantization;
        model_json["capabilities"] = model.capabilities;
        model_json["loaded"] = is_resident;
        // llama.cpp WebUI compat (Fase 9, additive only)
        model_json["owned_by"] = "barskuy";
        model_json["in_cache"] = true;
        model_json["status"] = {{"value", is_resident ? "loaded" : "unloaded"}};
        data.push_back(model_json);
    }

    nlohmann::json response;
    response["object"] = "list";
    response["data"] = data;
    res.set_content(response.dump(), "application/json");
}

void Server::handle_model_register(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    try {
        auto json = nlohmann::json::parse(req.body);
        std::string path = json.value("path", "");
        std::string alias = json.value("alias", "");

        if (path.empty() || alias.empty()) {
            send_error(res, 400, "invalid_request", "path and alias are required", "MISSING_PARAMS");
            return;
        }

        registry::Model model;
        model.id = alias;
        model.path = path;
        // Format from extension (gguf/safetensors); image/video weights register
        // with explicit capabilities and are served by their engines.
        std::string lp = path;
        for (auto& c : lp) c = (char)std::tolower((unsigned char)c);
        if (lp.size() >= 11 && lp.compare(lp.size() - 11, 11, ".safetensors") == 0) model.format = "safetensors";
        else model.format = "gguf";
        model.architecture = "unknown";
        model.quantization = "unknown";
        // Capabilities: client-provided (whitelisted) or default ["text"].
        // Fase 8: UI gates Text/Image/Video buttons on these.
        static const std::vector<std::string> known_caps = {"text", "vision", "image", "video", "embeddings"};
        model.capabilities = {"text"};
        if (json.contains("capabilities") && json["capabilities"].is_array()) {
            std::vector<std::string> caps;
            for (auto& c : json["capabilities"]) {
                if (!c.is_string()) continue;
                std::string s = c.get<std::string>();
                if (std::find(known_caps.begin(), known_caps.end(), s) != known_caps.end()) caps.push_back(s);
            }
            if (!caps.empty()) model.capabilities = caps;
        }
        model.loaded = false;
        model.created_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (model_registry_.register_model(model)) {
            nlohmann::json response;
            response["id"] = model.id;
            response["object"] = "model";
            response["created"] = model.created_at;
            response["format"] = model.format;
            response["quantization"] = model.quantization;
            response["capabilities"] = model.capabilities;
            response["loaded"] = model.loaded;
            res.set_content(response.dump(), "application/json");
        } else {
            send_error(res, 500, "server_error", "Failed to register model", "REGISTER_FAILED");
        }
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_model_unload(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    try {
        auto json = nlohmann::json::parse(req.body);
        std::string model_id = json.value("model", json.value("id", ""));
        if (model_id.empty()) {
            send_error(res, 400, "invalid_request", "model is required", "MISSING_MODEL");
            return;
        }
        bool ok = text_engine_.unload_model(model_id);
        nlohmann::json response;
        response["id"] = model_id;
        response["unloaded"] = ok;
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_complete(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    try {
        auto json = nlohmann::json::parse(req.body);
        std::string model_id = json.value("model", "");
        bool stream = json.value("stream", false);
        // F9-10: max_tokens mengikuti ctx (default -1 = sampai EOS dalam
        // batas ctx). n_predict adalah alias ala llama.cpp (UI kirim ini).
        int max_tokens = -1;
        if (json.contains("max_tokens") && json["max_tokens"].is_number())
            max_tokens = json["max_tokens"].get<int>();
        else if (json.contains("n_predict") && json["n_predict"].is_number())
            max_tokens = json["n_predict"].get<int>();
        float temperature = json.value("temperature", 0.7f);
        float top_p = json.value("top_p", 0.9f);
        std::vector<std::string> stop = json.value("stop", std::vector<std::string>());

        // Generation options: guided decoding (F3-8), LoRA (F3-9), speculative (F3-7)
        engine::GenerationOptions gen_opts;
        // F9-10: UI kirim chat_template_kwargs.enable_thinking (menu Reasoning).
        // false = tutup prefill <think>, model jawab langsung (jauh lebih cepat).
        if (json.contains("chat_template_kwargs") && json["chat_template_kwargs"].is_object()) {
            auto& kwargs = json["chat_template_kwargs"];
            if (kwargs.contains("enable_thinking") && kwargs["enable_thinking"].is_boolean())
                gen_opts.enable_thinking = kwargs["enable_thinking"].get<bool>();
        }
        if (json.contains("enable_thinking") && json["enable_thinking"].is_boolean())
            gen_opts.enable_thinking = json["enable_thinking"].get<bool>();
        if (json.contains("response_format") && json["response_format"].is_object()) {
            std::string rf_type = json["response_format"].value("type", "text");
            if (rf_type == "json_object") {
                gen_opts.json_mode = true;
            } else if (rf_type == "json_schema" && json["response_format"].contains("json_schema")) {
                // Accept raw schema string; GBNF conversion is client-side for now
                auto& js = json["response_format"]["json_schema"];
                if (js.contains("schema") && js["schema"].is_string()) {
                    gen_opts.grammar = js["schema"].get<std::string>();
                } else {
                    gen_opts.json_mode = true;
                }
            }
        }
        if (json.contains("grammar") && json["grammar"].is_string()) {
            gen_opts.grammar = json["grammar"].get<std::string>();
            gen_opts.json_mode = false;
        }
        if (json.contains("grammar_root") && json["grammar_root"].is_string()) {
            gen_opts.grammar_root = json["grammar_root"].get<std::string>();
        }
        if (json.contains("lora_adapters") && json["lora_adapters"].is_array()) {
            for (const auto& a : json["lora_adapters"]) {
                std::string aid = a.value("id", "");
                float scale = a.value("scale", 1.0f);
                if (!aid.empty()) gen_opts.lora_adapters.emplace_back(aid, scale);
            }
        } else if (json.contains("lora_adapter") && json["lora_adapter"].is_string()) {
            gen_opts.lora_adapters.emplace_back(json["lora_adapter"].get<std::string>(), 1.0f);
        }
        if (json.contains("draft_model") && json["draft_model"].is_string()) {
            gen_opts.draft_model = json["draft_model"].get<std::string>();
            gen_opts.draft_tokens = json.value("draft_tokens", 5);
        }
        // Sampling parity with llama.cpp WebUI settings (F9-8)
        if (json.contains("top_k") && json["top_k"].is_number()) gen_opts.top_k = json["top_k"].get<int>();
        if (json.contains("min_p") && json["min_p"].is_number()) gen_opts.min_p = json["min_p"].get<float>();
        if (json.contains("repeat_last_n") && json["repeat_last_n"].is_number())
            gen_opts.repeat_last_n = json["repeat_last_n"].get<int>();
        if (json.contains("repeat_penalty") && json["repeat_penalty"].is_number())
            gen_opts.repeat_penalty = json["repeat_penalty"].get<float>();
        if (json.contains("presence_penalty") && json["presence_penalty"].is_number())
            gen_opts.presence_penalty = json["presence_penalty"].get<float>();
        if (json.contains("frequency_penalty") && json["frequency_penalty"].is_number())
            gen_opts.frequency_penalty = json["frequency_penalty"].get<float>();
        if (json.contains("seed") && json["seed"].is_number()) gen_opts.seed = json["seed"].get<int>();
        if (json.contains("mirostat") && json["mirostat"].is_number()) gen_opts.mirostat = json["mirostat"].get<int>();
        if (json.contains("mirostat_tau") && json["mirostat_tau"].is_number())
            gen_opts.mirostat_tau = json["mirostat_tau"].get<float>();
        if (json.contains("mirostat_eta") && json["mirostat_eta"].is_number())
            gen_opts.mirostat_eta = json["mirostat_eta"].get<float>();
        if (json.contains("dry_multiplier") && json["dry_multiplier"].is_number())
            gen_opts.dry_multiplier = json["dry_multiplier"].get<float>();
        if (json.contains("dry_base") && json["dry_base"].is_number())
            gen_opts.dry_base = json["dry_base"].get<float>();
        if (json.contains("dry_allowed_length") && json["dry_allowed_length"].is_number())
            gen_opts.dry_allowed_length = json["dry_allowed_length"].get<int>();
        if (json.contains("dry_penalty_last_n") && json["dry_penalty_last_n"].is_number())
            gen_opts.dry_penalty_last_n = json["dry_penalty_last_n"].get<int>();
        // Requested defaults (F9-9): apply only when the client omits the field
        if (!json.contains("temperature")) temperature = 0.6f;
        if (!json.contains("top_k")) gen_opts.top_k = 20;
        if (!json.contains("top_p")) top_p = 0.9f;
        if (!json.contains("min_p")) gen_opts.min_p = 0.05f;
        if (!json.contains("repeat_penalty")) gen_opts.repeat_penalty = 1.1f;
        bool max_tokens_explicit = (max_tokens >= 0);

        // Parse messages (F9-10: content may be null for assistant tool-call
        // turns - upstream sends {"content": null, "tool_calls": [...]})
        std::vector<std::pair<std::string, std::string>> messages;
        auto messages_json = json.value("messages", nlohmann::json::array());
        for (const auto& msg : messages_json) {
            std::string role = "user";
            if (msg.contains("role") && msg["role"].is_string())
                role = msg["role"].get<std::string>();
            std::string content;
            if (msg.contains("content") && msg["content"].is_string())
                content = msg["content"].get<std::string>();
            messages.emplace_back(role, content);
        }

        // F9-10 request entry log: proves the request ARRIVED (vs stuck client)
        core::Logger::info("chat in model={} stream={} max_tokens={} tools={}",
            model_id.empty() ? "(default)" : model_id, stream ? 1 : 0,
            max_tokens, json.contains("tools") ? "yes" : "no");
        if (model_id.empty()) {
            // llama.cpp MODEL-mode compat (Fase 9): active swap first, then the
            // single loaded model, else the single registered one.
            std::string fallback = get_active_model();
            if (fallback.empty()) {
                for (auto& lm : text_engine_.list_loaded_models()) {
                    if (lm.loaded) {
                        if (fallback.empty()) fallback = lm.model_id;
                        else { fallback.clear(); break; }
                    }
                }
            }
            if (fallback.empty()) {
                auto all = model_registry_.list_models();
                if (all.size() == 1) fallback = all[0].id;
            }
            if (fallback.empty()) {
                send_error(res, 400, "invalid_request", "model is required", "MISSING_MODEL");
                return;
            }
            model_id = fallback;
        }

        // Ensure model is loaded
        if (!text_engine_.is_loaded(model_id)) {
            // Try to load from registry
            core::Logger::info("chat load model={}", model_id);
            auto model = model_registry_.get_model(model_id);
            if (model) {
                if (!text_engine_.load_model(model_id, model->path)) {
                    send_error(res, 500, "server_error", "Failed to load model", "MODEL_LOAD_FAILED");
                    return;
                }
                core::Logger::info("chat loaded model={}", model_id);
            } else {
                send_error(res, 404, "not_found", "Model not found: " + model_id, "MODEL_NOT_FOUND");
                return;
            }
        }

        // Ensure draft model is loaded for speculative decoding (F3-7)
        if (!gen_opts.draft_model.empty() && !text_engine_.is_loaded(gen_opts.draft_model)) {
            auto dmodel = model_registry_.get_model(gen_opts.draft_model);
            if (dmodel) {
                if (!text_engine_.load_model(gen_opts.draft_model, dmodel->path)) {
                    send_error(res, 500, "server_error", "Failed to load draft model", "MODEL_LOAD_FAILED");
                    return;
                }
            } else {
                send_error(res, 404, "not_found", "Draft model not found: " + gen_opts.draft_model, "MODEL_NOT_FOUND");
                return;
            }
        }

        // F9-10: max_tokens mengikuti ctx model. -1/omitted = sampai EOS
        // dalam batas ctx; nilai eksplisit di-clamp ke n_ctx (lebih dari
        // itu mustahil: posisi KV tak muat).
        {
            int n_ctx = 8192;
            for (auto& lm : text_engine_.list_loaded_models()) {
                if (lm.model_id == model_id && lm.loaded && lm.config.n_ctx > 0) {
                    n_ctx = lm.config.n_ctx;
                    break;
                }
            }
            if (!max_tokens_explicit || max_tokens < 0) {
                max_tokens = n_ctx;
            } else if (max_tokens > n_ctx) {
                core::Logger::info("max_tokens {} di-clamp ke n_ctx {}", max_tokens, n_ctx);
                max_tokens = n_ctx;
            }
        }

        // Hot-swap may have evicted the target while loading the draft (F1-8).
        // Speculative needs both resident -> requires budget >= 2.
        if (!gen_opts.draft_model.empty() && gen_opts.draft_model != model_id
            && !text_engine_.is_loaded(model_id)) {
            send_error(res, 400, "invalid_request",
                "Speculative decoding needs target + draft resident: raise --max-loaded-models to >= 2",
                "SPECULATIVE_NEEDS_BUDGET");
            return;
        }

        // F7 agent loop (explicit agent:true): server executes our tools.
        // F9-9 passthrough (upstream llama-server compat): request carries
        // tools but no agent flag -> single step; tool_calls return to the
        // client (UI executes browser/MCP tools itself and loops).
        // Plain requests keep the exact standard path.
        std::string tool_choice = "auto";
        if (json.contains("tool_choice")) {
            if (json["tool_choice"].is_string()) tool_choice = json["tool_choice"].get<std::string>();
            else tool_choice = "auto";
        }
        bool has_tools = json.contains("tools") && json["tools"].is_array() && !json["tools"].empty();
        bool want_agent = config_.agent_enabled() && !stream && json.value("agent", false);
        bool want_passthrough = !stream && has_tools && tool_choice != "none" && !json.value("agent", false);
        // Streaming with tools also goes passthrough (SSE tool_calls chunks).
        if (stream && has_tools && tool_choice != "none" && !config_.tools_mode().empty())
            want_passthrough = true;
        if (want_passthrough && config_.tools_mode().empty()) {
            send_error(res, 403, "server_error", "tools are disabled (enable with --tools all)", "TOOLS_DISABLED");
            return;
        }
        if (want_passthrough) {
            // All-server-tools selection behaves like the agent loop's first
            // step but returns calls instead of executing (client decides).
            // Exception: explicit agent loop only via agent:true (above).
            metric_complete_requests_++;
            nlohmann::json oa_tools = json["tools"];
            if (stream) {
                // F9-10 live passthrough: render prompt+tools, stream mentah
                // (think/content deltas + timings progresif, seperti chat),
                // parse calls di akhir. Ganti pola lama blocking-lalu-dump
                // yang bikin UI diam tanpa output sampai selesai.
                metric_complete_requests_++;
                std::string step_prompt, step_err;
                if (!agent_runner_.render_step(model_id, messages_json, oa_tools,
                                              step_prompt, step_err)) {
                    send_error(res, 500, "server_error", step_err, "AGENT_FAILED");
                    return;
                }
                std::string live_id = "cmpl_" + core::utils::random_id();
                int64_t live_created = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                auto queue = std::make_shared<SseQueue>();
                auto splitter = std::make_shared<ThinkSplitter>();
                auto t_prog0 = std::make_shared<std::chrono::steady_clock::time_point>(
                    std::chrono::steady_clock::now());
                auto n_prog = std::make_shared<int>(0);
                auto emit_delta = [queue, live_id, model_id, live_created, t_prog0, n_prog](
                        const std::string& content, const std::string& reasoning) {
                    std::string r = is_blank(reasoning) ? std::string() : reasoning;
                    if (content.empty() && r.empty()) return;
                    nlohmann::json delta;
                    if (!content.empty()) delta["content"] = content;
                    if (!r.empty()) delta["reasoning_content"] = r;
                    nlohmann::json chunk;
                    chunk["id"] = live_id;
                    chunk["object"] = "chat.completion.chunk";
                    chunk["created"] = live_created;
                    chunk["model"] = model_id;
                    chunk["choices"] = nlohmann::json::array({{
                        {"index", 0}, {"delta", delta}, {"finish_reason", nullptr}
                    }});
                    int k = ++(*n_prog);
                    if (k == 1) *t_prog0 = std::chrono::steady_clock::now();
                    if (k % 5 == 0) {
                        double ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - *t_prog0).count();
                        chunk["timings"] = timings_json(0, 0.0, k, ms);
                    }
                    queue->push("data: " + chunk.dump() + "\n\n");
                };
                int step_budget = max_tokens > 0 && max_tokens < 1024 ? max_tokens : 1024;
                std::thread([this, queue, splitter, emit_delta,
                             model_id, step_prompt, step_budget, temperature, top_p, stop,
                             messages_json, oa_tools, live_id, live_created]() {
                    std::string raw_acc, content_buf;
                    engine::TextEngine::Timings tm = text_engine_.complete_stream_raw(
                        model_id, step_prompt, step_budget, temperature, top_p, stop,
                        [&](const std::string& token, bool is_final) {
                            if (is_final) return;
                            if (token.empty()) return;
                            raw_acc += token;
                            auto parts = splitter->push(token);
                            // BUG-049: reasoning live; konten di-buffer dulu.
                            // XML tool_call sebagai konten + tool_calls dobel
                            // tampil di UI -> putuskan di akhir (ada calls =
                            // konten dibuang seperti non-stream).
                            emit_delta("", parts.reasoning);
                            content_buf += parts.content;
                        },
                        {},
                        [&](int pn, double pms, bool pre) {
                            // BUG-049: prompt prefill <think> -> mode think
                            splitter->in_think = splitter->in_think || pre;
                            nlohmann::json chunk;
                            chunk["id"] = live_id;
                            chunk["object"] = "chat.completion.chunk";
                            chunk["created"] = live_created;
                            chunk["model"] = model_id;
                            chunk["choices"] = nlohmann::json::array({{
                                {"index", 0}, {"delta", nlohmann::json::object()},
                                {"finish_reason", nullptr}
                            }});
                            chunk["timings"] = timings_json(pn, pms, 0, 0.0);
                            queue->push("data: " + chunk.dump() + "\n\n");
                        });
                    auto tail = splitter->flush();
                    emit_delta("", tail.reasoning);
                    content_buf += tail.content;
                    std::string text;
                    std::vector<agent::ToolCall> calls;
                    agent_runner_.parse_step_text(model_id, messages_json, oa_tools,
                                                 raw_acc, text, calls);
                    // BUG-051: coba perbaiki sekali bila args kosong (blocking
                    // pendek; kasus normal tanpa biaya karena langsung false).
                    {
                        std::string rt = text;
                        std::vector<agent::ToolCall> rc = calls;
                        if (agent_runner_.repair_calls(model_id, messages_json, oa_tools,
                                                      step_prompt, raw_acc, rt, rc)) {
                            text = rt;
                            calls = rc;
                        }
                    }
                    if (calls.empty()) {
                        // tanpa calls: tampilkan teks (bersih dari sisa XML)
                        std::string show = engine::strip_tool_markup(content_buf);
                        if (!show.empty()) emit_delta(show, "");
                    }
                    metric_prompt_tokens_ += static_cast<uint64_t>(tm.prompt_n);
                    metric_completion_tokens_ += static_cast<uint64_t>(tm.predicted_n);
                    {
                        std::string names;
                        for (auto& c : calls) {
                            if (!names.empty()) names += ",";
                            names += c.name;
                        }
                        core::Logger::info("agent-step model={} calls={}",
                            model_id, names.empty() ? "-" : names);
                    }
                    for (size_t i = 0; i < calls.size(); ++i) {
                        nlohmann::json args = calls[i].arguments;
                        try { args = nlohmann::json::parse(calls[i].arguments); } catch (...) {}
                        std::string args_str = args.is_string() ? args.get<std::string>() : args.dump();
                        nlohmann::json chunk;
                        chunk["id"] = live_id;
                        chunk["object"] = "chat.completion.chunk";
                        chunk["created"] = live_created;
                        chunk["model"] = model_id;
                        chunk["choices"] = nlohmann::json::array({{
                            {"index", 0},
                            {"delta", {{"tool_calls", nlohmann::json::array({{
                                {"index", (int)i},
                                {"id", calls[i].id.empty() ? ("call_" + std::to_string(i)) : calls[i].id},
                                {"type", "function"},
                                {"function", {{"name", calls[i].name}, {"arguments", args_str}}}
                            }})}}},
                            {"finish_reason", nullptr}
                        }});
                        queue->push("data: " + chunk.dump() + "\n\n");
                    }
                    nlohmann::json fin;
                    fin["id"] = live_id;
                    fin["object"] = "chat.completion.chunk";
                    fin["created"] = live_created;
                    fin["model"] = model_id;
                    fin["choices"] = nlohmann::json::array({{
                        {"index", 0}, {"delta", nlohmann::json::object()},
                        {"finish_reason", calls.empty() ? "stop" : "tool_calls"}
                    }});
                    fin["timings"] = timings_json(tm.prompt_n, tm.prompt_ms,
                        tm.predicted_n, tm.predicted_ms);
                    queue->push("data: " + fin.dump() + "\n\n");
                    queue->push("data: [DONE]\n\n");
                    queue->finish();
                    log_perf(model_id, tm.prompt_n, tm.prompt_ms,
                             tm.predicted_n, tm.predicted_ms);
                }).detach();
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [queue](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                        std::string line;
                        for (int i = 0; i < 14400; ++i) { // ~10 min cap
                            if (queue->pop(line, 25)) {
                                if (!sink.is_writable() || !sink.write(line.data(), line.size())) {
                                    sink.done();
                                    return false;
                                }
                                return true;
                            }
                            if (!sink.is_writable()) { sink.done(); return false; }
                            if (queue->finished()) { sink.done(); return false; }
                        }
                        sink.done();
                        return false;
                    });
                return;
            }
            agent::SingleStep st = agent_runner_.step(
                model_id, messages_json, oa_tools, max_tokens, temperature, top_p);
            metric_prompt_tokens_ += static_cast<uint64_t>(st.prompt_tokens);
            metric_completion_tokens_ += static_cast<uint64_t>(st.completion_tokens);
            if (st.error) {
                send_error(res, 500, "server_error", st.text, "AGENT_FAILED");
                return;
            }
            {
                std::string names;
                for (auto& c : st.calls) {
                    if (!names.empty()) names += ",";
                    names += c.name;
                }
                core::Logger::info("agent-step model={} calls={}",
                    model_id, names.empty() ? "-" : names);
            }
            log_perf(model_id, st.prompt_tokens, st.prompt_ms,
                     st.completion_tokens, st.predicted_ms);
            std::string resp_id = "cmpl_" + core::utils::random_id();
            int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (!stream) {
                nlohmann::json response;
                response["id"] = resp_id;
                response["object"] = "chat.completion";
                response["created"] = created;
                response["model"] = model_id;
                // Strip tool-call markup + echoed headers; empty content with
                // calls becomes null (upstream-exact OpenAI shape, F9-10).
                std::string clean_text = engine::strip_tool_markup(st.text);
                auto psplit = split_thinking(clean_text);
                nlohmann::json msg = {{"role", "assistant"}};
                if (psplit.first.empty() && !st.calls.empty()) msg["content"] = nullptr;
                else msg["content"] = psplit.first;
                if (!is_blank(psplit.second)) msg["reasoning_content"] =
                    psplit.second;
                nlohmann::json calls = nlohmann::json::array();
                for (size_t i = 0; i < st.calls.size(); ++i) {
                    nlohmann::json args = st.calls[i].arguments;
                    try { args = nlohmann::json::parse(st.calls[i].arguments); } catch (...) {}
                    std::string args_str = args.is_string() ? args.get<std::string>() : args.dump();
                    calls.push_back({{"id", st.calls[i].id.empty() ? ("call_" + std::to_string(i)) : st.calls[i].id},
                        {"type", "function"},
                        {"function", {{"name", st.calls[i].name}, {"arguments", args_str}}}});
                }
                if (!calls.empty()) msg["tool_calls"] = calls;
                nlohmann::json choice;
                choice["index"] = 0;
                choice["message"] = msg;
                choice["finish_reason"] = calls.empty() ? "stop" : "tool_calls";
                response["choices"] = nlohmann::json::array({choice});
                nlohmann::json usage;
                usage["prompt_tokens"] = st.prompt_tokens;
                usage["completion_tokens"] = st.completion_tokens;
                usage["total_tokens"] = st.prompt_tokens + st.completion_tokens;
                response["usage"] = usage;
                response["timings"] = timings_json(st.prompt_tokens, st.prompt_ms,
                    st.completion_tokens, st.predicted_ms);
                res.set_content(response.dump(), "application/json");
                return;
            }
        }
        if (want_agent) {
            core::Logger::info("agent begin model={}", model_id);
            metric_complete_requests_++;
            nlohmann::json oa_tools = json.contains("tools") ? json["tools"] : nlohmann::json(nullptr);
            agent::AgentResult ar = agent_runner_.run(
                model_id, messages_json, oa_tools, max_tokens, temperature, top_p,
                req.get_header_value("x-tool-cwd"));
            metric_prompt_tokens_ += static_cast<uint64_t>(ar.prompt_tokens);
            metric_completion_tokens_ += static_cast<uint64_t>(ar.completion_tokens);
            metric_agent_steps_ += static_cast<uint64_t>(ar.steps.size());
            if (ar.finish_reason == "error") {
                send_error(res, 500, "server_error", ar.text, "AGENT_FAILED");
                return;
            }
            nlohmann::json response;
            response["id"] = "cmpl_" + core::utils::random_id();
            response["object"] = "text_completion";
            response["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            response["model"] = model_id;
            auto asplit = split_thinking(engine::strip_tool_markup(ar.text));
            nlohmann::json choices = nlohmann::json::array();
            nlohmann::json choice;
            choice["index"] = 0;
            nlohmann::json amsg = {{"role", "assistant"}, {"content", asplit.first}};
            if (!is_blank(asplit.second)) amsg["reasoning_content"] =
                asplit.second;
            choice["message"] = amsg;
            choice["finish_reason"] = ar.finish_reason;
            choices.push_back(choice);
            response["choices"] = choices;
            nlohmann::json usage;
            usage["prompt_tokens"] = ar.prompt_tokens;
            usage["completion_tokens"] = ar.completion_tokens;
            usage["total_tokens"] = ar.prompt_tokens + ar.completion_tokens;
            response["usage"] = usage;
            response["timings"] = timings_json(ar.prompt_tokens, ar.prompt_ms,
                ar.completion_tokens, ar.predicted_ms);
            log_perf(model_id, ar.prompt_tokens, ar.prompt_ms,
                     ar.completion_tokens, ar.predicted_ms);
            nlohmann::json steps = nlohmann::json::array();
            for (auto& s : ar.steps) {
                nlohmann::json st;
                try { st["arguments"] = nlohmann::json::parse(s.args); }
                catch (...) { st["arguments"] = s.args; }
                steps.push_back({{"tool", s.tool}, {"arguments", st["arguments"]}, {"result", s.result}});
            }
            response["agent_steps"] = steps;
            res.set_content(response.dump(), "application/json");
            return;
        }

        if (stream) {
            // F9-10 true streaming: generate in a worker thread, emit SSE
            // lines (content + reasoning_content deltas) as tokens arrive.
            core::Logger::info("chat start model={} stream=1 max_tokens={} tools={}",
                model_id, max_tokens, has_tools ? "yes" : "no");
            metric_complete_requests_++;
            std::string resp_id = "cmpl_" + core::utils::random_id();
            int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            auto queue = std::make_shared<SseQueue>();
            auto splitter = std::make_shared<ThinkSplitter>();
            // F9-10 realtime stats: UI baca chunk.timings tiap delta untuk
            // footer tok/s + gauge konteks live (sebelumnya hanya di fin).
            auto t_prog0 = std::make_shared<std::chrono::steady_clock::time_point>(
                std::chrono::steady_clock::now());
            auto n_prog = std::make_shared<int>(0);
            auto emit_delta = [queue, resp_id, model_id, created, t_prog0, n_prog](
                    const std::string& content, const std::string& reasoning) {
                // BUG-046: drop whitespace-only reasoning pieces (empty think);
                // otherwise the UI shows an empty "Reasoning" block.
                std::string r = is_blank(reasoning) ? std::string() : reasoning;
                if (content.empty() && r.empty()) return;
                nlohmann::json delta;
                if (!content.empty()) delta["content"] = content;
                if (!r.empty()) delta["reasoning_content"] = r;
                nlohmann::json chunk;
                chunk["id"] = resp_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = created;
                chunk["model"] = model_id;
                chunk["choices"] = nlohmann::json::array({{
                    {"index", 0}, {"delta", delta}, {"finish_reason", nullptr}
                }});
                int k = ++(*n_prog);
                if (k == 1) *t_prog0 = std::chrono::steady_clock::now();
                if (k % 5 == 0) {
                    double ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - *t_prog0).count();
                    chunk["timings"] = timings_json(0, 0.0, k, ms);
                }
                queue->push("data: " + chunk.dump() + "\n\n");
            };
            std::thread([this, queue, splitter, emit_delta,
                         model_id, messages, max_tokens, temperature, top_p, stop, gen_opts,
                         resp_id, created]() {
                engine::TextEngine::Timings tm = text_engine_.complete_stream(
                    model_id, messages, max_tokens, temperature, top_p, stop,
                    [&](const std::string& token, bool is_final) {
                        if (is_final) return;
                        if (token.empty()) return;
                        auto parts = splitter->push(token);
                        emit_delta(parts.content, parts.reasoning);
                    },
                    gen_opts,
                    [&](int pn, double pms, bool pre) {
                        // BUG-049: prompt prefill <think> -> output mulai mode think
                        splitter->in_think = splitter->in_think || pre;
                        nlohmann::json chunk;
                        chunk["id"] = resp_id;
                        chunk["object"] = "chat.completion.chunk";
                        chunk["created"] = created;
                        chunk["model"] = model_id;
                        chunk["choices"] = nlohmann::json::array({{
                            {"index", 0}, {"delta", nlohmann::json::object()},
                            {"finish_reason", nullptr}
                        }});
                        chunk["timings"] = timings_json(pn, pms, 0, 0.0);
                        queue->push("data: " + chunk.dump() + "\n\n");
                    });
                auto tail = splitter->flush();
                emit_delta(tail.content, tail.reasoning);
                metric_prompt_tokens_ += static_cast<uint64_t>(tm.prompt_n);
                metric_completion_tokens_ += static_cast<uint64_t>(tm.predicted_n);
                // Same numbers go to the final chunk (WebUI) and the log (F9-10 sync)
                nlohmann::json fin;
                fin["id"] = resp_id;
                fin["object"] = "chat.completion.chunk";
                fin["created"] = created;
                fin["model"] = model_id;
                fin["choices"] = nlohmann::json::array({{
                    {"index", 0}, {"delta", nlohmann::json::object()},
                    {"finish_reason", "stop"}
                }});
                fin["timings"] = timings_json(
                    tm.prompt_n, tm.prompt_ms, tm.predicted_n, tm.predicted_ms);
                queue->push("data: " + fin.dump() + "\n\n");
                queue->push("data: [DONE]\n\n");
                queue->finish();
                log_perf(model_id, tm.prompt_n, tm.prompt_ms, tm.predicted_n, tm.predicted_ms);
            }).detach();
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [queue](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                    std::string line;
                    // End ONLY when the worker finished AND the queue drained;
                    // slow prefills (big CPU models) just wait longer here.
                    // NOTE: sink.done() writes the chunked terminator; a bare
                    // false aborts the body (client sees truncated stream).
                    // A dead peer ends the stream early instead of spinning.
                    for (int i = 0; i < 14400; ++i) { // ~10 min cap
                        if (queue->pop(line, 25)) {
                            if (!sink.is_writable() || !sink.write(line.data(), line.size())) {
                                sink.done();
                                return false;
                            }
                            return true;
                        }
                        if (!sink.is_writable()) { sink.done(); return false; }
                        if (queue->finished()) { sink.done(); return false; }
                    }
                    sink.done();
                    return false;
                });
            return;
        } else {
            // Non-streaming completion
            core::Logger::info("chat start model={} stream=0 max_tokens={} tools={}",
                model_id, max_tokens, has_tools ? "yes" : "no");
            metric_complete_requests_++;
            auto result = text_engine_.complete(model_id, messages, max_tokens, temperature, top_p, stop, gen_opts);
            metric_prompt_tokens_ += static_cast<uint64_t>(result.prompt_tokens);
            metric_completion_tokens_ += static_cast<uint64_t>(result.completion_tokens);
            log_perf(model_id, result.prompt_tokens, result.prompt_ms,
                     result.completion_tokens, result.predicted_ms);

            auto split = split_thinking(result.text);
            nlohmann::json response;
            response["id"] = "cmpl_" + core::utils::random_id();
            response["object"] = "text_completion";
            response["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            response["model"] = model_id;

            nlohmann::json choices = nlohmann::json::array();
            nlohmann::json choice;
            choice["index"] = 0;
            nlohmann::json msg = {{"role", "assistant"}, {"content", split.first}};
            if (!is_blank(split.second)) msg["reasoning_content"] =
                split.second;
            choice["message"] = msg;
            choice["finish_reason"] = result.finish_reason;
            choices.push_back(choice);
            response["choices"] = choices;

            nlohmann::json usage;
            usage["prompt_tokens"] = result.prompt_tokens;
            usage["completion_tokens"] = result.completion_tokens;
            usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
            response["usage"] = usage;
            response["timings"] = timings_json(result.prompt_tokens, result.prompt_ms,
                result.completion_tokens, result.predicted_ms);

            res.set_content(response.dump(), "application/json");
        }
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_embeddings(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    try {
        auto json = nlohmann::json::parse(req.body);
        std::string model_id = json.value("model", "");
        auto input = json.value("input", nlohmann::json::array());

        if (model_id.empty()) {
            send_error(res, 400, "invalid_request", "model is required", "MISSING_MODEL");
            return;
        }

        // Ensure model is loaded
        if (!text_engine_.is_loaded(model_id)) {
            auto model = model_registry_.get_model(model_id);
            if (model) {
                if (!text_engine_.load_model(model_id, model->path)) {
                    send_error(res, 500, "server_error", "Failed to load model", "MODEL_LOAD_FAILED");
                    return;
                }
            } else {
                send_error(res, 404, "not_found", "Model not found: " + model_id, "MODEL_NOT_FOUND");
                return;
            }
        }

        std::vector<std::string> inputs;
        if (input.is_string()) {
            inputs.push_back(input.get<std::string>());
        } else if (input.is_array()) {
            for (const auto& item : input) {
                if (item.is_string()) {
                    inputs.push_back(item.get<std::string>());
                }
            }
        }

        metric_embeddings_requests_++;
        auto embeddings = text_engine_.embed(model_id, inputs);
        for (const auto& emb : embeddings) {
            metric_prompt_tokens_ += static_cast<uint64_t>(emb.size());
        }

        nlohmann::json data = nlohmann::json::array();
        for (size_t i = 0; i < embeddings.size(); ++i) {
            nlohmann::json embedding_json;
            embedding_json["object"] = "embedding";
            embedding_json["index"] = static_cast<int>(i);
            embedding_json["embedding"] = embeddings[i];
            data.push_back(embedding_json);
        }

        nlohmann::json response;
        response["object"] = "list";
        response["data"] = data;
        response["model"] = model_id;
        
        nlohmann::json usage;
        int total_tokens = 0;
        for (const auto& emb : embeddings) {
            total_tokens += emb.size();
        }
        usage["prompt_tokens"] = total_tokens;
        usage["total_tokens"] = total_tokens;
        response["usage"] = usage;
        
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_generate_image(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    try {
        auto json = nlohmann::json::parse(req.body);
        
        // Extract parameters (defaults from --image-* flags, override-able per-request)
        engine::ImageGenerationParams params;
        params.model_id = json.value("model", config_.image_model());
        params.prompt = json.value("prompt", "");
        params.negative_prompt = json.value("negative_prompt", "");
        params.width = json.value("width", config_.image_width());
        params.height = json.value("height", config_.image_height());
        params.steps = json.value("steps", config_.image_steps());
        params.guidance_scale = json.value("guidance_scale", config_.image_cfg());
        params.seed = json.value("seed", -1);
        params.batch_size = json.value("n", 1);
        
        std::string scheduler_str = json.value("scheduler", "ddim");
        if (scheduler_str == "euler") params.scheduler = engine::ImageGenerationParams::SchedulerType::Euler;
        else if (scheduler_str == "euler_a") params.scheduler = engine::ImageGenerationParams::SchedulerType::EulerAncestral;
        else if (scheduler_str == "dpmpp_2m") params.scheduler = engine::ImageGenerationParams::SchedulerType::DPMpp2M;
        else params.scheduler = engine::ImageGenerationParams::SchedulerType::DDIM;

        bool wait = json.value("wait", false);

        metric_image_requests_++;

        if (params.model_id.empty()) {
            send_error(res, 400, "invalid_request", "model is required", "MISSING_MODEL");
            return;
        }
        if (params.prompt.empty()) {
            send_error(res, 400, "invalid_request", "prompt is required", "MISSING_PROMPT");
            return;
        }
        // Ensure image model known (registry -> engine path; sd lazy-loads).
        // Toleran hyphen/underscore (z-image-turbo vs z_image_turbo) + fallback
        // ke image model pertama jika id tak ditemukan — cegah 404 di UI.
        std::optional<registry::Model> img_model;
        if (auto m = model_registry_.get_model(params.model_id)) img_model = m;
        else {
            std::string alt = params.model_id;
            for (auto& c : alt) if (c == '_') c = '-'; else if (c == '-') c = '_';
            if (auto m = model_registry_.get_model(alt)) img_model = m;
            else {
                for (auto& mm : model_registry_.list_models()) {
                    for (auto& cap : mm.capabilities) if (cap == "image") { img_model = model_registry_.get_model(mm.id); break; }
                    if (img_model) break;
                }
            }
        }
        if (!image_engine_.is_loaded(params.model_id)) {
            if (!img_model || !image_engine_.register_model(img_model->id, img_model->path)) {
                send_error(res, 404, "not_found", "Model not found: " + params.model_id, "MODEL_NOT_FOUND");
                return;
            }
            params.model_id = img_model->id; // pakai id yang ter-resolve
        }
        // Lazy VRAM handoff (Fase 10): teks (4.4GB) + difusi (3.7GB) tak muat
        // bareng di 6GB. Unload model teks dulu; chat berikutnya reload lazy
        // otomatis. Tanpa ini generate macet di tengah jalan (fragmentasi/OOM).
        for (auto& lm : text_engine_.list_loaded_models()) {
            if (lm.loaded) {
                core::Logger::info("image gen: unload text model '{}' for VRAM", lm.model_id);
                text_engine_.unload_model(lm.model_id);
            }
        }

        if (wait) {
            // Synchronous generation
            auto result = image_engine_.generate(params);

            nlohmann::json response;
            response["id"] = "img_" + core::utils::random_id();
            response["object"] = "image.generation.job";
            response["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            auto urls = save_preview_frames(response["id"].get<std::string>(),
                result.images, result.widths, result.heights);
            nlohmann::json data_array = nlohmann::json::array();
            for (auto& u : urls) data_array.push_back({{"url", u}});
            if (urls.empty()) {
                response["status"] = "failed";
                response["error"] = "engine produced no output";
            } else {
                response["status"] = "completed";
                nlohmann::json result_obj;
                result_obj["data"] = data_array;
                result_obj["preview"] = true;
                response["result"] = result_obj;
            }
            res.set_content(response.dump(), "application/json");
        } else {
            // Async job-based generation
            std::string job_id = "img_" + core::utils::random_id();
            int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            registry::GenerationJob job;
            job.id = job_id;
            job.type = "image";
            job.model_id = params.model_id;
            job.status = "queued";
            job.progress = 0;
            job.request_params = json;
            job.created_at = now;

            if (model_registry_.create_job(job)) {
                // Use ImageEngine async generation - capture params by value
                auto params_copy = params;
                image_engine_.generate_async(params_copy,
                    [this, job_id](const engine::ImageGenerationResult& result, bool is_final) -> void {
                        if (is_final) {
                            auto urls = save_preview_frames(job_id,
                                result.images, result.widths, result.heights);
                            if (urls.empty()) {
                                model_registry_.fail_job(job_id, "engine produced no output");
                            } else {
                                model_registry_.complete_job(job_id, urls[0]);
                            }
                        } else {
                            model_registry_.update_job_status(job_id, "processing", 50);
                        }
                    });

                nlohmann::json response;
                response["id"] = job_id;
                response["object"] = "image.generation.job";
                response["status"] = "queued";
                response["created"] = now;
                res.set_content(response.dump(), "application/json");
            } else {
                send_error(res, 500, "server_error", "Failed to create job", "JOB_CREATE_FAILED");
            }
        }
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_generate_image_status(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    std::string job_id = req.matches[1];
    auto job = model_registry_.get_job(job_id);

    if (!job) {
        send_error(res, 404, "not_found", "Job not found", "JOB_NOT_FOUND");
        return;
    }

    nlohmann::json response;
    response["id"] = job->id;
    response["object"] = "image.generation.job";
    response["status"] = job->status;
    response["progress"] = job->progress;
    if (job->result_path) {
        nlohmann::json result;
        nlohmann::json data_array = nlohmann::json::array();
        data_array.push_back({{"url", *job->result_path}});
        result["data"] = data_array;
        response["result"] = result;
    } else {
        response["result"] = nullptr;
    }
    response["error"] = job->error.value_or("");
    res.set_content(response.dump(), "application/json");
}

void Server::handle_generate_video(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    try {
        auto json = nlohmann::json::parse(req.body);
        
        // Extract parameters
        engine::VideoGenerationParams params;
        params.model_id = json.value("model", "");
        params.prompt = json.value("prompt", "");
        params.negative_prompt = json.value("negative_prompt", "");
        params.num_frames = json.value("num_frames", 16);
        params.fps = json.value("fps", 8);
        params.width = json.value("width", 576);
        params.height = json.value("height", 320);
        params.steps = json.value("steps", 25);
        params.guidance_scale = json.value("guidance_scale", 7.0f);
        params.seed = json.value("seed", -1);
        
        std::string scheduler_str = json.value("scheduler", "ddim");
        if (scheduler_str == "euler") params.scheduler = engine::VideoGenerationParams::SchedulerType::Euler;
        else if (scheduler_str == "euler_a") params.scheduler = engine::VideoGenerationParams::SchedulerType::EulerAncestral;
        else if (scheduler_str == "dpmpp_2m") params.scheduler = engine::VideoGenerationParams::SchedulerType::DPMpp2M;
        else params.scheduler = engine::VideoGenerationParams::SchedulerType::DDIM;

        bool wait = json.value("wait", false);

        metric_video_requests_++;

        if (params.model_id.empty()) {
            send_error(res, 400, "invalid_request", "model is required", "MISSING_MODEL");
            return;
        }
        if (params.prompt.empty()) {
            send_error(res, 400, "invalid_request", "prompt is required", "MISSING_PROMPT");
            return;
        }

        // Validate frame count
        if (params.num_frames <= 0 || params.num_frames > 64) {
            send_error(res, 400, "invalid_request", "num_frames must be between 1 and 64", "INVALID_NUM_FRAMES");
            return;
        }

        // Estimate generation time
        int eta_seconds = video_engine_.estimate_generation_time(params);
        
        if (wait) {
            // Synchronous generation
            auto result = video_engine_.generate(params);
            
            nlohmann::json response;
            response["id"] = "vid_" + core::utils::random_id();
            response["object"] = "video.generation.job";
            response["status"] = "completed";
            response["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            response["eta_seconds"] = 0;
            
            auto urls = save_preview_frames(response["id"].get<std::string>(),
                result.frames, result.frame_widths, result.frame_heights);
            nlohmann::json data_array = nlohmann::json::array();
            for (auto& u : urls) data_array.push_back({{"url", u}});
            if (urls.empty()) {
                response["status"] = "failed";
                response["error"] = "engine produced no output";
            } else {
                nlohmann::json result_obj;
                result_obj["data"] = data_array;
                result_obj["preview"] = true;
                result_obj["fps"] = result.fps;
                response["result"] = result_obj;
            }
            response["eta_seconds"] = 0;
            res.set_content(response.dump(), "application/json");
        } else {
            // Async job-based generation
            std::string job_id = "vid_" + core::utils::random_id();
            int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            registry::GenerationJob job;
            job.id = job_id;
            job.type = "video";
            job.model_id = params.model_id;
            job.status = "queued";
            job.progress = 0;
            job.request_params = json;
            job.created_at = now;

            if (model_registry_.create_job(job)) {
                // Use VideoEngine async generation
                auto params_copy = params;
                video_engine_.generate_async(params_copy, [this, job_id](const engine::VideoGenerationResult& result, bool is_final) -> void {
                    if (is_final) {
                        auto urls = save_preview_frames(job_id,
                            result.frames, result.frame_widths, result.frame_heights);
                        if (urls.empty()) {
                            model_registry_.fail_job(job_id, "engine produced no output");
                        } else {
                            model_registry_.complete_job(job_id, urls[0]);
                        }
                    } else {
                        model_registry_.update_job_status(job_id, "processing", 50);
                    }
                });

                nlohmann::json response;
                response["id"] = job_id;
                response["object"] = "video.generation.job";
                response["status"] = "queued";
                response["created"] = now;
                response["eta_seconds"] = video_engine_.estimate_generation_time(params);
                res.set_content(response.dump(), "application/json");
            } else {
                send_error(res, 500, "server_error", "Failed to create job", "JOB_CREATE_FAILED");
            }
        }
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_generate_video_status(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    std::string job_id = req.matches[1];
    auto job = model_registry_.get_job(job_id);

    if (!job) {
        send_error(res, 404, "not_found", "Job not found", "JOB_NOT_FOUND");
        return;
    }

    nlohmann::json response;
    response["id"] = job->id;
    response["object"] = "video.generation.job";
    response["status"] = job->status;
    response["progress"] = job->progress;
    // Collect preview frames <job_id>_f*.bmp (Fase 8) plus legacy single url.
    std::vector<std::string> frame_urls;
    {
        std::error_code ec;
        for (auto& e : std::filesystem::directory_iterator("./files", ec)) {
            if (ec) break;
            std::string name = e.path().filename().string();
            if (name.compare(0, job->id.size(), job->id) == 0 &&
                name.size() > job->id.size() + 6 &&
                name.substr(name.size() - 4) == ".bmp") {
                frame_urls.push_back("/files/" + name);
            }
        }
        std::sort(frame_urls.begin(), frame_urls.end());
    }
    if (!frame_urls.empty() || job->result_path) {
        nlohmann::json result;
        nlohmann::json data_array = nlohmann::json::array();
        if (!frame_urls.empty()) {
            for (auto& u : frame_urls) data_array.push_back({{"url", u}});
            result["preview"] = true;
        } else {
            data_array.push_back({{"url", *job->result_path}});
        }
        result["data"] = data_array;
        response["result"] = result;
    } else {
        response["result"] = nullptr;
    }
    response["error"] = job->error.value_or("");
    response["eta_seconds"] = (job->status == "processing" ? 30 : 0);
    res.set_content(response.dump(), "application/json");
}

void Server::handle_jobs_list(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    std::optional<std::string> status;
    std::optional<std::string> type;

    if (req.has_param("status")) status = req.get_param_value("status");
    if (req.has_param("type")) type = req.get_param_value("type");

    auto jobs = model_registry_.list_jobs(status, type);

    nlohmann::json data = nlohmann::json::array();
    for (const auto& job : jobs) {
        nlohmann::json job_json;
        job_json["id"] = job.id;
        job_json["type"] = job.type;
        job_json["model_id"] = job.model_id;
        job_json["status"] = job.status;
        job_json["progress"] = job.progress;
        job_json["request_params"] = job.request_params;
        job_json["result_path"] = job.result_path.value_or("");
        job_json["error"] = job.error.value_or("");
        job_json["created_at"] = job.created_at;
        job_json["completed_at"] = job.completed_at.value_or(0);
        data.push_back(job_json);
    }

    nlohmann::json response;
    response["object"] = "list";
    response["data"] = data;
    res.set_content(response.dump(), "application/json");
}

void Server::handle_api_keys(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }

    try {
        auto json = nlohmann::json::parse(req.body);
        std::string label = json.value("label", "");

        std::string new_key = "sk-" + core::utils::random_id(32);
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(new_key.c_str()), new_key.length(), hash);

        std::stringstream ss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
        }
        std::string key_hash = ss.str();

        registry::ApiKey key;
        key.id = "key_" + core::utils::random_id();
        key.key_hash = key_hash;
        key.label = label;
        key.created_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (model_registry_.create_api_key(key)) {
            nlohmann::json response;
            response["id"] = key.id;
            response["key"] = new_key;
            response["label"] = key.label;
            response["created_at"] = key.created_at;
            res.set_content(response.dump(), "application/json");
        } else {
            send_error(res, 500, "server_error", "Failed to create API key", "KEY_CREATE_FAILED");
        }
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_lora_load(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    try {
        auto json = nlohmann::json::parse(req.body);
        std::string model_id = json.value("model", "");
        std::string adapter_id = json.value("adapter_id", json.value("adapter", ""));
        std::string path = json.value("path", "");
        if (model_id.empty() || adapter_id.empty() || path.empty()) {
            send_error(res, 400, "invalid_request", "model, adapter_id and path are required", "MISSING_PARAMS");
            return;
        }
        // Auto-load base model if needed
        if (!text_engine_.is_loaded(model_id)) {
            auto model = model_registry_.get_model(model_id);
            if (model) {
                if (!text_engine_.load_model(model_id, model->path)) {
                    send_error(res, 500, "server_error", "Failed to load base model", "MODEL_LOAD_FAILED");
                    return;
                }
            } else {
                send_error(res, 404, "not_found", "Model not found: " + model_id, "MODEL_NOT_FOUND");
                return;
            }
        }
        if (text_engine_.load_lora_adapter(model_id, adapter_id, path)) {
            nlohmann::json response;
            response["model"] = model_id;
            response["adapter_id"] = adapter_id;
            response["loaded"] = true;
            res.set_content(response.dump(), "application/json");
        } else {
            send_error(res, 500, "server_error", "Failed to load LoRA adapter", "LORA_LOAD_FAILED");
        }
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_lora_unload(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    try {
        auto json = nlohmann::json::parse(req.body);
        std::string model_id = json.value("model", "");
        std::string adapter_id = json.value("adapter_id", json.value("adapter", ""));
        if (model_id.empty() || adapter_id.empty()) {
            send_error(res, 400, "invalid_request", "model and adapter_id are required", "MISSING_PARAMS");
            return;
        }
        if (text_engine_.unload_lora_adapter(model_id, adapter_id)) {
            nlohmann::json response;
            response["model"] = model_id;
            response["adapter_id"] = adapter_id;
            response["loaded"] = false;
            res.set_content(response.dump(), "application/json");
        } else {
            send_error(res, 404, "not_found", "Adapter not found", "ADAPTER_NOT_FOUND");
        }
    } catch (const std::exception& e) {
        send_error(res, 400, "invalid_request", "Invalid JSON: " + std::string(e.what()), "INVALID_JSON");
    }
}

void Server::handle_lora_list(const httplib::Request& req, httplib::Response& res) {
    std::string api_key_id;
    if (!authenticate(req, api_key_id)) {
        send_error(res, 401, "auth_error", "Invalid or missing API key", "UNAUTHORIZED");
        return;
    }
    std::string model_id = req.has_param("model") ? req.get_param_value("model") : "";
    auto adapters = text_engine_.list_lora_adapters(model_id);
    nlohmann::json data = nlohmann::json::array();
    for (auto& a : adapters) data.push_back(a);
    nlohmann::json response;
    response["object"] = "list";
    response["model"] = model_id;
    response["data"] = data;
    res.set_content(response.dump(), "application/json");
}

bool Server::authenticate(const httplib::Request& req, std::string& api_key_id) {
      // ala llama-server: auth required only when at least one key is configured
      // (--api-key / BARSKUY_API_KEY / created via API). Fresh server = open.
      bool has_keys = !model_registry_.list_api_keys().empty();
      if (!has_keys) { api_key_id = "open"; return true; }
      if (!req.has_header("Authorization")) {
          return false;
      }

    std::string auth = req.get_header_value("Authorization");
    if (auth.rfind("Bearer ", 0) != 0) {
        return false;
    }

    std::string token = auth.substr(7);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(token.c_str()), token.length(), hash);

    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    std::string key_hash = ss.str();

    auto key = model_registry_.get_api_key_by_hash(key_hash);
    if (!key || key->revoked_at) {
        return false;
    }

    api_key_id = key->id;
    return true;
}

void Server::send_error(httplib::Response& res, int status, const std::string& type,
                        const std::string& message, const std::string& code) {
    metric_errors_total_++;
    nlohmann::json error_obj;
    error_obj["type"] = type;
    error_obj["message"] = message;
    error_obj["code"] = code;
    
    nlohmann::json error;
    error["error"] = error_obj;
    res.status = status;
    res.set_content(error.dump(), "application/json");
}

void Server::send_sse_chunk(httplib::Response& res, const std::string& data) {
    std::string chunk = "data: " + data + "\n\n";
    // For streaming, we need to use a content provider or write directly to the socket
    // httplib doesn't have a simple write method, so we'll use a different approach
    // This is a simplified version - in production you'd use a proper streaming approach
    auto& response = const_cast<httplib::Response&>(res);
    // We'll accumulate in a buffer and flush at the end for now
    // This is a placeholder - proper SSE streaming needs ContentProvider
    static thread_local std::string sse_buffer;
    sse_buffer += chunk;
    // Note: This is simplified. Proper SSE streaming requires ContentProvider
}

} // namespace barskuy::api