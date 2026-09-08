#include "agent_mcp.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <thread>

#include "subproc.h"

#ifdef _WIN32
#include <windows.h>
#endif

using nlohmann::json;

namespace barskuy::agent {

namespace {
// Parent environment as K=V list (Windows), merged under cfg overrides.
std::vector<std::string> parent_env() {
    std::vector<std::string> out;
#ifdef _WIN32
    LPWCH block = GetEnvironmentStringsW();
    if (!block) return out;
    for (LPWCH p = block; *p; ) {
        std::wstring w(p);
        int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if (n > 1) {
            std::string s(n - 1, 0);
            WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
            out.push_back(s);
        }
        p += w.size() + 1;
    }
    FreeEnvironmentStringsW(block);
#endif
    return out;
}
}

struct McpManager::Server {
    McpServerConfig cfg;
    common_subproc proc;
    std::thread reader;
    std::thread errdrain;
    std::mutex mu;
    std::condition_variable cv;
    std::map<int64_t, std::string> pending; // id -> response json text
    std::atomic<int64_t> next_id{1};
    std::atomic<bool> stop{false};
    bool ready = false;
    json tools = json::array(); // [{name,description,inputSchema}]

    ~Server() { shutdown(); }

    void shutdown() {
        stop = true;
        cv.notify_all();
        proc.terminate();
        if (reader.joinable()) reader.join();
        if (errdrain.joinable()) errdrain.join();
        proc.join();
    }

    void reader_loop() {
        FILE* f = proc.stdout_file();
        if (!f) return;
        char buf[65536];
        std::string acc;
        while (!stop) {
            if (!fgets(buf, sizeof(buf), f)) break;
            acc += buf;
            size_t pos;
            while ((pos = acc.find('\n')) != std::string::npos) {
                std::string line = acc.substr(0, pos);
                acc.erase(0, pos + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                try {
                    json m = json::parse(line);
                    if (m.contains("id") && !m["id"].is_null()) {
                        int64_t id = m["id"].is_number() ? m["id"].get<int64_t>()
                                     : std::stoll(m["id"].get<std::string>());
                        std::lock_guard<std::mutex> lk(mu);
                        pending[id] = line;
                        cv.notify_all();
                    }
                } catch (...) {}
            }
        }
    }

    bool send(const json& msg) {
        std::string s = msg.dump() + "\n";
        FILE* f = proc.stdin_file();
        if (!f) return false;
        size_t off = 0;
        while (off < s.size()) {
            size_t w = fwrite(s.data() + off, 1, s.size() - off, f);
            if (w == 0) return false;
            off += w;
        }
        fflush(f);
        return true;
    }

    // Wait for response with matching id. Empty string on timeout/death.
    std::string request(const json& msg, int timeout_s = 120) {
        int64_t id = next_id++;
        json m = msg;
        m["id"] = id;
        m["jsonrpc"] = "2.0";
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!send(m)) return {};
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        std::unique_lock<std::mutex> lk(mu);
        while (!stop && proc.alive()) {
            auto it = pending.find(id);
            if (it != pending.end()) {
                std::string r = it->second;
                pending.erase(it);
                return r;
            }
            if (cv.wait_until(lk, deadline) == std::cv_status::timeout) break;
        }
        auto it = pending.find(id);
        if (it != pending.end()) {
            std::string r = it->second;
            pending.erase(it);
            return r;
        }
        return {};
    }

    bool ensure_started(std::string& err) {
        if (proc.alive() && ready) return true;
        shutdown();
        stop = false;
        pending.clear();
        if (!common_subproc::is_supported()) { err = "subprocess not supported in this build"; return false; }
        std::vector<std::string> argv;
#ifdef _WIN32
        // CreateProcess cannot run .cmd/.bat/.ps1 directly; route through cmd.
        std::string lc = cfg.command;
        for (auto& c : lc) c = (char)std::tolower((unsigned char)c);
        bool is_exe = lc.size() > 4 && lc.compare(lc.size() - 4, 4, ".exe") == 0;
        if (!is_exe) {
            argv.push_back("cmd");
            argv.push_back("/c");
        }
#endif
        argv.push_back(cfg.command);
        for (auto& a : cfg.args) argv.push_back(a);
        int options = subprocess_option_no_window | subprocess_option_search_user_path;
        std::vector<std::string> envp;
        if (cfg.env.empty()) {
            options |= subprocess_option_inherit_environment;
        } else {
            envp = parent_env();
            for (auto& [k, v] : cfg.env) {
                bool replaced = false;
                for (auto& e : envp) {
                    if (e.size() > k.size() && e.compare(0, k.size(), k) == 0 && e[k.size()] == '=') {
                        e = k + "=" + v;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) envp.push_back(k + "=" + v);
            }
        }
        if (!proc.create(argv, options, envp, cfg.cwd.empty() ? nullptr : cfg.cwd.c_str())) {
            err = "cannot spawn: " + cfg.command;
            return false;
        }
        reader = std::thread(&Server::reader_loop, this);
        FILE* ef = proc.stderr_file();
        errdrain = std::thread([ef]() {
            if (!ef) return;
            char b[4096];
            while (fgets(b, sizeof(b), ef)) {}
        });
        json init = {{"method", "initialize"}, {"params", {
            {"protocolVersion", "2025-11-25"},
            {"capabilities", json::object()},
            {"clientInfo", {{"name", "barskuy-llm"}, {"version", "0.1.0"}}}}}};
        std::string r = request(init, 60);
        if (r.empty()) { err = "MCP initialize timeout: " + cfg.name; shutdown(); return false; }
        json notif = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
        send(notif);
        json lst = {{"method", "tools/list"}, {"params", json::object()}};
        r = request(lst, 60);
        if (r.empty()) { err = "MCP tools/list timeout: " + cfg.name; shutdown(); return false; }
        try {
            json m = json::parse(r);
            if (m.contains("result") && m["result"].contains("tools")) tools = m["result"]["tools"];
        } catch (...) { err = "bad tools/list response: " + cfg.name; shutdown(); return false; }
        ready = true;
        return true;
    }
};

McpManager::McpManager() = default;
McpManager::~McpManager() = default;

bool McpManager::load_config_json(const json& cfg, std::string& err) {
    if (!cfg.contains("mcpServers") || !cfg["mcpServers"].is_object()) {
        err = "no mcpServers object";
        return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [name, sc] : cfg["mcpServers"].items()) {
        if (!sc.contains("command") || !sc["command"].is_string()) continue;
        auto s = std::make_shared<Server>();
        s->cfg.name = name;
        s->cfg.command = sc["command"].get<std::string>();
        if (sc.contains("args") && sc["args"].is_array())
            for (auto& a : sc["args"]) if (a.is_string()) s->cfg.args.push_back(a.get<std::string>());
        if (sc.contains("env") && sc["env"].is_object())
            for (auto& [k, v] : sc["env"].items()) if (v.is_string()) s->cfg.env[k] = v.get<std::string>();
        if (sc.contains("cwd") && sc["cwd"].is_string()) s->cfg.cwd = sc["cwd"].get<std::string>();
        servers_[name] = s;
    }
    if (servers_.empty()) { err = "no usable servers in config"; return false; }
    return true;
}

bool McpManager::load_config_file(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    try {
        json cfg = json::parse(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
        return load_config_json(cfg, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

std::vector<std::string> McpManager::server_names() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    for (auto& [k, _] : servers_) out.push_back(k);
    return out;
}

int McpManager::discover_all() {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (auto& [name, s] : servers_) {
        std::string err;
        if (s->ensure_started(err)) {
            n += (int)s->tools.size();
        } else {
            fprintf(stderr, "[mcp] discover failed for '%s': %s\n", name.c_str(), err.c_str());
        }
    }
    return n;
}

json McpManager::definitions_openai() const {
    json out = json::array();
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [srv, s] : servers_) {
        for (auto& t : s->tools) {
            std::string full = "mcp__" + srv + "__" + t.value("name", "");
            std::string desc = "[mcp:" + srv + "] " + t.value("description", "");
            json schema = t.contains("inputSchema") ? t["inputSchema"]
                                                    : json({{"type", "object"}});
            out.push_back({{"type", "function"},
                {"function", {{"name", full}, {"description", desc}, {"parameters", schema}}}});
        }
    }
    return out;
}

bool McpManager::has_tool(const std::string& full_name) const {
    if (full_name.compare(0, 5, "mcp__") != 0) return false;
    size_t p = full_name.find("__", 5);
    if (p == std::string::npos) return false;
    std::string srv = full_name.substr(5, p - 5);
    std::string tool = full_name.substr(p + 2);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = servers_.find(srv);
    if (it == servers_.end()) return false;
    for (auto& t : it->second->tools)
        if (t.value("name", "") == tool) return true;
    return false;
}

std::string McpManager::execute(const std::string& full_name, const json& args) {
    size_t p = full_name.find("__", 5);
    std::string srv = full_name.substr(5, p - 5);
    std::string tool = full_name.substr(p + 2);
    std::shared_ptr<Server> s;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = servers_.find(srv);
        if (it == servers_.end()) return "Error: unknown MCP server";
        s = it->second;
    }
    std::string err;
    if (!s->ensure_started(err)) return "Error: MCP server start failed: " + err;
    json call = {{"method", "tools/call"}, {"params", {
        {"name", tool}, {"arguments", args.is_object() ? args : json::object()}}}};
    std::string r = s->request(call, 300);
    if (r.empty()) return "Error: MCP call timeout/failed";
    try {
        json m = json::parse(r);
        if (m.contains("error")) return "Error: " + m["error"].value("message", "mcp error");
        std::string out;
        for (auto& c : m["result"]["content"]) {
            if (c.value("type", "") == "text") out += c.value("text", "");
            else out += c.dump();
            out += "\n";
        }
        if (m["result"].value("isError", false)) return "Error: " + out;
        if (out.empty()) return "(empty result)";
        if (out.size() > 64 * 1024) out = out.substr(0, 64 * 1024) + "\n[truncated: 64KB cap]";
        return out;
    } catch (const std::exception& e) {
        return std::string("Error: bad MCP response: ") + e.what();
    }
}

} // namespace barskuy::agent
