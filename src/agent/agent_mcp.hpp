#pragma once

// MCP stdio client (Cursor-compatible mcp.json).
// Spawns each server lazily via common_subproc, speaks JSON-RPC (NDJSON),
// discovers tools once, keeps the process alive, respawns if dead.
// Tools are exposed as mcp__<server>__<tool> (never shadows built-ins).

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace barskuy::agent {

struct McpServerConfig {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;
    std::string cwd;
};

class McpManager {
public:
    McpManager();
    ~McpManager();

    bool load_config_file(const std::string& path, std::string& err);
    bool load_config_json(const nlohmann::json& cfg, std::string& err);
    std::vector<std::string> server_names() const;

    // Discover tools on all servers (spawns each once). Returns count.
    int discover_all();

    nlohmann::json definitions_openai() const; // [{type:function,...}]
    bool has_tool(const std::string& full_name) const;
    // Execute mcp__server__tool. Returns plain text (or "Error: ...").
    std::string execute(const std::string& full_name, const nlohmann::json& args);

private:
    struct Server; // pimpl per server (proc + reader thread + tool cache)
    std::map<std::string, std::shared_ptr<Server>> servers_;
    mutable std::mutex mu_;
};

} // namespace barskuy::agent
