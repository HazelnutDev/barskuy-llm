#pragma once

// Built-in agent tools (mirrors llama.cpp --tools names/semantics):
// get_datetime, read_file, write_file, edit_file,
// file_glob_search, grep_search, exec_shell_command.
// All file/shell access is sandboxed under tools_root.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace barskuy::agent {

class AgentTools {
public:
    explicit AgentTools(const std::string& tools_root = ".");
    void set_enabled(const std::vector<std::string>& names); // empty = all
    void set_enabled_all(bool on) { enabled_all_ = on; }

    bool has_tool(const std::string& name) const;
    // Upstream /tools shape for one tool (ServerToolInfo)
    nlohmann::json server_info(const std::string& name) const;
    std::vector<std::string> tool_names() const;
    nlohmann::json definitions_openai() const; // [{type:function,...}]
    nlohmann::json definitions_bare() const;   // [{tool,display_name,definition}]

    // Execute by name with OpenAI-style arguments object. Returns plain text.
    // cwd: per-call working dir override (UI x-tool-cwd header). Absolute
    // existing dir only; falls back to root_ otherwise. Thread-safe.
    std::string execute(const std::string& name, const nlohmann::json& args,
                        const std::string& cwd = "");

    const std::string& root() const { return root_; }

private:
    std::string root_;
    bool enabled_all_ = true;
    std::vector<std::string> enabled_;

    bool allowed(const std::string& name) const;
    // Effective root for this call (override or configured root).
    const std::string& eff_root() const;
    // Resolve user path against root; empty string if escape attempt.
    std::string resolve(const std::string& path) const;

    std::string t_datetime(const nlohmann::json& args);
    std::string t_info(const nlohmann::json& args);
    std::string t_read_file(const nlohmann::json& args);
    std::string t_write_file(const nlohmann::json& args);
    std::string t_edit_file(const nlohmann::json& args);
    std::string t_glob(const nlohmann::json& args);
    std::string t_grep(const nlohmann::json& args);
    std::string t_shell(const nlohmann::json& args);
};

} // namespace barskuy::agent
