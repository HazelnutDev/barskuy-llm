#include "agent_tools.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace barskuy::agent {

namespace {
// Bracket class matcher for one [...] segment. Returns end index (past ']')
// or npos; sets hit. Supports [abc], [a-z], [!...]/[^...]. Needed because
// the WebUI sends case-insensitive globs like [Mm][Oo][Dd][Ee][Ll][Ss].
size_t glob_bracket(const std::string& pat, size_t p, char c, bool& hit) {
    hit = false;
    size_t q = p + 1;
    bool neg = false;
    if (q < pat.size() && (pat[q] == '!' || pat[q] == '^')) { neg = true; ++q; }
    bool any = false;
    while (q < pat.size() && pat[q] != ']') {
        if (q + 2 < pat.size() && pat[q + 1] == '-' && pat[q + 2] != ']') {
            if (c >= pat[q] && c <= pat[q + 2]) any = true;
            q += 3;
        } else {
            if (pat[q] == c) any = true;
            ++q;
        }
    }
    if (q >= pat.size()) return std::string::npos; // unclosed: literal fallback
    hit = neg ? !any : any;
    return q + 1;
}
bool glob_match(const std::string& pat, const std::string& s) {
    // iterative matcher with '*' backtracking + single-char classes
    size_t p = 0, i = 0;
    size_t star_p = std::string::npos, star_i = 0;
    auto fail = [&]() -> bool {
        if (star_p != std::string::npos && star_i < s.size()) {
            ++star_i;
            p = star_p + 1; i = star_i;
            return true;
        }
        return false;
    };
    while (true) {
        if (p < pat.size() && pat[p] == '*') { star_p = p++; star_i = i; continue; }
        if (i >= s.size()) {
            while (p < pat.size() && pat[p] == '*') ++p;
            if (p == pat.size()) return true;
            if (!fail()) return false;
            continue;
        }
        if (p >= pat.size()) { if (!fail()) return false; continue; }
        if (pat[p] == '?') { ++p; ++i; continue; }
        if (pat[p] == '[') {
            bool hit = false;
            size_t e = glob_bracket(pat, p, s[i], hit);
            if (e == std::string::npos) { // literal '['
                if (pat[p] != s[i]) { if (!fail()) return false; continue; }
                ++p; ++i; continue;
            }
            if (!hit) { if (!fail()) return false; continue; }
            p = e; ++i; continue;
        }
        if (pat[p] != s[i]) { if (!fail()) return false; continue; }
        ++p; ++i;
    }
}
std::string lower_str(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
}

AgentTools::AgentTools(const std::string& tools_root) {
    std::error_code ec;
    root_ = fs::absolute(tools_root, ec).lexically_normal().string();
    if (ec || root_.empty()) root_ = fs::current_path(ec).string();
}

namespace {
// Per-call cwd override (UI x-tool-cwd). thread_local: no races between
// concurrent requests, no leak across calls (RAII guard clears it).
thread_local std::string t_cwd_override;
std::string normalize_cwd(const std::string& d) {
    if (d.empty()) return {};
    std::error_code ec;
    fs::path p(d);
    if (!p.is_absolute()) return {};
    if (!fs::is_directory(p, ec)) return {};
    return p.lexically_normal().string();
}
struct CwdGuard {
    CwdGuard(const std::string& cwd) { t_cwd_override = normalize_cwd(cwd); }
    ~CwdGuard() { t_cwd_override.clear(); }
};
}

const std::string& AgentTools::eff_root() const {
    if (!t_cwd_override.empty()) return t_cwd_override;
    return root_;
}

void AgentTools::set_enabled(const std::vector<std::string>& names) {
    enabled_ = names;
    enabled_all_ = false;
}

bool AgentTools::allowed(const std::string& name) const {
    if (enabled_all_) return true;
    return std::find(enabled_.begin(), enabled_.end(), name) != enabled_.end();
}

bool AgentTools::has_tool(const std::string& name) const {
    for (auto& n : tool_names()) if (n == name) return true;
    return false;
}

std::vector<std::string> AgentTools::tool_names() const {
    std::vector<std::string> all = {"get_datetime", "get_info", "read_file", "write_file", "edit_file",
        "file_glob_search", "grep_search", "exec_shell_command"};
    if (enabled_all_) return all;
    std::vector<std::string> out;
    for (auto& n : all) if (allowed(n)) out.push_back(n);
    return out;
}

std::string AgentTools::resolve(const std::string& path) const {
    std::error_code ec;
    const std::string& er = eff_root();
    fs::path p(path);
    fs::path abs = p.is_absolute() ? p.lexically_normal()
                                   : (fs::path(er) / p).lexically_normal();
    std::string a = abs.string(), r = er;
#ifdef _WIN32
    a = lower_str(a); r = lower_str(r);
#endif
    if (a == r || (a.size() > r.size() && a.compare(0, r.size(), r) == 0 &&
        (a[r.size()] == '/' || a[r.size()] == '\\'))) {
        return abs.string();
    }
    return {};
}

json AgentTools::definitions_openai() const {
    json tools = json::array();
    auto fn = [&](const std::string& name, const std::string& desc, const json& params) {
        tools.push_back({{"type", "function"},
            {"function", {{"name", name}, {"description", desc}, {"parameters", params}}}});
    };
    json obj = {{"type", "object"}, {"properties", json::object()}};
    if (!allowed("get_datetime")) {} else fn("get_datetime", "Current local date and time. Call ONLY when the user explicitly asks about date, time, day, or something time-sensitive. Otherwise answer directly without calling any tool.",
        {{"type", "object"}, {"properties", json::object()}});
    if (allowed("get_info")) fn("get_info", "Server and loaded-model summary. Call ONLY when the user explicitly asks about server/model status or info. Otherwise answer directly.",
        {{"type", "object"}, {"properties", json::object()}});
    if (allowed("read_file")) fn("read_file", "Read a local text file. Call ONLY when the user explicitly names a file or path to read. Never explore files on your own. Supports offset/limit in lines.",
        {{"type", "object"}, {"properties", {
            {"path", {{"type", "string"}, {"description", "Relative or absolute path inside tools root"}}},
            {"offset", {{"type", "integer"}, {"description", "First line (1-based)"}}},
            {"limit", {{"type", "integer"}, {"description", "Max lines"}}}}},
            {"required", json::array({"path"})}});
    if (allowed("write_file")) fn("write_file", "Create or overwrite a file. Call ONLY when the user explicitly asks to write, save, or create a file. Never write unprompted. Parent dirs are created.",
        {{"type", "object"}, {"properties", {
            {"path", {{"type", "string"}}},
            {"content", {{"type", "string"}}}}},
            {"required", json::array({"path", "content"})}});
    if (allowed("edit_file")) fn("edit_file", "Edit a file. Call ONLY when the user explicitly asks to modify a file. old_text must occur exactly once.",
        {{"type", "object"}, {"properties", {
            {"path", {{"type", "string"}}},
            {"old_text", {{"type", "string"}}},
            {"new_text", {{"type", "string"}}}}},
            {"required", json::array({"path", "old_text", "new_text"})}});
    if (allowed("file_glob_search")) fn("file_glob_search", "Find files by name. Call ONLY when the user explicitly asks to find or list files. Never browse the filesystem unprompted.",
        {{"type", "object"}, {"properties", {
            {"pattern", {{"type", "string"}, {"description", "e.g. *.cpp"}}},
            {"path", {{"type", "string"}, {"description", "Directory, default tools root"}}}}},
            {"required", json::array({"pattern"})}});
    if (allowed("grep_search")) fn("grep_search", "Search inside file contents. Call ONLY when the user explicitly asks to search code or text. Returns file:line matches.",
        {{"type", "object"}, {"properties", {
            {"pattern", {{"type", "string"}}},
            {"path", {{"type", "string"}, {"description", "File or directory, default tools root"}}},
            {"include", {{"type", "string"}, {"description", "Filename glob filter, e.g. *.md"}}}}},
            {"required", json::array({"pattern"})}});
    if (allowed("exec_shell_command")) fn("exec_shell_command", "Run a shell command. Call ONLY when the user explicitly asks to run or execute something. Never run commands unprompted. Cwd is the tools root. Killed after timeout.",
        {{"type", "object"}, {"properties", {
            {"command", {{"type", "string"}}},
            {"timeout", {{"type", "integer"}, {"description", "Kill after N seconds (1-600, default 60)"}}}}},
            {"required", json::array({"command"})}});
    (void)obj;
    return tools;
}

json AgentTools::server_info(const std::string& name) const {
    // Upstream ServerToolInfo shape so llama.cpp WebUI lists/executes natively
    for (auto& t : definitions_openai()) {
        if (t["function"]["name"] != name) continue;
        bool w = (name == "write_file" || name == "edit_file" || name == "exec_shell_command");
        return {{"display_name", name}, {"tool", name}, {"type", "server"},
            {"permissions", {{"write", w}}}, {"uses_cwd", true},
            {"definition", {{"type", "function"}, {"function", t["function"]}}}};
    }
    return nullptr;
}

json AgentTools::definitions_bare() const {
    json out = json::array();
    for (auto& t : definitions_openai()) {
        auto& f = t["function"];
        out.push_back({{"tool", f["name"]}, {"display_name", f["name"]},
            {"definition", {{"function", f}}}});
    }
    return out;
}

std::string AgentTools::execute(const std::string& name, const json& args,
                                const std::string& cwd) {
    if (!has_tool(name)) return "Error: tool '" + name + "' is not enabled";
    json a = args.is_object() ? args : json::object();
    CwdGuard g(cwd);
    if (!cwd.empty()) {
        if (t_cwd_override.empty())
            return "Error: cwd not found or not a directory: " + cwd;
        core::Logger::info("tool {} cwd override: {}", name, t_cwd_override);
    }
    try {
        if (name == "get_datetime") return t_datetime(a);
        if (name == "get_info") return t_info(a);
        if (name == "read_file") return t_read_file(a);
        if (name == "write_file") return t_write_file(a);
        if (name == "edit_file") return t_edit_file(a);
        if (name == "file_glob_search") return t_glob(a);
        if (name == "grep_search") return t_grep(a);
        if (name == "exec_shell_command") return t_shell(a);
    } catch (const std::exception& e) {
        return std::string("Error: ") + e.what();
    }
    return "Error: unknown tool";
}

std::string AgentTools::t_datetime(const json&) {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %z", std::localtime(&t));
    return buf;
}

std::string AgentTools::t_info(const json&) {
    // Upstream get_info parity: short server/model summary as plain text
    std::string out = "barskuy-llm 0.1.0 (llama.cpp backend)\n";
    out += "tools_root: " + eff_root() + "\n";
    out += "tools: ";
    auto names = tool_names();
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) out += ", ";
        out += names[i];
    }
    return out;
}

std::string AgentTools::t_read_file(const json& a) {
    std::string rp = resolve(a.value("path", ""));
    if (rp.empty()) return "Error: path escapes tools root";
    std::ifstream f(rp, std::ios::binary);
    if (!f) return "Error: cannot open file";
    int offset = a.value("offset", 1), limit = a.value("limit", 2000);
    if (offset < 1) offset = 1;
    if (limit < 1 || limit > 2000) limit = 2000;
    std::string line, out;
    int ln = 0, kept = 0;
    while (std::getline(f, line)) {
        ++ln;
        if (ln < offset) continue;
        if (kept >= limit) { out += "\n[truncated]"; break; }
        if (out.size() > 100 * 1024) { out += "\n[truncated: 100KB cap]"; break; }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out += line + "\n";
        ++kept;
    }
    if (kept == 0) return "(empty or beyond EOF)";
    return out;
}

std::string AgentTools::t_write_file(const json& a) {
    std::string rp = resolve(a.value("path", ""));
    if (rp.empty()) return "Error: path escapes tools root";
    std::error_code ec;
    fs::create_directories(fs::path(rp).parent_path(), ec);
    std::ofstream f(rp, std::ios::binary | std::ios::trunc);
    if (!f) return "Error: cannot write file";
    f << a.value("content", "");
    return "Wrote " + std::to_string(a.value("content", "").size()) + " bytes to " + a.value("path", "");
}

std::string AgentTools::t_edit_file(const json& a) {
    std::string rp = resolve(a.value("path", ""));
    if (rp.empty()) return "Error: path escapes tools root";
    std::ifstream f(rp, std::ios::binary);
    if (!f) return "Error: cannot open file";
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string old_t = a.value("old_text", ""), new_t = a.value("new_text", "");
    if (old_t.empty()) return "Error: old_text is empty";
    size_t p1 = content.find(old_t);
    if (p1 == std::string::npos) return "Error: old_text not found";
    if (content.find(old_t, p1 + 1) != std::string::npos) return "Error: old_text matches multiple spans, be more specific";
    content.replace(p1, old_t.size(), new_t);
    std::ofstream o(rp, std::ios::binary | std::ios::trunc);
    if (!o) return "Error: cannot write file";
    o << content;
    return "Edited 1 span in " + a.value("path", "");
}

std::string AgentTools::t_glob(const json& a) {
    // Upstream llama.cpp WebUI compat (F9-10): its resolveServerHome calls
    // with {limit, max_depth, path:"~", type:"dir"} and reads res.base.
    // JSON shape iff upstream-only params (type/max_depth) are present,
    // otherwise legacy plain text (direct callers + agent readability).
    bool upstream = a.contains("type") || a.contains("max_depth");
    std::string dir = a.value("path", "");
    if (dir == "~" || dir.compare(0, 2, "~/") == 0) dir.clear(); // ~ = root
    std::string rp = dir.empty() ? eff_root() : resolve(dir);
    if (rp.empty()) {
        if (upstream) return json({{"entries", json::array()}, {"base", eff_root()}}).dump();
        return "Error: path escapes tools root";
    }
    // F9-10 BUG-056b: UI kirim "include", skema OA kita "pattern". Tanpa
    // ini pola selalu "*" -> daftar tak terfilter -> folder tak ketemu.
    std::string pat = a.value("include", "");
    if (pat.empty()) pat = a.value("pattern", "*");
    if (pat.empty()) pat = "*";
    std::string type = a.value("type", "all");
    int max_depth = a.value("max_depth", 1000000);
    int limit = a.value("limit", 100);
    if (limit < 1) limit = 1;
    if (limit > 500) limit = 500;
    std::error_code ec;
    json entries = json::array();
    std::string out;
    int n = 0;
    std::function<void(const fs::path&, int)> walk = [&](const fs::path& d, int depth) {
        if (n >= limit || depth > max_depth) return;
        for (auto it = fs::directory_iterator(d, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (ec || n >= limit) break;
            bool is_dir = it->is_directory(ec);
            bool is_file = it->is_regular_file(ec);
            if (type == "dir" && !is_dir) {
                if (is_dir && depth < max_depth) walk(it->path(), depth + 1);
                continue;
            }
            if (type == "file" && !is_file) {
                if (is_dir && depth < max_depth) walk(it->path(), depth + 1);
                continue;
            }
            if (!is_dir && !is_file) continue;
            if (!upstream && !is_file) {
                if (is_dir && depth < max_depth) walk(it->path(), depth + 1);
                continue; // legacy text shape lists files only
            }
            std::string name = it->path().filename().string();
            if (!glob_match(pat, name)) {
                if (is_dir && depth < max_depth) walk(it->path(), depth + 1);
                continue;
            }
            std::string rel = fs::relative(it->path(), eff_root(), ec).string();
            if (upstream) entries.push_back({{"path", rel}, {"type", is_dir ? "dir" : "file"}});
            else out += rel + "\n";
            if (++n >= limit) break;
            if (is_dir && depth < max_depth) walk(it->path(), depth + 1);
        }
    };
    walk(rp, 0);
    if (upstream) return json({{"entries", entries}, {"base", eff_root()}}).dump();
    if (n == 0) return "(no matches)";
    if (n >= 100) out += "[truncated: 100 results]";
    return out;
}

std::string AgentTools::t_grep(const json& a) {
    std::string target = a.value("path", "");
    std::string rp = target.empty() ? eff_root() : resolve(target);
    if (rp.empty()) return "Error: path escapes tools root";
    std::string inc = a.value("include", "*");
    std::regex re;
    try { re = std::regex(a.value("pattern", "")); }
    catch (...) { return "Error: invalid regex"; }
    std::error_code ec;
    std::vector<fs::path> files;
    if (fs::is_regular_file(rp, ec)) files.push_back(rp);
    else {
        for (auto it = fs::recursive_directory_iterator(rp, ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file(ec)) continue;
            if (!glob_match(inc, it->path().filename().string())) continue;
            if (it->file_size(ec) > 1024 * 1024) continue;
            files.push_back(it->path());
            if (files.size() > 2000) break;
        }
    }
    std::string out;
    int n = 0;
    for (auto& fp : files) {
        std::ifstream f(fp, std::ios::binary);
        if (!f) continue;
        std::string line;
        int ln = 0;
        while (std::getline(f, line)) {
            ++ln;
            if (line.size() > 2000) continue;
            std::smatch m;
            std::string probe = line;
            if (!probe.empty() && probe.back() == '\r') probe.pop_back();
            if (std::regex_search(probe, m, re)) {
                std::string rel = fs::relative(fp, eff_root(), ec).string();
                out += rel + ":" + std::to_string(ln) + ": " + probe.substr(0, 300) + "\n";
                if (++n >= 50) { out += "[truncated: 50 matches]"; return out; }
            }
        }
    }
    if (n == 0) return "(no matches)";
    return out;
}

std::string AgentTools::t_shell(const json& a) {
    std::string cmd = a.value("command", "");
    if (cmd.empty()) return "Error: empty command";
    int timeout_s = a.value("timeout", 60);
    if (timeout_s < 1) timeout_s = 1;
    if (timeout_s > 600) timeout_s = 600;
#ifdef _WIN32
    // CreateProcess with timeout so agentic loops can't hang on interactive
    // commands (F9-10). stdout+stderr merged, cwd = tools root.
    std::string inner = "cd /d \"" + eff_root() + "\" && " + cmd;
    std::string cmdline = "cmd.exe /S /C \"" + inner + "\"";
    std::vector<char> cl(cmdline.begin(), cmdline.end());
    cl.push_back(0);
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE hRead = nullptr, hWrite = nullptr;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0)) return "Error: cannot create pipe";
    SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = hWrite;
    si.hStdError = hWrite;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::string out;
    int rc = -1;
    bool timed_out = false;
    if (!CreateProcessA(nullptr, cl.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        CloseHandle(hRead);
        CloseHandle(hWrite);
        return "Error: cannot spawn shell";
    }
    CloseHandle(hWrite);
    hWrite = nullptr;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    char buf[4096];
    DWORD avail = 0;
    bool live = true;
    while (live) {
        while (PeekNamedPipe(hRead, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
            DWORD got = 0;
            DWORD want = (DWORD)std::min<size_t>(sizeof(buf), avail);
            if (!ReadFile(hRead, buf, want, &got, nullptr) || got == 0) break;
            out.append(buf, got);
            if (out.size() > 64 * 1024) {
                out += "\n[truncated: 64KB cap]";
                live = false;
                break;
            }
        }
        if (!live) break;
        DWORD wr = WaitForSingleObject(pi.hProcess, 50);
        if (wr == WAIT_OBJECT_0) {
            while (PeekNamedPipe(hRead, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
                DWORD got = 0;
                DWORD want = (DWORD)std::min<size_t>(sizeof(buf), avail);
                if (!ReadFile(hRead, buf, want, &got, nullptr) || got == 0) break;
                out.append(buf, got);
                if (out.size() > 64 * 1024) { out += "\n[truncated: 64KB cap]"; break; }
            }
            DWORD code = 0;
            GetExitCodeProcess(pi.hProcess, &code);
            rc = (int)code;
            live = false;
        } else if (std::chrono::steady_clock::now() >= deadline) {
            TerminateProcess(pi.hProcess, 124);
            WaitForSingleObject(pi.hProcess, 2000);
            rc = 124;
            timed_out = true;
            live = false;
        }
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hRead);
    if (timed_out) out += "\n[timeout after " + std::to_string(timeout_s) + "s, process killed]";
    out += "\n[exit: " + std::to_string(rc) + "]";
    return out;
#else
    std::string full = "cd \"" + eff_root() + "\" && " + cmd + " 2>&1";
    FILE* pipe = popen(full.c_str(), "r");
    if (!pipe) return "Error: cannot spawn shell";
    std::string out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) {
        out += buf;
        if (out.size() > 64 * 1024) { out += "\n[truncated: 64KB cap]"; break; }
    }
    int rc = pclose(pipe);
    out += "\n[exit: " + std::to_string(rc) + "]";
    return out;
#endif
}

} // namespace barskuy::agent
