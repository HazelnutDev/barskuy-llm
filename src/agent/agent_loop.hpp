#pragma once

// Agentic loop (F7, mirrors llama.cpp --tools/--agent behavior):
// render prompt WITH tools via the model's native chat template,
// generate, parse tool calls, execute (built-in or MCP), repeat.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace barskuy::engine { class TextEngine; }
namespace barskuy::agent {
class AgentTools;
class McpManager;

struct AgentStep {
    std::string tool;
    std::string args;
    std::string result;
};

struct AgentResult {
    std::string text;
    std::string finish_reason = "stop";
    std::vector<AgentStep> steps;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double prompt_ms = 0.0;     // F9-10 synced perf
    double predicted_ms = 0.0;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments; // JSON object string
};

struct SingleStep {
    std::string text;
    std::vector<ToolCall> calls;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double prompt_ms = 0.0;     // F9-10 synced perf
    double predicted_ms = 0.0;
    bool error = false;
};

class AgentRunner {
public:
    AgentRunner(engine::TextEngine& eng, AgentTools& tools, McpManager& mcp);
    void set_max_steps(int n) { max_steps_ = n; }

    // One render->generate->parse step (passthrough ala llama-server, F9-9).
    // oa_messages: OpenAI array; oa_tools: OpenAI array or null (= all ours).
    // transcript_out receives the assistant message (for loop callers).
    SingleStep step(const std::string& model_id,
                    const nlohmann::json& oa_messages,
                    const nlohmann::json& oa_tools,
                    int max_tokens = 512,
                    float temperature = 0.7f,
                    float top_p = 0.9f);

    // F9-10 live passthrough: render saja. false + err bila gagal;
    // prompt_out siap untuk complete_stream_raw.
    bool render_step(const std::string& model_id,
                     const nlohmann::json& oa_messages,
                     const nlohmann::json& oa_tools,
                     std::string& prompt_out, std::string& err);

    // F9-10: parse teks mentah step menjadi {text, calls} (render ulang
    // params template secara internal; murah, tanpa inferensi). Dipakai
    // step blocking maupun hasil akumulasi streaming.
    void parse_step_text(const std::string& model_id,
                         const nlohmann::json& oa_messages,
                         const nlohmann::json& oa_tools,
                         const std::string& gen_text,
                         std::string& text_out,
                         std::vector<ToolCall>& calls_out);

    // BUG-051: perbaiki tool call tanpa argumen (sekali coba).
    // true bila calls diganti hasil yang argumennya terisi.
    bool repair_calls(const std::string& model_id,
                      const nlohmann::json& oa_messages,
                      const nlohmann::json& oa_tools,
                      const std::string& rendered_prompt,
                      const std::string& gen_text,
                      std::string& text_out,
                      std::vector<ToolCall>& calls_out);

    // oa_messages: OpenAI [{role,content,...}]. oa_tools: OpenAI tools array
    // or null for default (all enabled built-in + discovered MCP tools).
    AgentResult run(const std::string& model_id,
                    const nlohmann::json& oa_messages,
                    const nlohmann::json& oa_tools,
                    int max_tokens = 512,
                    float temperature = 0.7f,
                    float top_p = 0.9f,
                    const std::string& cwd = "");

private:
    engine::TextEngine& eng_;
    AgentTools& tools_;
    McpManager& mcp_;
    int max_steps_ = 8;
};

} // namespace barskuy::agent
