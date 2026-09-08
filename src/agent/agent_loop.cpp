#include "agent_loop.hpp"
#include "agent_tools.hpp"
#include "agent_mcp.hpp"
#include "core/logger.hpp"

#include "../engine/text_engine.hpp"

#include "chat.h"
#include "llama.h"

using nlohmann::json;

namespace barskuy::agent {

AgentRunner::AgentRunner(engine::TextEngine& eng, AgentTools& tools, McpManager& mcp)
    : eng_(eng), tools_(tools), mcp_(mcp) {}

// Lenient extractor for malformed tool calls (see above). Appends to msg.
static void fallback_tool_calls(const std::string& text, common_chat_msg& msg) {
    auto add = [&](const std::string& name, const std::string& args) {
        if (name.empty()) return;
        common_chat_tool_call tc;
        tc.name = name;
        tc.arguments = args.empty() ? "{}" : args;
        tc.id = "fb" + std::to_string(msg.tool_calls.size());
        msg.tool_calls.push_back(std::move(tc));
    };
    // <tool_call> ... </tool_call> with optional doubled braces
    size_t pos = 0;
    while ((pos = text.find("<tool_call>", pos)) != std::string::npos) {
        size_t beg = pos + 11;
        size_t end = text.find("</tool_call>", beg);
        if (end == std::string::npos) break;
        std::string inner = text.substr(beg, end - beg);
        pos = end + 12;
        // un-double braces: {{"name":..}} -> {"name":..}
        std::string u;
        for (size_t i = 0; i < inner.size(); ++i) {
            if (inner[i] == '{' && i + 1 < inner.size() && inner[i + 1] == '{') { u += '{'; ++i; }
            else if (inner[i] == '}' && i + 1 < inner.size() && inner[i + 1] == '}') { u += '}'; ++i; }
            else u += inner[i];
        }
        try {
            json j = json::parse(u);
            if (!j.contains("name")) continue;
            std::string args = j.contains("arguments")
                ? (j["arguments"].is_string() ? j["arguments"].get<std::string>() : j["arguments"].dump())
                : "{}";
            add(j["name"].is_string() ? j["name"].get<std::string>() : "", args);
        } catch (...) {}
    }
    if (!msg.tool_calls.empty()) return;
    // Hermes-style: <function=name>json-args</function>
    pos = 0;
    while ((pos = text.find("<function=", pos)) != std::string::npos) {
        size_t beg = pos + 10;
        size_t close = text.find('>', beg);
        if (close == std::string::npos) break;
        std::string name = text.substr(beg, close - beg);
        size_t end = text.find("</function>", close);
        if (end == std::string::npos) break;
        std::string inner = text.substr(close + 1, end - close - 1);
        pos = end + 11;
        // trim whitespace
        size_t s = inner.find_first_not_of(" \t\r\n");
        size_t e = inner.find_last_not_of(" \t\r\n");
        std::string args = (s == std::string::npos) ? "{}" : inner.substr(s, e - s + 1);
        if (!args.empty() && args.front() != '{') args = "{}";
        add(name, args);
    }
}

// BUG-051b: template Qwen ajari format XML <parameter=nilai>v</parameter>,
// tapi parse native/fallback kadang hanya dapat nama (args kosong -> {}).
// Pulihkan args dari pasangan XML di teks mentah untuk call yang blank.
static void recover_xml_args(const std::string& text,
                             std::vector<common_chat_tool_call>& calls) {
    auto blank = [](const std::string& a) {
        try {
            json j = json::parse(a);
            return !j.is_object() || j.empty();
        } catch (...) { return true; }
    };
    auto trim = [](std::string s) {
        size_t b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return std::string();
        return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
    };
    size_t pos = 0;
    for (auto& tc : calls) {
        if (!blank(tc.arguments)) continue;
        // cari <function=NAMA> setelah posisi terakhir (urut per call)
        std::string open = "<function=" + tc.name + ">";
        size_t f = text.find(open, pos);
        if (f == std::string::npos) f = text.find(open);
        if (f == std::string::npos) continue;
        size_t fend = text.find("</function>", f);
        if (fend == std::string::npos) continue;
        pos = fend + 11;
        json obj = json::object();
        size_t pp = f;
        while ((pp = text.find("<parameter=", pp)) != std::string::npos && pp < fend) {
            size_t bn = pp + 11;
            size_t be = text.find('>', bn);
            if (be == std::string::npos || be > fend) break;
            std::string pname = trim(text.substr(bn, be - bn));
            size_t ve = text.find("</parameter>", be);
            if (ve == std::string::npos || ve > fend) break;
            obj[pname] = trim(text.substr(be + 1, ve - be - 1));
            pp = ve + 12;
        }
        if (!obj.empty()) {
            tc.arguments = obj.dump();
            core::Logger::info("agent args recovered for {}", tc.name);
        }
    }
}

// Build common_chat_tool list from OpenAI-style tool defs.
static std::vector<common_chat_tool> to_ctools(const json& oa_tools) {
    std::vector<common_chat_tool> out;
    if (!oa_tools.is_array()) return out;
    for (auto& t : oa_tools) {
        if (!t.contains("function")) continue;
        auto& f = t["function"];
        common_chat_tool c;
        c.name = f.value("name", "");
        c.description = f.value("description", "");
        c.parameters = f.contains("parameters") ? f["parameters"].dump() : "{\"type\":\"object\"}";
        if (!c.name.empty()) out.push_back(std::move(c));
    }
    return out;
}

AgentResult AgentRunner::run(const std::string& model_id, const json& oa_messages,
                             const json& oa_tools, int max_tokens,
                             float temperature, float top_p,
                             const std::string& cwd) {
    AgentResult res;
    struct llama_model* lmodel = eng_.get_llama_model(model_id);
    if (!lmodel) { res.text = "Error: model not loaded"; res.finish_reason = "error"; return res; }

    common_chat_templates_ptr tmpls = common_chat_templates_init(lmodel, "", "", "");
    if (!tmpls) { res.text = "Error: cannot init chat templates"; res.finish_reason = "error"; return res; }

    std::vector<common_chat_msg> transcript;
    try {
        transcript = common_chat_msgs_parse_oaicompat(common_json::parse(oa_messages.dump()));
    } catch (...) {
        res.text = "Error: bad messages";
        res.finish_reason = "error";
        return res;
    }

    std::vector<common_chat_tool> ctools;
    if (oa_tools.is_null()) {
        ctools = to_ctools(tools_.definitions_openai());
        json mcp_defs = mcp_.definitions_openai();
        auto extra = to_ctools(mcp_defs);
        ctools.insert(ctools.end(), extra.begin(), extra.end());
    } else {
        try {
            ctools = common_chat_tools_parse_oaicompat(common_json::parse(oa_tools.dump()));
        } catch (...) {
            res.text = "Error: bad tools";
            res.finish_reason = "error";
            return res;
        }
    }
    if (ctools.empty()) { res.text = "Error: no tools available"; res.finish_reason = "error"; return res; }

    core::Logger::info("agent-loop start model={} max_steps={}", model_id, max_steps_);
    for (int i = 0; i < max_steps_; ++i) {
        core::Logger::info("agent-loop iter={} render", i);
        // Single render->generate->parse step over the live transcript
        SingleStep s;
        {
            common_chat_templates_inputs inputs;
            inputs.messages = transcript;
            inputs.tools = ctools;
            inputs.add_generation_prompt = true;
            common_chat_params params;
            try {
                params = common_chat_templates_apply(tmpls.get(), inputs);
            } catch (...) {
                res.text = "Error: template apply failed";
                res.finish_reason = "error";
                return res;
            }
            core::Logger::info("agent-loop iter applied prompt_toks~{}", (int)params.prompt.size());
            auto gen = eng_.complete_raw(model_id, params.prompt, max_tokens, temperature, top_p);
            core::Logger::info("agent-loop iter generated n={}", gen.completion_tokens);
            res.prompt_tokens += gen.prompt_tokens;
            res.completion_tokens += gen.completion_tokens;
            res.prompt_ms += gen.prompt_ms;
            res.predicted_ms += gen.predicted_ms;
            if (gen.finish_reason == "error") {
                res.text = gen.text;
                res.finish_reason = "error";
                return res;
            }
            // F9-10 BUG-051: parse terpusat (native + fallback + XML
            // recover) + repair sekali, sama seperti SingleStep. Tanpa ini
            // args XML Qwen hilang -> tool gagal -> loop sia-sia.
            std::string stext;
            std::vector<ToolCall> scalls;
            parse_step_text(model_id, oa_messages, oa_tools, gen.text, stext, scalls);
            repair_calls(model_id, oa_messages, oa_tools, params.prompt, gen.text,
                         stext, scalls);
            s.text = stext;
            for (auto& c : scalls) s.calls.push_back({c.id, c.name, c.arguments});
            {
                std::string names;
                for (auto& c : s.calls) {
                    if (!names.empty()) names += ",";
                    names += c.name + "(" + c.arguments.substr(0, 80) + ")";
                }
                core::Logger::info("agent-loop iter done gen={} calls={} [{}]",
                    (int)gen.completion_tokens, (int)s.calls.size(),
                    names.empty() ? "-" : names);
            }
        }

        if (s.calls.empty()) {
            res.text = s.text;
            res.finish_reason = "stop";
            return res;
        }

        common_chat_msg asst;
        asst.role = "assistant";
        asst.content = s.text;
        for (auto& c : s.calls) {
            common_chat_tool_call tc;
            tc.name = c.name; tc.arguments = c.arguments; tc.id = c.id;
            asst.tool_calls.push_back(tc);
        }
        transcript.push_back(asst);

        for (auto& c : s.calls) {
            json args = json::object();
            try { args = json::parse(c.arguments); } catch (...) {}
            std::string out;
            if (tools_.has_tool(c.name)) out = tools_.execute(c.name, args, cwd);
            else if (mcp_.has_tool(c.name)) out = mcp_.execute(c.name, args);
            else out = "Error: unknown tool '" + c.name + "'";
            res.steps.push_back({c.name, c.arguments, out});
            common_chat_msg tm;
            tm.role = "tool";
            tm.content = out;
            tm.tool_name = c.name;
            tm.tool_call_id = c.id;
            transcript.push_back(tm);
        }
    }

    res.finish_reason = "length";
    if (res.text.empty()) res.text = "(agent stopped after max steps)";
    return res;
}

SingleStep AgentRunner::step(const std::string& model_id, const json& oa_messages,
                             const json& oa_tools, int max_tokens,
                             float temperature, float top_p) {
    SingleStep s;
    std::string prompt, err;
    if (!render_step(model_id, oa_messages, oa_tools, prompt, err)) {
        s.text = "Error: " + err;
        s.error = true;
        return s;
    }
    // F9-10: single tool-decision step dibatasi. Budget ctx-penuh (8192)
    // bikin blocking menit-menit saat model meramble (UI stream stuck di
    // "Processing..."). 1024 cukup untuk think + tool_call.
    int budget = max_tokens > 0 && max_tokens < 1024 ? max_tokens : 1024;
    auto gen = eng_.complete_raw(model_id, prompt, budget, temperature, top_p);
    s.prompt_tokens = gen.prompt_tokens;
    s.completion_tokens = gen.completion_tokens;
    s.prompt_ms = gen.prompt_ms;
    s.predicted_ms = gen.predicted_ms;
    if (gen.finish_reason == "error") {
        s.text = gen.text;
        s.error = true;
        return s;
    }
    parse_step_text(model_id, oa_messages, oa_tools, gen.text, s.text, s.calls);
    repair_calls(model_id, oa_messages, oa_tools, prompt, gen.text, s.text, s.calls);
    return s;
}

// BUG-051: model kecil kadang emit tool call tanpa argumen ({}), lalu
// client looping error validasi. Coba sekali perbaiki: render ulang prompt
// + output parsial + teguran, generate pendek, pakai bila argumen terisi.
bool AgentRunner::repair_calls(const std::string& model_id,
                               const json& oa_messages,
                               const json& oa_tools,
                               const std::string& prompt,
                               const std::string& gen_text,
                               std::string& text_out,
                               std::vector<ToolCall>& calls_out) {
    auto has_args = [](const std::string& a) {
        try {
            json j = json::parse(a);
            return j.is_object() && !j.empty();
        } catch (...) { return !a.empty(); }
    };
    if (calls_out.empty()) return false;
    for (auto& c : calls_out) {
        if (has_args(c.arguments)) return false; // sudah bagus
    }
    std::string nudge = prompt + gen_text +
        "\n\n[SYSTEM: tool call di atas kehilangan argumen wajib. Ulangi tool call "
        "YANG SAMA dengan SEMUA parameter wajib terisi dari permintaan user. "
        "Hanya output tool call, tanpa teks lain.]";
    auto gen = eng_.complete_raw(model_id, nudge, 256, 0.0f, 0.9f);
    if (gen.finish_reason == "error" || gen.text.empty()) return false;
    std::string t2;
    std::vector<ToolCall> c2;
    parse_step_text(model_id, oa_messages, oa_tools, gen.text, t2, c2);
    for (auto& c : c2) {
        if (has_args(c.arguments)) {
            core::Logger::info("agent repair: args fixed for {}", c.name);
            text_out = t2;
            calls_out = c2;
            return true;
        }
    }
    return false;
}

// Render prompt WITH tools via native template (tanpa inferensi).
// Dipakai step() blocking dan jalur live (complete_stream_raw).
bool AgentRunner::render_step(const std::string& model_id,
                              const json& oa_messages,
                              const json& oa_tools,
                              std::string& prompt_out, std::string& err) {
    struct llama_model* lmodel = eng_.get_llama_model(model_id);
    if (!lmodel) { err = "model not loaded"; return false; }
    common_chat_templates_ptr tmpls = common_chat_templates_init(lmodel, "", "", "");
    if (!tmpls) { err = "cannot init chat templates"; return false; }
    std::vector<common_chat_msg> transcript;
    try {
        transcript = common_chat_msgs_parse_oaicompat(common_json::parse(oa_messages.dump()));
    } catch (...) { err = "bad messages"; return false; }
    std::vector<common_chat_tool> ctools;
    if (oa_tools.is_null()) {
        ctools = to_ctools(tools_.definitions_openai());
        auto extra = to_ctools(mcp_.definitions_openai());
        ctools.insert(ctools.end(), extra.begin(), extra.end());
    } else {
        try {
            ctools = common_chat_tools_parse_oaicompat(common_json::parse(oa_tools.dump()));
        } catch (...) { err = "bad tools"; return false; }
    }
    if (ctools.empty()) { err = "no tools available"; return false; }
    common_chat_templates_inputs inputs;
    inputs.messages = transcript;
    inputs.tools = ctools;
    inputs.add_generation_prompt = true;
    try {
        common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
        prompt_out = params.prompt;
    } catch (...) { err = "template apply failed"; return false; }
    return true;
}

// Parse teks mentah step -> {text, calls}. Render ulang params template
// secara internal (murah, tanpa inferensi).
void AgentRunner::parse_step_text(const std::string& model_id,
                                  const json& oa_messages,
                                  const json& oa_tools,
                                  const std::string& gen_text,
                                  std::string& text_out,
                                  std::vector<ToolCall>& calls_out) {
    text_out.clear();
    calls_out.clear();
    common_chat_msg msg;
    bool parsed = false;
    if (struct llama_model* lmodel = eng_.get_llama_model(model_id)) {
        common_chat_templates_ptr tmpls = common_chat_templates_init(lmodel, "", "", "");
        if (tmpls) {
            try {
                auto transcript =
                    common_chat_msgs_parse_oaicompat(common_json::parse(oa_messages.dump()));
                std::vector<common_chat_tool> ctools;
                if (oa_tools.is_null()) {
                    ctools = to_ctools(tools_.definitions_openai());
                    auto extra = to_ctools(mcp_.definitions_openai());
                    ctools.insert(ctools.end(), extra.begin(), extra.end());
                } else {
                    ctools =
                        common_chat_tools_parse_oaicompat(common_json::parse(oa_tools.dump()));
                }
                common_chat_templates_inputs inputs;
                inputs.messages = transcript;
                inputs.tools = ctools;
                inputs.add_generation_prompt = true;
                common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
                common_chat_parser_params pp(params);
                pp.parse_tool_calls = true;
                msg = common_chat_parse(gen_text, false, pp);
                parsed = true;
            } catch (...) {}
        }
    }
    if (!parsed) msg.content = gen_text;
    if (msg.tool_calls.empty()) fallback_tool_calls(gen_text, msg);
    recover_xml_args(gen_text, msg.tool_calls);
    text_out = !msg.content.empty() ? msg.content : gen_text;
    for (auto& tc : msg.tool_calls) {
        // F9-10: empty/blank arguments crash the WebUI's JSON.parse -> turn
        // ends up empty. Normalize to {} (upstream sends {} by default).
        std::string args = tc.arguments;
        bool blank = true;
        for (char c : args) {
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') { blank = false; break; }
        }
        if (blank) args = "{}";
        else {
            try {
                json j = json::parse(args);
                if (!j.is_object()) args = "{}";
            } catch (...) { args = "{}"; }
        }
        calls_out.push_back({tc.id, tc.name, args});
    }
}

} // namespace barskuy::agent
