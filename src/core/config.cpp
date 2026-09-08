#include "config.hpp"
#include "logger.hpp"
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>

namespace {
// "0.6,0.4" -> {0.6f, 0.4f}
std::vector<float> parse_tensor_split(const std::string& s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try { out.push_back(std::stof(tok)); } catch (...) {}
    }
    return out;
}
}

namespace barskuy::core {

bool Config::load(int argc, char* argv[]) {
    set_defaults();
    parse_env();
    parse_args(argc, argv);

    barskuy::core::Logger::info("Config: host={}, port={}, db={}, models_dir={}",
        host_, port_, database_path_, models_dir_);
    return true;
}

bool Config::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        if (key == "host") host_ = value;
        else if (key == "port") port_ = static_cast<uint16_t>(std::stoi(value));
        else if (key == "database_path") database_path_ = value;
        else if (key == "models_dir") models_dir_ = value;
        else if (key == "max_context_size") max_context_size_ = std::stoull(value);
        else if (key == "max_batch_size") max_batch_size_ = std::stoull(value);
        else if (key == "gpu_layers") gpu_layers_ = std::stoi(value);
        else if (key == "enable_mmap") enable_mmap_ = (value == "true" || value == "1");
        else if (key == "num_threads") num_threads_ = std::stoi(value);
        else if (key == "max_loaded_models") max_loaded_models_ = std::stoi(value);
        else if (key == "tensor_split") tensor_split_ = parse_tensor_split(value);
        else if (key == "split_mode") split_mode_ = std::stoi(value);
        else if (key == "tools") tools_mode_ = value;
        else if (key == "agent") agent_enabled_ = (value == "true" || value == "1");
        else if (key == "mcp_config") mcp_config_ = value;
        else if (key == "mcp") mcp_enabled_ = (value == "true" || value == "1");
        else if (key == "tools_root") tools_root_ = value;
        else if (key == "max_agent_steps") max_agent_steps_ = std::stoi(value);
        else if (key == "model") preload_models_.push_back(value);
        else if (key == "api_key") api_key_ = value;
        else if (key == "n_batch") n_batch_ = std::stoi(value);
        else if (key == "n_ubatch") n_ubatch_ = std::stoi(value);
        else if (key == "n_cpu_moe") n_cpu_moe_ = std::stoi(value);
        else if (key == "flash_attn") flash_attn_ = value;
        else if (key == "cache_type_k") cache_type_k_ = value;
        else if (key == "cache_type_v") cache_type_v_ = value;
        else if (key == "prio") prio_ = std::stoi(value);
        else if (key == "prio_batch") prio_batch_ = std::stoi(value);
        else if (key == "np") np_ = std::stoi(value);
    }
    return true;
}

void Config::set_defaults() {
    char* env_host = std::getenv("BARSKUY_HOST");
    char* env_port = std::getenv("BARSKUY_PORT");
    char* env_db = std::getenv("BARSKUY_DATABASE_PATH");
    char* env_models = std::getenv("BARSKUY_MODELS_DIR");

    if (env_host) host_ = env_host;
    if (env_port) port_ = static_cast<uint16_t>(std::stoi(env_port));
    if (env_db) database_path_ = env_db;
    if (env_models) models_dir_ = env_models;
}

void Config::parse_env() {
    char* val;
    if ((val = std::getenv("BARSKUY_MAX_CONTEXT_SIZE"))) max_context_size_ = std::stoull(val);
    if ((val = std::getenv("BARSKUY_MAX_BATCH_SIZE"))) max_batch_size_ = std::stoull(val);
    if ((val = std::getenv("BARSKUY_GPU_LAYERS"))) gpu_layers_ = std::stoi(val);
    if ((val = std::getenv("BARSKUY_ENABLE_MMAP"))) enable_mmap_ = (std::string(val) == "true" || std::string(val) == "1");
    if ((val = std::getenv("BARSKUY_NUM_THREADS"))) num_threads_ = std::stoi(val);
    if ((val = std::getenv("BARSKUY_MAX_LOADED_MODELS"))) max_loaded_models_ = std::stoi(val);
    if ((val = std::getenv("BARSKUY_TENSOR_SPLIT"))) tensor_split_ = parse_tensor_split(val);
    if ((val = std::getenv("BARSKUY_SPLIT_MODE"))) split_mode_ = std::stoi(val);
    if ((val = std::getenv("BARSKUY_TOOLS"))) tools_mode_ = val;
    if ((val = std::getenv("BARSKUY_AGENT"))) agent_enabled_ = (std::string(val) == "true" || std::string(val) == "1");
    if ((val = std::getenv("BARSKUY_MCP_CONFIG"))) mcp_config_ = val;
    if ((val = std::getenv("BARSKUY_MCP"))) mcp_enabled_ = (std::string(val) == "true" || std::string(val) == "1");
    if ((val = std::getenv("BARSKUY_TOOLS_ROOT"))) tools_root_ = val;
    if ((val = std::getenv("BARSKUY_MAX_AGENT_STEPS"))) max_agent_steps_ = std::stoi(val);
    if ((val = std::getenv("BARSKUY_MODEL"))) preload_models_.push_back(val);
    if ((val = std::getenv("BARSKUY_API_KEY_FLAG"))) api_key_ = val;
}

void Config::parse_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) host_ = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port_ = static_cast<uint16_t>(std::stoi(argv[++i]));
        else if (arg == "--db" && i + 1 < argc) database_path_ = argv[++i];
        else if (arg == "--models-dir" && i + 1 < argc) models_dir_ = argv[++i];
        else if (arg == "--gpu-layers" && i + 1 < argc) gpu_layers_ = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads_ = std::stoi(argv[++i]);
        else if (arg == "--no-mmap") enable_mmap_ = false;
        else if (arg == "--mmap") enable_mmap_ = true;
        else if (arg == "--max-loaded-models" && i + 1 < argc) max_loaded_models_ = std::stoi(argv[++i]);
        else if (arg == "--tensor-split" && i + 1 < argc) tensor_split_ = parse_tensor_split(argv[++i]);
        else if (arg == "--split-mode" && i + 1 < argc) split_mode_ = std::stoi(argv[++i]);
        else if (arg == "--tools" && i + 1 < argc) tools_mode_ = argv[++i];
        else if (arg == "--no-tools") tools_mode_.clear();
        else if (arg == "--agent") agent_enabled_ = true;
        else if (arg == "--no-agent") agent_enabled_ = false;
        else if (arg == "--mcp-servers-config" && i + 1 < argc) { mcp_config_ = argv[++i]; mcp_enabled_ = true; }
        else if (arg == "--mcp") mcp_enabled_ = true;
        else if (arg == "--no-mcp") mcp_enabled_ = false;
        else if (arg == "--tools-root" && i + 1 < argc) tools_root_ = argv[++i];
        else if (arg == "--max-agent-steps" && i + 1 < argc) max_agent_steps_ = std::stoi(argv[++i]);
        else if ((arg == "--model" || arg == "-m") && i + 1 < argc) preload_models_.push_back(argv[++i]);
        else if (arg == "--api-key" && i + 1 < argc) api_key_ = argv[++i];
        else if (arg == "--jinja") { core::Logger::info("Note: native chat templates always on (--jinja accepted)"); }
        else if (arg == "--perf") { core::Logger::info("Note: perf counters at /metrics (--perf accepted)"); }
        else if (arg == "--no-context-shift") { core::Logger::info("Note: no context shift implemented (--no-context-shift accepted)"); }
        else if (arg == "--no-warmup") { core::Logger::info("Note: no warmup implemented (--no-warmup accepted)"); }
        else if (arg == "--n-cpu-moe" && i + 1 < argc) n_cpu_moe_ = std::stoi(argv[++i]);
        else if (arg == "--flash-attn" && i + 1 < argc) flash_attn_ = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) n_batch_ = std::stoi(argv[++i]);
        else if (arg == "--ubatch-size" && i + 1 < argc) n_ubatch_ = std::stoi(argv[++i]);
        else if (arg == "--cache-type-k" && i + 1 < argc) cache_type_k_ = argv[++i];
        else if (arg == "--cache-type-v" && i + 1 < argc) cache_type_v_ = argv[++i];
        else if (arg == "--prio" && i + 1 < argc) prio_ = std::stoi(argv[++i]);
        else if (arg == "--prio-batch" && i + 1 < argc) prio_batch_ = std::stoi(argv[++i]);
        else if (arg == "--np" && i + 1 < argc) np_ = std::stoi(argv[++i]);
        else if ((arg == "--context-size" || arg == "-ctk") && i + 1 < argc)
            max_context_size_ = std::stoull(argv[++i]);
        // Image (F10, ComfyUI defaults; semua override-able)
        else if (arg == "--image-model" && i + 1 < argc) image_model_ = argv[++i];
        else if (arg == "--image-width" && i + 1 < argc) image_width_ = std::stoi(argv[++i]);
        else if (arg == "--image-height" && i + 1 < argc) image_height_ = std::stoi(argv[++i]);
        else if (arg == "--image-steps" && i + 1 < argc) image_steps_ = std::stoi(argv[++i]);
        else if (arg == "--image-cfg" && i + 1 < argc) image_cfg_ = std::stof(argv[++i]);
        else if (arg == "--image-sampler" && i + 1 < argc) image_sampler_ = argv[++i];
        else if (arg == "--image-scheduler" && i + 1 < argc) image_scheduler_ = argv[++i];
        else if (arg == "--image-flow-shift" && i + 1 < argc) image_flow_shift_ = std::stof(argv[++i]);
        else if (arg == "--image-cpu" && i + 1 < argc) image_cpu_ = std::stoi(argv[++i]);
        // Video (F10)
        else if (arg == "--video-model" && i + 1 < argc) video_model_ = argv[++i];
        else if (arg == "--video-width" && i + 1 < argc) video_width_ = std::stoi(argv[++i]);
        else if (arg == "--video-height" && i + 1 < argc) video_height_ = std::stoi(argv[++i]);
        else if (arg == "--video-steps" && i + 1 < argc) video_steps_ = std::stoi(argv[++i]);
        else if (arg == "--video-frames" && i + 1 < argc) video_frames_ = std::stoi(argv[++i]);
        else if (arg == "--video-cpu" && i + 1 < argc) video_cpu_ = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: barskuy-llm [options]\n"
                      << "\n[General]\n"
                      << "  --host <addr>        Bind address (default: 0.0.0.0)\n"
                      << "  --port <num>         Port number (default: 8080)\n"
                      << "  --db <path>          Database file path (default: barskuy.db)\n"
                      << "  --models-dir <path>  Models directory (default: ./models)\n"
                      << "  --gpu-layers <num>   GPU layers to offload (default: -1 = all)\n"
                      << "  --threads <num>      Number of CPU threads (default: auto)\n"
                      << "  --no-mmap            Disable mmap\n"
                      << "  --max-loaded-models <num>  Max loaded models before LRU hot-swap (default: 2)\n"
                      << "  --tensor-split <a,b,..>    Multi-GPU split ratios (default: none)\n"
                      << "  --split-mode <0-3>    0=none 1=layer 2=row 3=tensor (default: 1)\n"
                      << "  --tools <all|a,b,..> Built-in agent tools (default: all)\n"
                      << "  --no-tools           Disable built-in tools\n"
                      << "  --agent / --no-agent Agent loop for chat tools (default: on)\n"
                      << "  --mcp / --no-mcp     MCP servers (default: on; autoload ./mcp.json)\n"
                      << "  --mcp-servers-config <path>  MCP servers JSON (implies --mcp)\n"
                      << "  --tools-root <dir>   Sandbox root for file/shell tools (default: .)\n"
                      << "  --max-agent-steps <n>  Agent loop cap (default: 8)\n"
                      << "  -m, --model <path>   Register + preload model at startup (repeatable)\n"
                      << "  --api-key <key>      Seed fixed API key (in addition to BARSKUY_API_KEY)\n"
                      << "  --jinja / --perf / --no-context-shift / --no-warmup  Accepted (native behavior)\n"
                      << "\n[Text generation]\n"
                      << "  --n-cpu-moe <n>      Batch threads for MoE CPU (default: 1)\n"
                      << "  --flash-attn <on|off|auto>  KV flash attention (default: auto)\n"
                      << "  --batch-size <n>     Physical batch (default: 2048)\n"
                      << "  --ubatch-size <n>    Micro batch (default: 1024)\n"
                      << "  --cache-type-k/v <f16|q8_0|q4_0|auto>  KV cache type (default: auto)\n"
                      << "  --no-mmap (default) / --mmap  Memory-map model file\n"
                      << "  --prio/--prio-batch <0-4>  Process priority (default: 3)\n"
                      << "  --np <n>             Parallel sequences (default: 1)\n"
                      << "  -ctk, --context-size <n>  Effective context cap (default: 8192, auto-compact at 85%)\n"
                      << "\n[Image generation]  (ComfyUI preset: 512x512, 9 steps, cfg 1.0, res_multistep/simple, flow 3.0)\n"
                      << "  --image-model <path>      Default image model (default: auto-pick)\n"
                      << "  --image-width <n>         Width (default: 512)\n"
                      << "  --image-height <n>        Height (default: 512)\n"
                      << "  --image-steps <n>         Steps (default: 9)\n"
                      << "  --image-cfg <f>           CFG scale (default: 1.0)\n"
                      << "  --image-sampler <name>    Sampler: euler_a, euler, dpmpp2m, res_multistep, lcm, tcd...\n"
                      << "  --image-scheduler <name>  Scheduler: simple, karras, exponential, discrete...\n"
                      << "  --image-flow-shift <f>    Flow shift / AuraFlow shift (default: 3.0)\n"
                      << "  --image-cpu <n>           CPU threads for image (default: 6)\n"
                      << "\n[Video generation]\n"
                      << "  --video-model <path>      Default video model\n"
                      << "  --video-width <n>         Width (default: 512)\n"
                      << "  --video-height <n>        Height (default: 320)\n"
                      << "  --video-steps <n>         Steps (default: 20)\n"
                      << "  --video-frames <n>        Frames (default: 8)\n"
                      << "  --video-cpu <n>           CPU threads for video (default: 6)\n"
                      << "\n[Env]\n"
                      << "  BARSKUY_LOG_FILE=<path>  Append server log to file (daemon mode)\n"
                      << "  --help, -h           Show this help\n";
            std::exit(0);
        }
    }
}

} // namespace barskuy::core