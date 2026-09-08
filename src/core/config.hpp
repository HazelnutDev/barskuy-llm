#pragma once

#include <string>
#include <optional>
#include <vector>

namespace barskuy::core {

class Config {
public:
    Config() = default;
    ~Config() = default;

    bool load(int argc, char* argv[]);
    bool load_from_file(const std::string& path);

    const std::string& host() const { return host_; }
    uint16_t port() const { return port_; }
    const std::string& database_path() const { return database_path_; }
    const std::string& models_dir() const { return models_dir_; }
    size_t max_context_size() const { return max_context_size_; }
    size_t max_batch_size() const { return max_batch_size_; }
    int gpu_layers() const { return gpu_layers_; }
    bool enable_mmap() const { return enable_mmap_; }
    int num_threads() const { return num_threads_; }
    int max_loaded_models() const { return max_loaded_models_; }
    const std::vector<float>& tensor_split() const { return tensor_split_; }
    int split_mode() const { return split_mode_; }
    // Agent & tools (F7): tools default "all", agent loop default on.
    // Disable with --no-tools / --no-agent / --no-mcp.
    const std::string& tools_mode() const { return tools_mode_; }
    bool agent_enabled() const { return agent_enabled_; }
    const std::string& mcp_config() const { return mcp_config_; }
    bool mcp_enabled() const { return mcp_enabled_; }
    const std::string& tools_root() const { return tools_root_; }
    int max_agent_steps() const { return max_agent_steps_; }
    // WebUI-style startup (Fase 8, ala llama-server -m/--api-key)
    const std::vector<std::string>& preload_models() const { return preload_models_; }
    const std::string& api_key() const { return api_key_; }
    // Perf/context defaults (F9-9, aktif tanpa flags)
    int n_batch() const { return n_batch_; }
    int n_ubatch() const { return n_ubatch_; }
    int n_cpu_moe() const { return n_cpu_moe_; }
    const std::string& flash_attn() const { return flash_attn_; }
    const std::string& cache_type_k() const { return cache_type_k_; }
    const std::string& cache_type_v() const { return cache_type_v_; }
    int prio() const { return prio_; }
    int prio_batch() const { return prio_batch_; }
    int np() const { return np_; }
    // Image
    int image_width() const { return image_width_; }
    int image_height() const { return image_height_; }
    int image_steps() const { return image_steps_; }
    float image_cfg() const { return image_cfg_; }
    const std::string& image_sampler() const { return image_sampler_; }
    const std::string& image_scheduler() const { return image_scheduler_; }
    float image_flow_shift() const { return image_flow_shift_; }
    int image_cpu() const { return image_cpu_; }
    const std::string& image_model() const { return image_model_; }
    // Video
    int video_width() const { return video_width_; }
    int video_height() const { return video_height_; }
    int video_steps() const { return video_steps_; }
    int video_frames() const { return video_frames_; }
    int video_cpu() const { return video_cpu_; }
    const std::string& video_model() const { return video_model_; }

private:
    void set_defaults();
    void parse_args(int argc, char* argv[]);
    void parse_env();

    std::string host_ = "0.0.0.0";
    uint16_t port_ = 8080;
    std::string database_path_ = "barskuy.db";
    std::string models_dir_ = "./models";
    size_t max_context_size_ = 8192; // F9-10 -ctk/--context-size default
    size_t max_batch_size_ = 512;
    int gpu_layers_ = -1;
    bool enable_mmap_ = false; // F9-9: --no-mmap default aktif; nyalakan via --mmap
    int num_threads_ = 0;
    int max_loaded_models_ = 2;       // F1-8 hot-swap budget
    std::vector<float> tensor_split_; // F3-6 e.g. "0.6,0.4"
    int split_mode_ = 1;              // LLAMA_SPLIT_MODE_LAYER default
    std::string tools_mode_ = "all";  // F7: all (default) | comma list | "" (off)
    bool agent_enabled_ = true;       // F7 agent loop (default on)
    std::string mcp_config_;          // F7: mcp.json path ("" = autoload ./mcp.json)
    bool mcp_enabled_ = true;         // F7 MCP stdio (default on; dormant bila tak ada mcp.json)
    std::string tools_root_ = ".";    // F7 sandbox root for file/shell tools
    int max_agent_steps_ = 8;         // F7 agent loop cap
    std::vector<std::string> preload_models_; // Fase 8: --model (repeatable)
    std::string api_key_;             // Fase 8: --api-key fixed key seed
    int n_batch_ = 2048;              // F9-9 --batch-size
    int n_ubatch_ = 1024;             // F9-9 --ubatch-size
    int n_cpu_moe_ = 1;               // F9-10 --n-cpu-moe (batch threads; 1 = serial
                                        // prefill, works around multi-threaded prefill wedge)
    std::string flash_attn_ = "auto"; // F9-9 on|off|auto (bundle-managed)
    std::string cache_type_k_ = "auto"; // F9-9 f16|q8_0|q4_0|auto
    std::string cache_type_v_ = "auto";
    int prio_ = 3;                    // F9-9 process priority 0-4
    int prio_batch_ = 3;
    int np_ = 1;                      // F9-9 parallel sequences
    // Image defaults (F10, ComfyUI preset; override via --image-* or JSON per-request)
    int image_width_ = 512;
    int image_height_ = 512;
    int image_steps_ = 9;
    float image_cfg_ = 1.0f;
    std::string image_sampler_ = "res_multistep";
    std::string image_scheduler_ = "simple";
    float image_flow_shift_ = 3.0f;
    int image_cpu_ = 6;
    std::string image_model_ = "";
    // Video defaults (F10)
    int video_width_ = 512;
    int video_height_ = 320;
    int video_steps_ = 20;
    int video_frames_ = 8;
    int video_cpu_ = 6;
    std::string video_model_ = "";
};

} // namespace barskuy::core