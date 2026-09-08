#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <ggml.h>
#include <gguf.h>
// Forward declare llama.cpp types to avoid heavy include in header
struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_adapter_lora;

namespace barskuy::registry {
    class ArchitectureRegistry;
}

namespace barskuy::engine {

enum class ModelFormat {
    GGUF,
    SafeTensors
};

struct ModelConfig {
    int n_vocab = 0;
    int n_ctx = 0;        // effective (capped by --context-size)
    int n_ctx_train = 0;  // native from file (F9-10 dialog)
    int n_embd = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int n_rot = 0;
    float f_norm_eps = 1e-5f;
    float f_rope_freq_base = 10000.0f;
    float f_rope_freq_scale = 1.0f;
    int n_ff = 0;
    std::string architecture;
    std::string quantization;
};

struct LoadedModel {
    std::string id;
    std::string model_id;
    std::string file_path;
    ModelFormat format = ModelFormat::GGUF;
    ModelConfig config;
    struct ggml_context* ctx = nullptr;
    struct gguf_context* gguf_ctx = nullptr;
    struct ggml_cgraph* graph = nullptr;
    std::unordered_map<std::string, struct ggml_tensor*> tensors;
    // Safetensors: canonical ggml name -> safetensors tensor name
    std::unordered_map<std::string, std::string> tensor_name_map;
    // llama.cpp real inference (GGUF)
    struct llama_model* lmodel = nullptr;
    struct llama_context* lctx = nullptr;
    bool loaded = false;
    int n_gpu_layers_used = 0; // F9-10 actual offload (dialog)
    int64_t loaded_at = 0;
    int64_t last_used_at = 0;  // LRU hot-swap tracking (F1-8)
    size_t mem_size = 0;
    // LoRA adapters loaded on this model: adapter_id -> adapter (F3-9)
    std::unordered_map<std::string, struct llama_adapter_lora*> lora_adapters;
};

// Per-request generation options (guided decoding, LoRA, speculative)
struct GenerationOptions {
    std::string grammar;            // Raw GBNF grammar (F3-8); empty = disabled
    std::string grammar_root = "root";
    bool json_mode = false;         // response_format type=json_object -> builtin JSON grammar
    std::vector<std::pair<std::string, float>> lora_adapters;  // (adapter_id, scale) (F3-9)
    std::string draft_model;        // draft model id for speculative decoding (F3-7)
    int draft_tokens = 5;           // max draft tokens per round
    bool enable_thinking = true;    // F9-10: false = tutup prefill <think> (Qwen3 jawab langsung)
    // Sampling parity with llama.cpp WebUI settings (F9-8, all optional)
    int top_k = 0;                  // 0 = disabled
    float min_p = 0.0f;             // 0 = disabled
    int repeat_last_n = 0;          // 0 = disabled with default penalties
    float repeat_penalty = 1.0f;    // 1 = disabled
    float presence_penalty = 0.0f;  // 0 = disabled
    float frequency_penalty = 0.0f; // 0 = disabled
    int seed = 0;                   // 0 = random
    int mirostat = 0;               // 0 = disabled, 2 = v2
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;
    float dry_multiplier = 0.0f;    // 0 = disabled
    float dry_base = 1.75f;
    int dry_allowed_length = 2;
    int dry_penalty_last_n = -1;    // -1 = full context window below
};

class TextEngine {
public:
    TextEngine();
    ~TextEngine();

    bool initialize(const std::string& models_dir = "./models");
    void shutdown();

    // Load model from file (auto-detect format by extension)
    bool load_model(const std::string& model_id, const std::string& file_path);
    
    // Explicit format loading
    bool load_gguf_model(const std::string& model_id, const std::string& file_path);
    bool load_safetensors_model(const std::string& model_id, const std::string& file_path);
    
    bool unload_model(const std::string& model_id);
    bool is_loaded(const std::string& model_id) const;

    struct CompletionResult {
        std::string text;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        std::string finish_reason;
        double prompt_ms = 0.0;     // F9-10 prefill time (sync UI+log)
        double predicted_ms = 0.0;  // F9-10 decode time
    };

    struct Timings {
        int prompt_n = 0;
        double prompt_ms = 0.0;
        int predicted_n = 0;
        double predicted_ms = 0.0;
    };

    // Raw-prompt completion (no chat template) for the agent loop (F7)
    CompletionResult complete_raw(
        const std::string& model_id,
        const std::string& prompt,
        int max_tokens = 512,
        float temperature = 0.7f,
        float top_p = 0.9f,
        const std::vector<std::string>& stop = {},
        const GenerationOptions& opts = {}
    );

    // Raw llama_model pointer for chat-template rendering (agent loop, F7).
    // nullptr if not loaded / not a llama model. Pointer stays valid while loaded.
    struct llama_model* get_llama_model(const std::string& model_id) const;

    // Non-streaming completion
    CompletionResult complete(
        const std::string& model_id,
        const std::vector<std::pair<std::string, std::string>>& messages,
        int max_tokens = 512,
        float temperature = 0.7f,
        float top_p = 0.9f,
        const std::vector<std::string>& stop = {},
        const GenerationOptions& opts = {}
    );

    // Streaming completion - callback for each token; returns timings (F9-10)
    using StreamCallback = std::function<void(const std::string& token, bool is_final)>;
    // Fired once after prefill with exact prompt size (live gauge, F9-10).
    // think_prefilled = prompt diakhiri <think> terbuka (Qwen3): output
    // mulai dalam mode think tanpa opener (BUG-049).
    using ProgressCallback = std::function<void(int prompt_n, double prompt_ms,
                                               bool think_prefilled)>;
    Timings complete_stream(
        const std::string& model_id,
        const std::vector<std::pair<std::string, std::string>>& messages,
        int max_tokens,
        float temperature,
        float top_p,
        const std::vector<std::string>& stop,
        StreamCallback callback,
        const GenerationOptions& opts = {},
        ProgressCallback progress = {}
    );

    // Streaming raw-prompt completion (tools passthrough live, F9-10)
    Timings complete_stream_raw(
        const std::string& model_id,
        const std::string& prompt,
        int max_tokens,
        float temperature,
        float top_p,
        const std::vector<std::string>& stop,
        StreamCallback callback,
        const GenerationOptions& opts = {},
        ProgressCallback progress = {}
    );

    // Hot-swap (F1-8): max simultaneously loaded models (LRU eviction)
    void set_max_loaded_models(int n);
    int max_loaded_models() const;

    // Tensor parallelism (F3-6): split ratios per device + split mode (llama split_mode enum)
    void set_tensor_split(const std::vector<float>& splits);
    void set_split_mode(int mode);

    // Batching + KV cache (F9-9 defaults: batch 2048, ubatch 1024, cache auto)
    void set_n_batch(int n);
    void set_n_ubatch(int n);
    void set_n_threads_batch(int n);
    void set_cache_types(const std::string& k, const std::string& v);
    void set_use_mmap(bool on);
    void set_context_cap(int n); // F9-10 -ctk/--context-size (default 8192)

    // LoRA adapter serving (F3-9)
    bool load_lora_adapter(const std::string& model_id, const std::string& adapter_id, const std::string& path);
    bool unload_lora_adapter(const std::string& model_id, const std::string& adapter_id);
    std::vector<std::string> list_lora_adapters(const std::string& model_id) const;

    // Embeddings
    std::vector<std::vector<float>> embed(
        const std::string& model_id,
        const std::vector<std::string>& inputs
    );

    struct ModelInfo {
        std::string id;
        std::string model_id;
        ModelConfig config;
        ModelFormat format;
        bool loaded;
        int n_gpu_layers_used = 0; // F9-10 dialog
        int64_t loaded_at;
        size_t mem_size;
    };

    std::vector<ModelInfo> list_loaded_models() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Output hygiene shared with the API layer (F9-10): strip echoed assistant
// headers, blank jungles and tool-call markup from texts.
std::string clean_output_text(std::string s);
std::string strip_tool_markup(std::string s);

} // namespace barskuy::engine