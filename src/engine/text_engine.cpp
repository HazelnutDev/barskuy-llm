#include "text_engine.hpp"
#include "core/logger.hpp"
#include "utils/utils.hpp"
#include "safetensors_loader.hpp"
#include "registry/architecture_registry.hpp"
#include <ggml.h>
#include <gguf.h>
#include <ggml-backend.h>
#include "llama.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <random>
#include <chrono>
#include <filesystem>
#include <unordered_set>

// C++17 compatible ends_with
static bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

namespace barskuy::engine {

// F9-10 shared output hygiene (used by Impl methods and API layer).
static std::string clean_prefix(std::string s) {
    size_t p = s.find_first_not_of(" \t\r\n");
    if (p == std::string::npos) return "";
    s.erase(0, p);
    const char* hdr = "<|im_start|>assistant";
    size_t hl = strlen(hdr);
    if (s.compare(0, hl, hdr) == 0) {
        s.erase(0, hl);
        p = s.find_first_not_of(" \t\r\n");
        s = (p == std::string::npos) ? "" : s.substr(p);
    }
    return s;
}
static std::string clean_output(std::string s) {
    s = clean_prefix(s);
    std::string o;
    o.reserve(s.size());
    int nl = 0;
    for (char c : s) {
        if (c == '\n') { if (++nl <= 2) o += c; }
        else { nl = 0; o += c; }
    }
    while (!o.empty() && (o.back() == ' ' || o.back() == '\t' ||
                          o.back() == '\r' || o.back() == '\n'))
        o.pop_back();
    return o;
}
std::string clean_output_text(std::string s) { return clean_output(std::move(s)); }
std::string strip_tool_markup(std::string s) {
    for (;;) {
        size_t p = s.find("<tool_call>");
        if (p == std::string::npos) break;
        size_t e = s.find("</tool_call>", p);
        s.erase(p, e == std::string::npos ? std::string::npos : e - p + 12);
    }
    for (;;) {
        size_t p = s.find("<function=");
        if (p == std::string::npos) break;
        size_t e = s.find("</function>", p);
        s.erase(p, e == std::string::npos ? std::string::npos : e - p + 11);
    }
    return clean_output(std::move(s));
}

} // namespace barskuy::engine

namespace barskuy::engine {

struct TextEngine::Impl {
    std::string models_dir_;
    std::unordered_map<std::string, LoadedModel> loaded_models_;
    // F9-10: last-used context params per model for per-request fresh contexts
    std::unordered_map<std::string, ::llama_context_params> used_cparams_;
    mutable std::recursive_mutex models_mutex_;
    struct ggml_context* backend_ctx_ = nullptr;
    struct ggml_backend* backend_ = nullptr;
    std::unique_ptr<barskuy::registry::ArchitectureRegistry> arch_registry_;
    int max_loaded_models_ = 2;              // F1-8 hot-swap budget
    std::vector<float> tensor_split_;        // F3-6 multi-GPU ratios
    int split_mode_ = 0;                     // llama_split_mode enum value
    int n_batch_ = 2048;                     // F9-9 --batch-size default
    int n_ubatch_ = 1024;                    // F9-9 --ubatch-size default
    int n_threads_batch_ = 0;                // F9-9 --n-cpu-moe (0 = = n_threads)
    std::string cache_type_k_ = "auto";      // F9-9 f16|q8_0|q4_0|auto
    std::string cache_type_v_ = "auto";
    bool use_mmap_ = false;                  // F9-9 --no-mmap default aktif
    int context_cap_ = 8192;                 // F9-10 -ctk/--context-size
    // F9-10 BUG-045: 512-token prefill chunks wedge llama_decode
    // intermittently (upstream race, CPU+Vulkan, any model/quant/KV-type).
    // 128-token chunks verified 72/72 on the raciest config (16/16 threads).
    static constexpr int kPrefillChunk = 128;  // F3 chunked prefill

    // Builtin JSON grammar for response_format json_object (F3-8, GBNF)
    static const char* json_grammar() {
        return R"GBNF(
root ::= object
value ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ \t\n] ws)?
)GBNF";
    }

    int64_t now_secs() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void touch(LoadedModel& model) { model.last_used_at = now_secs(); }

    // F1-8: evict least-recently-used models when over budget
    void evict_if_needed(const std::string& except_id) {
        while ((int)loaded_models_.size() >= max_loaded_models_ && max_loaded_models_ > 0) {
            std::string lru_id;
            int64_t oldest = INT64_MAX;
            for (auto& [id, m] : loaded_models_) {
                if (id == except_id) continue;
                if (m.last_used_at < oldest) { oldest = m.last_used_at; lru_id = id; }
            }
            if (lru_id.empty()) break;
            core::Logger::info("Hot-swap: evicting LRU model {}", lru_id);
            auto it = loaded_models_.find(lru_id);
            if (it != loaded_models_.end()) {
                unload_model_internal(it->second);
                loaded_models_.erase(it);
            } else break;
        }
    }

    struct SamplingOpts {
        int top_k = 0;
        float min_p = 0.0f;
        int pen_last_n = 0;
        float pen_repeat = 1.0f;
        float pen_freq = 0.0f;
        float pen_present = 0.0f;
        uint32_t seed = 0;
        int mirostat = 0;
        float mirostat_tau = 5.0f;
        float mirostat_eta = 0.1f;
        float dry_multiplier = 0.0f;
        float dry_base = 1.75f;
        int dry_allowed_length = 2;
        int dry_last_n = -1;
    };
    // Build sampler chain with optional GBNF grammar constraint (F3-8).
    // Order mirrors llama.cpp: grammar, penalties, top_k, top_p, min_p, temp, dist.
    llama_sampler* make_sampler(const llama_vocab* vocab, float temp, float top_p,
                                const std::string& grammar, const std::string& grammar_root,
                                const SamplingOpts& so = {}) {
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler* smpl = llama_sampler_chain_init(sparams);
        if (!grammar.empty()) {
            const char* root = grammar_root.empty() ? "root" : grammar_root.c_str();
            llama_sampler* g = llama_sampler_init_grammar(vocab, grammar.c_str(), root);
            if (g) llama_sampler_chain_add(smpl, g);
            else core::Logger::warn("Invalid grammar, continuing unconstrained");
        }
        if (so.pen_repeat != 1.0f || so.pen_freq != 0.0f || so.pen_present != 0.0f) {
            int last_n = so.pen_last_n < 0 ? 4096 : (so.pen_last_n != 0 ? so.pen_last_n : 64);
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                llama_vocab_n_tokens(vocab), last_n,
                so.pen_repeat, so.pen_freq, so.pen_present));
        }
        if (so.top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(so.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (so.min_p > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(so.min_p, 1));
        if (so.mirostat == 2) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_mirostat_v2(so.seed, so.mirostat_tau, so.mirostat_eta));
        }
        if (so.dry_multiplier > 0.0f) {
            int dry_n = so.dry_last_n < 0 ? 4096 : so.dry_last_n;
            llama_sampler_chain_add(smpl, llama_sampler_init_dry(
                vocab, so.dry_multiplier, so.dry_base, so.dry_allowed_length, dry_n,
                nullptr, 0));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(so.seed));
        return smpl;
    }
    static SamplingOpts sampling_from_opts(const GenerationOptions& opts) {
        SamplingOpts so;
        so.top_k = opts.top_k; so.min_p = opts.min_p;
        so.pen_last_n = opts.repeat_last_n; so.pen_repeat = opts.repeat_penalty;
        so.pen_freq = opts.frequency_penalty; so.pen_present = opts.presence_penalty;
        so.seed = (uint32_t)opts.seed;
        so.mirostat = opts.mirostat; so.mirostat_tau = opts.mirostat_tau;
        so.mirostat_eta = opts.mirostat_eta;
        so.dry_multiplier = opts.dry_multiplier; so.dry_base = opts.dry_base;
        so.dry_allowed_length = opts.dry_allowed_length; so.dry_last_n = opts.dry_penalty_last_n;
        return so;
    }

    // F3 chunked prefill: decode long prompts in fixed-size chunks.
    // Null-pos batches (monotonic continuation) proved hang-free across
    // clear/generate cycles; explicit 0-based re-prefill hangs llama_decode
    // on reused contexts (F9-10 investigation) - keep null-pos here.
    bool decode_chunked(struct llama_context* lctx, const std::vector<llama_token>& toks) {
        size_t off = 0;
        while (off < toks.size()) {
            size_t n = std::min<size_t>(kPrefillChunk, toks.size() - off);
            core::Logger::info("chunk off={} n={} seqmax={}", off, n,
                (long long)llama_memory_seq_pos_max(llama_get_memory(lctx), 0));
            llama_batch batch = llama_batch_get_one(
                const_cast<llama_token*>(toks.data() + off), (int32_t)n);
            if (llama_decode(lctx, batch) != 0) return false;
            core::Logger::info("chunk done off={}", off);
            off += n;
        }
        return true;
    }

    // Explicit-position batch decode. Unlike null-pos batches (which continue
    // from memory->seq_pos_max and break across clear/trim cycles), explicit
    // positions are self-consistent: the caller tracks KV length per context.
    // Fills caller-provided storage which must outlive the decode call.
    struct ExplicitBatch {
        std::vector<llama_pos> pos;
        std::vector<int32_t> n_seq;
        std::vector<llama_seq_id> seq0;
        std::vector<llama_seq_id*> seqs;
        std::vector<int8_t> logits;
    };
    bool decode_explicit(struct llama_context* lctx, const llama_token* toks, int n,
                         int start_pos, bool want_logits, ExplicitBatch& st) {
        st.pos.resize(n);
        st.n_seq.resize(n);
        st.seq0.assign(n, 0);
        st.seqs.resize(n);
        // Always materialize last-position logits (llama_batch_get_one semantics):
        // callers sample idx -1 right after decode. want_logits=true adds all positions.
        st.logits.assign(n, want_logits ? 1 : 0);
        if (n > 0) st.logits[n - 1] = 1;
        for (int i = 0; i < n; ++i) {
            st.pos[i] = (llama_pos)(start_pos + i);
            st.n_seq[i] = 1;
            st.seqs[i] = st.seq0.data() + i;
        }
        llama_batch b;
        b.n_tokens = (int32_t)n;
        b.token = const_cast<llama_token*>(toks);
        b.embd = nullptr;
        b.pos = st.pos.data();
        b.n_seq_id = st.n_seq.data();
        b.seq_id = st.seqs.data();
        b.logits = st.logits.data();
        return llama_decode(lctx, b) == 0;
    }

    // F3-9: apply requested LoRA adapters to a context; returns count applied
    int apply_lora_adapters(LoadedModel& model, const std::vector<std::pair<std::string,float>>& req) {
        if (req.empty()) return 0;
        std::vector<llama_adapter_lora*> adapters;
        std::vector<float> scales;
        for (auto& [aid, scale] : req) {
            auto it = model.lora_adapters.find(aid);
            if (it == model.lora_adapters.end()) {
                core::Logger::warn("LoRA adapter '{}' not loaded on model {}", aid, model.model_id);
                continue;
            }
            adapters.push_back(it->second);
            scales.push_back(scale);
        }
        if (adapters.empty()) return 0;
        if (llama_set_adapters_lora(model.lctx, adapters.data(), adapters.size(), scales.data()) != 0) {
            core::Logger::warn("llama_set_adapters_lora failed");
            return 0;
        }
        return (int)adapters.size();
    }

    void clear_lora_adapters(LoadedModel& model) {
        llama_set_adapters_lora(model.lctx, nullptr, 0, nullptr);
    }

    Impl() = default;
    ~Impl() {
        shutdown();
    }

    void shutdown() {
        std::lock_guard<std::recursive_mutex> lock(models_mutex_);
        for (auto& [id, model] : loaded_models_) {
            unload_model_internal(model);
        }
        loaded_models_.clear();

        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
        if (backend_ctx_) {
            ggml_free(backend_ctx_);
            backend_ctx_ = nullptr;
        }
        llama_backend_free();
    }

    bool initialize_backend() {
        // llama.cpp backend handles ggml backends + optimizations
        try {
            ggml_backend_load_all();
            llama_backend_init();
        } catch (...) {
            core::Logger::warn("llama_backend_init failed, continuing without GPU backends");
        }
        
        if (backend_) return true;

        backend_ = ggml_backend_init_by_name("CPU", nullptr);
        if (!backend_) {
            // not fatal - llama will still work
            core::Logger::warn("CPU backend not available, using llama defaults");
        } else {
            struct ggml_init_params params = {
                1024 * 1024 * 1024, nullptr, false,
            };
            backend_ctx_ = ggml_init(params);
        }

        // Load Architecture Registry
        arch_registry_ = std::make_unique<barskuy::registry::ArchitectureRegistry>();
        
        // Try multiple locations for architectures directory
        std::vector<std::string> arch_dirs = {
            models_dir_ + "/../architectures",
            "./architectures",
            "../architectures",
            "../../architectures",
            "architectures"
        };
        
        bool loaded = false;
        for (const auto& arch_dir : arch_dirs) {
            if (std::filesystem::exists(arch_dir)) {
                try {
                    arch_registry_->load_from_directory(arch_dir);
                    loaded = true;
                    core::Logger::info("Architecture registry loaded from: {}", arch_dir);
                } catch (...) {
                    core::Logger::warn("Failed to load architecture registry from: {}", arch_dir);
                }
                break;
            }
        }
        
        if (!loaded) {
            core::Logger::warn("Architecture directory not found in any expected location");
        }
        
        core::Logger::info("Architecture registry loaded with {} architectures", 
            arch_registry_ ? arch_registry_->list_architectures().size() : 0);

        return true;
    }

    bool load_gguf_model(const std::string& file_path, LoadedModel& model) {
        core::Logger::info("Loading GGUF model from: {}", file_path);

        model.file_path = file_path;

        // Hybrid CUDA+Vulkan+CPU+iGPU - single binary, auto VRAM combo, anti-OOM
        core::Logger::info("Loading GGUF via llama.cpp: {}", file_path);
        llama_model_params mparams = llama_model_default_params();
        mparams.load_mode = use_mmap_ ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE; // F9-9: default off (--no-mmap)
        // F3-6 tensor parallelism: multi-GPU split ratios + split mode
        if (!tensor_split_.empty()) {
            mparams.tensor_split = tensor_split_.data();
            core::Logger::info("Using tensor_split with {} devices", tensor_split_.size());
        }
        if (split_mode_ >= 0 && split_mode_ <= 3) {
            mparams.split_mode = (enum llama_split_mode)split_mode_;
        }
        // Auto-detect GPU backends (CUDA/Vulkan) for optimal throughput, fallback CPU
        bool has_gpu = llama_supports_gpu_offload();
        // Check available backends for hybrid combo (dGPU + iGPU + CPU)
        int backend_count = 0;
        try { backend_count = ggml_backend_reg_count(); } catch(...) { backend_count = has_gpu ? 1 : 0; }
        core::Logger::info("Backends detected: {} (has_gpu={}) - hybrid CPU+GPU enabled", backend_count, has_gpu);
        // Multi-GPU / iGPU: tensor_split auto (llama handles), expert offload via offload_kqv
        // Start with max offload, fallback to CPU if VRAM insufficient (OOM-safe)
        int try_layers[] = {99, 35, 20, 0};
        model.lmodel = nullptr;
        for (int layers : try_layers) {
            if (!has_gpu && layers != 0) continue;
            mparams.n_gpu_layers = layers;
            // For 2+ GPUs, enable NCCL-less tensor split (requires Vulkan/CUDA)
            // llama will auto-split across devices when n_gpu_layers>0 and multiple backends
            core::Logger::info("Trying n_gpu_layers={} (VRAM-aware, PCIe efficient)", layers);
            model.lmodel = llama_model_load_from_file(file_path.c_str(), mparams);
            if (model.lmodel) {
                core::Logger::info("Loaded with n_gpu_layers={} (hybrid success)", layers);
                model.n_gpu_layers_used = layers;
                break;
            }
            core::Logger::warn("Failed n_gpu_layers={}, retrying lower (anti-OOM)", layers);
        }
        if (!model.lmodel) {
            core::Logger::error("llama_model_load_from_file failed after all retries");
            return false;
        }
        // Keep GGUF ctx for metadata compatibility (optional)
        struct gguf_init_params gguf_params = {true, nullptr};
        // Use separate GGUF read for metadata only (no_alloc)
        model.gguf_ctx = gguf_init_from_file(file_path.c_str(), gguf_params);
        if (model.gguf_ctx) {
            read_model_metadata(model);
            // override vocab from llama vocab for accuracy
            const llama_vocab* vocab = llama_model_get_vocab(model.lmodel);
            int n_vocab = llama_vocab_n_tokens(vocab);
            if (n_vocab > 0) model.config.n_vocab = n_vocab;
            gguf_free(model.gguf_ctx);
            model.gguf_ctx = nullptr;
        } else {
            const llama_vocab* vocab = llama_model_get_vocab(model.lmodel);
            model.config.n_vocab = llama_vocab_n_tokens(vocab);
            model.config.architecture = "qwen2";
        }

        // Hybrid optimal: 40-80 tok/s, hemat VRAM, prefill/decode/stream super cepat
        llama_context_params cparams = llama_context_default_params();
        // Real embeddings need mean pooling; embeddings outputs are toggled
        // per-request via llama_set_embeddings so generation sampler layout is untouched
        cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        if (model.config.n_ctx > 0) model.config.n_ctx_train = model.config.n_ctx;
        else model.config.n_ctx_train = 4096;
        // F9-10: effective ctx = min(train, -ctk cap); default cap 8192
        model.config.n_ctx = std::min(model.config.n_ctx_train, context_cap_);
        cparams.n_ctx = model.config.n_ctx;
        cparams.n_batch = n_batch_; // F9-9 --batch-size (default 2048)
        cparams.n_ubatch = n_ubatch_; // F9-9 --ubatch-size (default 1024)
        cparams.n_threads = (int)std::thread::hardware_concurrency(); if (cparams.n_threads < 4) cparams.n_threads = 4;
        cparams.n_threads_batch = n_threads_batch_ > 0 ? n_threads_batch_ : cparams.n_threads;
        // F9-9 KV cache types: auto = q8_0 for quantized weights, f16 otherwise
        auto resolve_cache = [&](const std::string& want) {
            if (want == "q8_0") return GGML_TYPE_Q8_0;
            if (want == "q4_0") return GGML_TYPE_Q4_0;
            if (want == "f16") return GGML_TYPE_F16;
            const std::string& q = model.config.quantization;
            if (q.empty() || q.find("F16") != std::string::npos ||
                q.find("F32") != std::string::npos || q.find("BF16") != std::string::npos)
                return GGML_TYPE_F16;
            return GGML_TYPE_Q8_0;
        };
        cparams.type_k = resolve_cache(cache_type_k_);
        cparams.type_v = resolve_cache(cache_type_v_);
        // flash_attn auto in this llama version (field removed)
        cparams.offload_kqv = true; // keep KV di VRAM, expert offload ke RAM/iGPU

        model.lctx = llama_init_from_model(model.lmodel, cparams);
        if (!model.lctx) {
            core::Logger::error("llama_init_from_model failed");
            llama_model_free(model.lmodel);
            model.lmodel = nullptr;
            return false;
        }
        // F9-10: remember exact cparams for per-request fresh contexts
        used_cparams_[model.model_id] = cparams;

        model.loaded = true;
        try { model.mem_size = std::filesystem::file_size(file_path); } catch(...) { model.mem_size = 0; }
        model.loaded_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        core::Logger::info("Model loaded via llama.cpp: {} vocab={} ctx={} ({} MB)", model.model_id, model.config.n_vocab, cparams.n_ctx, model.mem_size / (1024*1024));
        return true;
    }

    bool read_model_metadata(LoadedModel& model) {
        auto& config = model.config;

        int64_t key_id;

        key_id = gguf_find_key(model.gguf_ctx, "general.architecture");
        if (key_id < 0) {
            core::Logger::error("Missing general.architecture in GGUF");
            return false;
        }
        config.architecture = gguf_get_val_str(model.gguf_ctx, key_id);
        std::string arch = config.architecture;
        // helper to try arch prefix, then fallback.
        // Scalar getters assert on array keys (e.g. bailingmoe3 head_count_kv
        // is arr[i32,24]), so read arrays element-wise. For head counts the
        // max over layers is the meaningful value (0 = recurrent/SSM layer).
        auto find_u32 = [&](const std::string& suffix, uint32_t& out) -> bool {
            std::vector<std::string> prefixes = {arch, "llama", "qwen2", "qwen3", "gemma", "mistral", "general"};
            for (auto& p : prefixes) {
                std::string key = p + "." + suffix;
                int64_t kid = gguf_find_key(model.gguf_ctx, key.c_str());
                if (kid < 0) continue;
                if (gguf_get_kv_type(model.gguf_ctx, kid) == GGUF_TYPE_ARRAY) {
                    enum gguf_type at = gguf_get_arr_type(model.gguf_ctx, kid);
                    size_t n = gguf_get_arr_n(model.gguf_ctx, kid);
                    if (n == 0) continue;
                    const void* d = gguf_get_arr_data(model.gguf_ctx, kid);
                    int64_t mx = 0;
                    if (at == GGUF_TYPE_UINT32) {
                        for (size_t i = 0; i < n; ++i) mx = std::max<int64_t>(mx, ((const uint32_t*)d)[i]);
                    } else if (at == GGUF_TYPE_INT32) {
                        for (size_t i = 0; i < n; ++i) mx = std::max<int64_t>(mx, ((const int32_t*)d)[i]);
                    } else continue;
                    out = (uint32_t)mx;
                    return true;
                }
                out = gguf_get_val_u32(model.gguf_ctx, kid);
                return true;
            }
            return false;
        };
        auto find_f32 = [&](const std::string& suffix, float& out) -> bool {
            std::vector<std::string> prefixes = {arch, "llama", "qwen2", "qwen3", "gemma", "general"};
            for (auto& p : prefixes) {
                std::string key = p + "." + suffix;
                int64_t kid = gguf_find_key(model.gguf_ctx, key.c_str());
                if (kid < 0) continue;
                if (gguf_get_kv_type(model.gguf_ctx, kid) == GGUF_TYPE_ARRAY) continue;
                out = gguf_get_val_f32(model.gguf_ctx, kid);
                return true;
            }
            return false;
        };

        uint32_t tmp;
        if (find_u32("vocab_size", tmp)) config.n_vocab = tmp;
        // Fallback for Qwen/Gemma where vocab is in tokenizer.ggml.tokens array
        if (config.n_vocab == 0) {
            int64_t tid = gguf_find_key(model.gguf_ctx, "tokenizer.ggml.tokens");
            if (tid >= 0) {
                size_t n = gguf_get_arr_n(model.gguf_ctx, tid);
                if (n > 0) config.n_vocab = (int)n;
            }
        }
        // Last fallback: infer from token_embd shape
        if (config.n_vocab == 0) {
            int64_t t = gguf_find_tensor(model.gguf_ctx, "token_embd.weight");
            if (t >= 0) {
                const int64_t* ne = gguf_get_tensor_ne(model.gguf_ctx, t);
                config.n_vocab = (int)ne[0];
            }
        }

        // try common keys with arch prefix
        {
            int64_t kid = -1;
            // context_length
            if (find_u32("context_length", tmp)) config.n_ctx = tmp;
            if (find_u32("embedding_length", tmp)) config.n_embd = tmp;
            if (find_u32("block_count", tmp)) config.n_layer = tmp;
            if (find_u32("attention.head_count", tmp)) config.n_head = tmp;
            if (find_u32("attention.head_count_kv", tmp)) config.n_head_kv = tmp;
            else config.n_head_kv = config.n_head;
            if (find_u32("rope.dimension_count", tmp)) config.n_rot = tmp;
            else {
                if (config.n_head != 0) config.n_rot = config.n_embd / config.n_head;
                else config.n_rot = 0;
            }
            float ftmp;
            if (find_f32("attention.layer_norm_rms_epsilon", ftmp)) config.f_norm_eps = ftmp;
            else find_f32("attention.layer_norm_epsilon", config.f_norm_eps);
            if (find_f32("rope.freq_base", ftmp)) config.f_rope_freq_base = ftmp;
            if (find_u32("feed_forward_length", tmp)) config.n_ff = tmp;
            else if (config.n_embd) config.n_ff = 4 * config.n_embd;
        }

        int64_t n_tensors = gguf_get_n_tensors(model.gguf_ctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            enum ggml_type type = gguf_get_tensor_type(model.gguf_ctx, i);
            if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
                config.quantization = ggml_type_name(type);
                break;
            }
        }

        core::Logger::info("Model config: arch={}, vocab={}, ctx={}, embd={}, layers={}, heads={}, quantization={}",
            config.architecture, config.n_vocab, config.n_ctx, config.n_embd,
            config.n_layer, config.n_head, config.quantization);

        return true;
    }

    bool build_model_graph(LoadedModel& model) {
        // For now, skip heavy graph building in model.ctx which is fixed-size from GGUF.
        // Just verify required tensors exist; real graph will be built in a separate compute context.
        int64_t embed_w_id = gguf_find_tensor(model.gguf_ctx, "token_embd.weight");
        if (embed_w_id < 0) {
            core::Logger::error("Missing token_embd.weight tensor");
            return false;
        }
        // Don't allocate new tensors in model.ctx to avoid GGML_ASSERT OOM.
        // Placeholder: graph will be built lazily during inference with a dedicated context.
        model.graph = nullptr;
        core::Logger::info("Model graph check passed (placeholder, full inference uses separate context)");
        return true;
    }

    struct ggml_tensor* get_tensor(LoadedModel& model, const std::string& name) {
        auto it = model.tensors.find(name);
        if (it != model.tensors.end()) {
            return it->second;
        }

        if (!model.gguf_ctx) return nullptr;

        int64_t tensor_id = gguf_find_tensor(model.gguf_ctx, name.c_str());
        if (tensor_id < 0) {
            return nullptr;
        }

        const char* tensor_name = gguf_get_tensor_name(model.gguf_ctx, tensor_id);
        enum ggml_type type = gguf_get_tensor_type(model.gguf_ctx, tensor_id);
        const int64_t* ne = gguf_get_tensor_ne(model.gguf_ctx, tensor_id);

        struct ggml_tensor* tensor = ggml_new_tensor(model.ctx, type, ne[0], ne);
        ggml_set_name(tensor, tensor_name);

        model.tensors[name] = tensor;
        return tensor;
    }

    bool load_tensor_weights(LoadedModel& model) {
        if (!model.gguf_ctx) return false;
        // With gguf_init no_alloc=false, tensors are already memory-mapped in model.ctx.
        core::Logger::info("Tensors already loaded via gguf_init ({} tensors), skipping manual copy", gguf_get_n_tensors(model.gguf_ctx));
        return true;
    }

    // F9-10: fresh context per request (kills reused-context decode hangs).
    // Model weights stay loaded; only the mutable context is recreated.
    bool fresh_context(LoadedModel& model) {
        if (!model.lmodel) return false;
        auto cpit = used_cparams_.find(model.model_id);
        if (cpit == used_cparams_.end()) return false;
        if (model.lctx) {
            llama_free(model.lctx);
            model.lctx = nullptr;
        }
        model.lctx = llama_init_from_model(model.lmodel, cpit->second);
        if (!model.lctx) {
            core::Logger::error("fresh context init failed for {}", model.model_id);
            return false;
        }
        return true;
    }

    void unload_model_internal(LoadedModel& model) {
        for (auto& [aid, adapter] : model.lora_adapters) {
            if (adapter) llama_adapter_lora_free(adapter);
        }
        model.lora_adapters.clear();
        used_cparams_.erase(model.model_id);
        if (model.lctx) {
            llama_free(model.lctx);
            model.lctx = nullptr;
        }
        if (model.lmodel) {
            llama_model_free(model.lmodel);
            model.lmodel = nullptr;
        }
        if (model.gguf_ctx) {
            gguf_free(model.gguf_ctx);
            model.gguf_ctx = nullptr;
        }
        if (model.ctx) {
            ggml_free(model.ctx);
            model.ctx = nullptr;
        }
        if (model.graph) {
            model.graph = nullptr;
        }
        model.tensors.clear();
        model.tensor_name_map.clear();
        model.loaded = false;
    }

    bool load_safetensors_model(const std::string& file_path, LoadedModel& model) {
        core::Logger::info("Loading SafeTensors model from: {}", file_path);

        model.file_path = file_path;

        SafetensorsLoader loader;
        SafetensorsMetadata metadata;

        if (!loader.parse_header(file_path, metadata)) {
            core::Logger::error("Failed to parse SafeTensors header");
            return false;
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ 1024 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        model.ctx = ggml_init(params);
        if (!model.ctx) {
            core::Logger::error("Failed to create ggml context");
            return false;
        }

        if (!read_safetensors_config(file_path, model, metadata)) {
            core::Logger::error("Failed to read SafeTensors config");
            return false;
        }

        // Get tensor name mappings from Architecture Registry
        std::unordered_map<std::string, std::string> tensor_mappings;
        if (arch_registry_ && arch_registry_->has_architecture(model.config.architecture)) {
            tensor_mappings = arch_registry_->get_all_tensor_mappings(model.config.architecture, model.config.n_layer);
            core::Logger::info("Using ArchitectureRegistry mappings for '{}': {} tensors mapped", 
                model.config.architecture, tensor_mappings.size());
        } else {
            core::Logger::warn("Architecture '{}' not found in registry, using raw tensor names", model.config.architecture);
        }

        // Detect quantization format from tensor names
        bool has_quantized = false;
        for (const auto& [name, info] : metadata.tensors) {
            if (ends_with(name, ".qweight") || ends_with(name, ".qzeros") || ends_with(name, ".g_idx")) {
                has_quantized = true;
                break;
            }
        }

        if (!create_safetensors_tensors(model, metadata, tensor_mappings, has_quantized)) {
            core::Logger::error("Failed to create tensors");
            return false;
        }

        bool load_success = false;
        if (has_quantized) {
            core::Logger::info("Detected quantized model (AWQ/GPTQ), using dequantized loading");
            load_success = loader.load_tensors_dequantized(file_path, model.ctx, model.tensors, metadata, model.tensor_name_map);
        } else {
            load_success = loader.load_tensors(file_path, model.ctx, model.tensors, metadata, model.tensor_name_map);
        }

        if (!load_success) {
            core::Logger::error("Failed to load tensor weights");
            return false;
        }

        if (!build_model_graph(model)) {
            core::Logger::error("Failed to build model graph");
            return false;
        }

        model.loaded = true;
        model.mem_size = ggml_used_mem(model.ctx);
        model.loaded_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        core::Logger::info("SafeTensors model loaded successfully: {} ({} MB)", model.model_id, model.mem_size / (1024 * 1024));
        return true;
    }

    bool read_safetensors_config(const std::string& file_path, LoadedModel& model, const SafetensorsMetadata& metadata) {
        auto& config = model.config;

        std::filesystem::path model_path(file_path);
        std::filesystem::path config_path = model_path.parent_path() / "config.json";

        if (std::filesystem::exists(config_path)) {
            std::ifstream config_file(config_path);
            if (config_file.is_open()) {
                try {
                    nlohmann::json config_json;
                    config_file >> config_json;

                    if (config_json.contains("architectures") && config_json["architectures"].is_array() && !config_json["architectures"].empty()) {
                        config.architecture = config_json["architectures"][0].get<std::string>();
                    }

                    config.n_vocab = config_json.value("vocab_size", 0);
                    config.n_ctx = config_json.value("max_position_embeddings", 0);
                    config.n_embd = config_json.value("hidden_size", 0);
                    config.n_layer = config_json.value("num_hidden_layers", 0);
                    config.n_head = config_json.value("num_attention_heads", 0);
                    config.n_head_kv = config_json.value("num_key_value_heads", config.n_head);
                    config.n_rot = static_cast<int>(config_json.value("partial_rotary_factor", 1.0f) * config.n_embd / config.n_head);
                    config.f_norm_eps = config_json.value("rms_norm_eps", 1e-5f);
                    config.f_rope_freq_base = config_json.value("rope_theta", 10000.0f);
                    config.n_ff = config_json.value("intermediate_size", 4 * config.n_embd);

                    for (const auto& [name, info] : metadata.tensors) {
                        if (info.dtype != "F32" && info.dtype != "F16" && info.dtype != "BF16") {
                            config.quantization = info.dtype;
                            break;
                        }
                    }

                    core::Logger::info("Read config from config.json: arch={}, vocab={}, ctx={}, embd={}, layers={}", 
                        config.architecture, config.n_vocab, config.n_ctx, config.n_embd, config.n_layer);
                    return true;
                } catch (const std::exception& e) {
                    core::Logger::warn("Failed to parse config.json: {}", e.what());
                }
            }
        }

        config.architecture = infer_architecture_from_tensors(metadata);
        config.n_vocab = infer_vocab_size(metadata);
        config.n_embd = infer_hidden_size(metadata);
        config.n_layer = infer_num_layers(metadata);
        config.n_head = infer_num_heads(metadata);
        config.n_head_kv = config.n_head;
        config.n_rot = config.n_embd / config.n_head;
        config.n_ff = 4 * config.n_embd;

        core::Logger::info("Inferred config: arch={}, vocab={}, embd={}, layers={}", 
            config.architecture, config.n_vocab, config.n_embd, config.n_layer);

        return true;
    }

    std::string infer_architecture_from_tensors(const SafetensorsMetadata& metadata) {
        for (const auto& [name, info] : metadata.tensors) {
            if (name.find("model.layers") != std::string::npos) {
                if (name.find("self_attn.q_proj") != std::string::npos) {
                    return "llama";
                }
            }
            if (name.find("transformer.h") != std::string::npos) {
                return "gpt";
            }
            if (name.find("model.decoder.layers") != std::string::npos) {
                return "opt";
            }
        }
        return "llama";
    }

    int infer_vocab_size(const SafetensorsMetadata& metadata) {
        for (const auto& [name, info] : metadata.tensors) {
            if (name.find("embed") != std::string::npos && !info.shape.empty()) {
                return static_cast<int>(info.shape[0]);
            }
        }
        return 32000;
    }

    int infer_hidden_size(const SafetensorsMetadata& metadata) {
        for (const auto& [name, info] : metadata.tensors) {
            if ((name.find("q_proj") != std::string::npos || name.find("k_proj") != std::string::npos) 
                && info.shape.size() >= 2) {
                return static_cast<int>(info.shape[0]);
            }
        }
        return 4096;
    }

    int infer_num_layers(const SafetensorsMetadata& metadata) {
        std::unordered_set<int> layers;
        for (const auto& [name, info] : metadata.tensors) {
            size_t pos = name.find("layers.");
            if (pos != std::string::npos) {
                pos += 7;
                size_t end = name.find('.', pos);
                if (end != std::string::npos) {
                    try {
                        int layer = std::stoi(name.substr(pos, end - pos));
                        layers.insert(layer);
                    } catch (...) {}
                }
            }
        }
        return static_cast<int>(layers.size());
    }

    int infer_num_heads(const SafetensorsMetadata& metadata) {
        for (const auto& [name, info] : metadata.tensors) {
            if (name.find("q_proj") != std::string::npos && info.shape.size() >= 2) {
                int hidden = info.shape[0];
                int head_dim = 128;
                return hidden / head_dim;
            }
        }
        return 32;
    }

    bool create_safetensors_tensors(LoadedModel& model, const SafetensorsMetadata& metadata, 
                                   const std::unordered_map<std::string, std::string>& tensor_mappings,
                                   bool has_quantized = false) {
        // Build reverse mapping: safetensors_name -> canonical_name
        std::unordered_map<std::string, std::string> reverse_map;
        for (const auto& [canonical, safetensors] : tensor_mappings) {
            reverse_map[safetensors] = canonical;
        }

        for (const auto& [safetensors_name, info] : metadata.tensors) {
            // Skip component tensors for quantized models
            if (has_quantized && (ends_with(safetensors_name, ".qweight") || 
                                  ends_with(safetensors_name, ".scales") ||
                                  ends_with(safetensors_name, ".qzeros") ||
                                  ends_with(safetensors_name, ".g_idx"))) {
                continue;
            }

            // For quantized models, output tensors are F16 (dequantized)
            enum ggml_type type = has_quantized ? GGML_TYPE_F16 : SafetensorsLoader::dtype_to_ggml_type(info.dtype);
            
            int n_dims = static_cast<int>(info.shape.size());
            if (n_dims > GGML_MAX_DIMS) n_dims = GGML_MAX_DIMS;
            
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int i = 0; i < n_dims; ++i) {
                ne[i] = info.shape[i];
            }

            // Determine canonical name
            std::string canonical_name = safetensors_name;
            auto it = reverse_map.find(safetensors_name);
            if (it != reverse_map.end()) {
                canonical_name = it->second;
            }

            struct ggml_tensor* tensor = ggml_new_tensor(model.ctx, type, n_dims, ne);
            if (!tensor) {
                core::Logger::error("Failed to create tensor: {}", canonical_name);
                return false;
            }
            
            ggml_set_name(tensor, canonical_name.c_str());
            model.tensors[canonical_name] = tensor;
            model.tensor_name_map[canonical_name] = safetensors_name;

            core::Logger::debug("Mapped tensor: {} -> {} (type: {})", canonical_name, safetensors_name, 
                has_quantized ? "F16(dequant)" : SafetensorsLoader::ggml_type_to_dtype(type));
        }

        core::Logger::info("Created {} tensors in ggml context", model.tensors.size());
        return true;
    }

    std::string strip_think(const std::string& s) {
        std::string out = s;
        // remove <think>...</think> including newlines - ponytail: hemat token
        size_t pos = 0;
        while ((pos = out.find("<think>")) != std::string::npos) {
            size_t end = out.find("</think>", pos);
            if (end == std::string::npos) { out.erase(pos); break; }
            out.erase(pos, end - pos + 8);
        }
        // trim leading whitespace
        size_t start = out.find_first_not_of(" \n\r\t");
        if (start != std::string::npos) out = out.substr(start);
        return out;
    }

    std::string build_chat_prompt(LoadedModel& model, const std::vector<std::pair<std::string,std::string>>& raw_messages, bool no_think = false) {
        // ponytail + caveman: concise, no-think, hemat token
        // caveman = simple, short sentences (STE100). ponytail = ultra-concise, bullet-free.
        std::vector<std::pair<std::string,std::string>> messages = raw_messages;
        // Keep template real from model - no hardcoded system prompt
        // ponytail/caveman now as opencode skills (.opencode/skills/), not injected here
        // If user wants concise, they can pass system message or enable_thinking=false
        if (model.lmodel) {
            std::vector<llama_chat_message> chat;
            for (auto &m : messages) chat.push_back({m.first.c_str(), m.second.c_str()});
            const char* tmpl = llama_model_chat_template(model.lmodel, nullptr);
            int32_t need = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, nullptr, 0);
            if (need > 0) {
                std::vector<char> buf(need);
                int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), buf.size());
                if (n >= 0) {
                    std::string p(buf.data(), n);
                    // F9-10: API template lama tak teruskan kwargs (enable_thinking
                    // hilang) dan tak prefill <think> -> model kadang emit opener,
                    // kadang tidak (acak, BUG-049). Samakan perilaku upstream:
                    bool think_tmpl = tmpl && std::string(tmpl).find("<think>") != std::string::npos;
                    if (think_tmpl) {
                        size_t a = p.rfind("<think>");
                        bool open = (a != std::string::npos && p.find("</think>", a) == std::string::npos);
                        if (no_think) {
                            // tutup yang terbuka, atau prefill blok kosong bila
                            // tak ada think sama sekali -> jawab langsung
                            if (open) p += "\n</think>\n\n";
                            else if (a == std::string::npos) p += "<think>\n\n</think>\n\n";
                        } else if (a == std::string::npos) {
                            p += "<think>\n"; // prefill ala upstream: output deterministik mode think
                        }
                    }
                    return p;
                }
            }
        }
        std::string p;
        for (auto &m : messages) {
            if (m.first == "system") p += "<|im_start|>system\n" + m.second + "<|im_end|>\n";
            else if (m.first == "user") p += "<|im_start|>user\n" + m.second + "<|im_end|>\n";
            else if (m.first == "assistant") p += "<|im_start|>assistant\n" + m.second + "<|im_end|>\n";
        }
        p += "<|im_start|>assistant\n";
        return p;
    }

    // F9-10: streaming UTF-8 assembler. Token pieces can split multibyte
    // chars (byte-fallback tokens); emitting partial bytes breaks JSON and
    // shows ???? in the UI. Buffer trailing incomplete sequences instead.
    struct Utf8Streamer {
        std::string pending;
        static size_t need(unsigned char c) {
            if (c < 0x80) return 1;
            if ((c & 0xE0) == 0xC0) return 2;
            if ((c & 0xF0) == 0xE0) return 3;
            if ((c & 0xF8) == 0xF0) return 4;
            return 0; // stray continuation
        }
        std::string push(const std::string& piece) {
            std::string s = pending + piece;
            pending.clear();
            size_t end = s.size();
            // find longest valid prefix; hold back a trailing incomplete char
            size_t i = 0;
            while (i < s.size()) {
                size_t n = need((unsigned char)s[i]);
                if (n == 0) { s.replace(i, 1, "?"); ++i; continue; }
                if (i + n > s.size()) break; // incomplete at tail
                bool ok = true;
                for (size_t k = 1; k < n; ++k)
                    if (((unsigned char)s[i + k] & 0xC0) != 0x80) { ok = false; break; }
                if (!ok) { s.replace(i, 1, "?"); ++i; continue; }
                i += n;
            }
            if (i < s.size()) {
                pending = s.substr(i);
                return s.substr(0, i);
            }
            return s;
        }
        std::string flush() {
            std::string o = pending;
            pending.clear();
            return o.empty() ? o : "?";
        }
    };

    static double now_ms() {
        using namespace std::chrono;
        return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
    }

    // F9-10 output hygiene (namespace-scope free functions below, shared with
    // the API layer): weak models echo the assistant header and leave blank
    // jungles. Strip ONE leading echoed header + outer blank space, collapse
    // 3+ newlines. Applied to final texts; streams suppress live.

    // BUG-026: token pieces can split multibyte UTF-8 (byte-fallback tokens).
    // nlohmann::json::dump() throws on invalid UTF-8 -> HTTP 400. Sanitize.
    static std::string utf8_fix(const std::string& s) {
        std::string o;
        o.reserve(s.size());
        for (size_t i = 0; i < s.size();) {
            unsigned char c = s[i];
            size_t len = 0;
            if (c < 0x80) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            else { o += '?'; ++i; continue; }
            if (i + len > s.size()) { o += '?'; ++i; continue; }
            bool ok = true;
            for (size_t k = 1; k < len; ++k)
                if ((s[i + k] & 0xC0) != 0x80) { ok = false; break; }
            if (!ok) { o += '?'; ++i; continue; }
            o.append(s, i, len);
            i += len;
        }
        return o;
    }

    // F7 agent loop: same as the standard path of llama_complete but the
    // prompt is already rendered (chat template + tools applied by caller).
    int count_text_toks(const llama_vocab* vocab, const std::string& text) {
        int c = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
        return c > 0 ? c : 1;
    }

    // F9-10 auto-compact ala opencode: bila prompt >= 85% ctx (sisa 15%),
    // ringkas turn lama via model yang sama, lalu reset ke ringkasan + ekor.
    // Konteks fresh per request => pemakaian kembali ~nol setelah compact.
    bool auto_compact(LoadedModel& model,
                      std::vector<std::pair<std::string,std::string>>& messages) {
        int n_ctx = model.config.n_ctx > 0 ? model.config.n_ctx : context_cap_;
        if (n_ctx < 512 || !model.lmodel) return true;
        const llama_vocab* vocab = llama_model_get_vocab(model.lmodel);
        int budget = (int)(n_ctx * 0.85);
        auto count_msgs = [&](const std::vector<std::pair<std::string,std::string>>& m) {
            return count_text_toks(vocab, build_chat_prompt(model, m));
        };
        int total = count_msgs(messages);
        if (total <= budget) return true;
        core::Logger::info("auto-compact: prompt {} toks >= 85% ctx {} -> ringkas", total, n_ctx);
        int rounds = 0;
        while (total > budget && rounds < 3) {
            std::string sys;
            std::vector<std::pair<std::string,std::string>> body;
            for (auto& m : messages) {
                if (m.first == "system" && sys.empty()) sys = m.second;
                else body.push_back(m);
            }
            if (body.size() <= 5) break; // tinggal ekor, tak ada yang diringkas
            size_t cut = body.size() - 4; // 4 pesan ekor dipertahankan verbatim
            std::string hist;
            for (size_t i = 0; i < cut; ++i)
                hist += "[" + body[i].first + "]\n" + body[i].second + "\n\n";
            const std::string instr =
                "Ringkas percakapan berikut secara singkat namun detail "
                "(fakta, keputusan, konteks penting). Tulis ringkasannya saja:\n\n";
            int hlim = (int)(n_ctx * 0.6);
            while (count_text_toks(vocab, instr + hist) > hlim && hist.size() > 2000) {
                size_t drop = hist.size() / 4;
                size_t nl = hist.find('\n', drop);
                hist = hist.substr(nl == std::string::npos ? drop : nl + 1);
            }
            TextEngine::CompletionResult s =
                llama_complete_raw(model, instr + hist, 512, 0.0f, 0.9f, {}, {});
            if (s.finish_reason == "error" || s.text.empty()) break;
            ++rounds;
            int before = total;
            std::string sys2 = sys.empty() ? "" : (sys + "\n\n");
            sys2 += "Ringkasan percakapan sebelumnya (auto-compact, konteks direset):\n"
                + clean_output(s.text);
            std::vector<std::pair<std::string,std::string>> nm;
            nm.emplace_back("system", sys2);
            for (size_t i = cut; i < body.size(); ++i) nm.push_back(body[i]);
            messages.swap(nm);
            total = count_msgs(messages);
            core::Logger::info("auto-compact round {}: {} -> {} toks (ctx {})",
                rounds, before, total, n_ctx);
        }
        // guard keras: satu pesan raksasa pun dipotong tengah bila masih over
        int hard = n_ctx - 256;
        total = count_msgs(messages);
        int gi = 0;
        while (total > hard && gi < 3) {
            ++gi;
            size_t bi = 0;
            for (size_t i = 0; i < messages.size(); ++i)
                if (messages[i].second.size() > messages[bi].second.size()) bi = i;
            std::string& big = messages[bi].second;
            if (big.size() < 2000) break;
            size_t keep = big.size() * 3 / 4;
            size_t head = keep / 4;
            size_t hnl = big.rfind('\n', head);
            size_t tnl = big.find('\n', big.size() - keep + head);
            std::string nb = big.substr(0, hnl == std::string::npos ? head : hnl);
            nb += "\n\n[...dipotong auto-compact...]\n\n";
            nb += big.substr(tnl == std::string::npos ? big.size() - keep + head : tnl + 1);
            big.swap(nb);
            total = count_msgs(messages);
        }
        if (total > hard)
            core::Logger::warn("auto-compact: prompt {} toks masih > batas {} (ctx {})",
                total, hard, n_ctx);
        // ringkasan memakai lctx yang sama -> segarkan agar prefill utama
        // mulai dari KV kosong (posisi monoton dari nol lagi).
        if (rounds > 0 && !fresh_context(model)) {
            core::Logger::error("auto-compact: fresh context gagal");
            return false;
        }
        return true;
    }

    // F9-10 BUG-047: <think> belum tutup = masih berpikir; jawaban ada =
    // teks non-blank setelah </think>. Tanpa ini, kepotong max_tokens saat
    // berpikir -> reasoning tanpa jawaban -> UI tandai "Cancelled".
    // prompt_prefills_think: prompt diakhiri <think> terbuka (prefill manual
    // ala upstream, atau template native). Output mulai dalam mode think.
    static bool prompt_prefills_think(const std::string& prompt) {
        size_t a = prompt.rfind("<think>");
        if (a == std::string::npos) return false;
        return prompt.find("</think>", a) == std::string::npos;
    }
    static bool think_unclosed(const std::string& out, bool prefilled = false) {
        if (prefilled) return out.find("</think>") == std::string::npos;
        size_t a = out.rfind("<think>");
        if (a == std::string::npos) return false;
        return out.find("</think>", a) == std::string::npos;
    }
    static bool think_has_answer(const std::string& out) {
        size_t c = out.find("</think>");
        if (c == std::string::npos) return false;
        for (size_t i = c + 8; i < out.size(); ++i) {
            char ch = out[i];
            if (ch!=' '&&ch!='\t'&&ch!='\n'&&ch!='\r') return true;
        }
        return false;
    }

    TextEngine::CompletionResult llama_complete_raw(LoadedModel& model, const std::string& prompt,
        int max_tokens, float temp, float top_p, const std::vector<std::string>& stops,
        const GenerationOptions& opts = {}) {
        TextEngine::CompletionResult res;
        if (!model.lmodel || !model.lctx) {
            res.text = "Error: model not loaded with llama";
            res.finish_reason = "error";
            return res;
        }
        const llama_vocab* vocab = llama_model_get_vocab(model.lmodel);
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) n_prompt = 1;
        std::vector<llama_token> toks(n_prompt);
        int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
        if (n < 0) n = 0;
        else toks.resize(n);
        // guard keras: prompt mentah raksasa dipotong tengah di level token
        // (head 1/4 + tail 3/4, BOS di [0] tetap) agar tak pernah over ctx.
        int n_ctx_hard = model.config.n_ctx > 0 ? model.config.n_ctx : context_cap_;
        if ((int)toks.size() > n_ctx_hard - 256 && toks.size() > 512) {
            size_t keep = (size_t)(n_ctx_hard - 256);
            size_t head = keep / 4;
            std::vector<llama_token> nt;
            nt.insert(nt.end(), toks.begin(), toks.begin() + head);
            nt.insert(nt.end(), toks.end() - (keep - head), toks.end());
            toks.swap(nt);
            core::Logger::warn("raw prompt truncated to {} toks (ctx {})",
                (int)toks.size(), n_ctx_hard);
        }
        res.prompt_tokens = (int)toks.size();

        bool lora_active = apply_lora_adapters(model, opts.lora_adapters) > 0;
        std::string grammar = opts.grammar.empty() && opts.json_mode ? json_grammar() : opts.grammar;
        llama_sampler* smpl = make_sampler(vocab, temp, top_p, grammar, opts.grammar_root, sampling_from_opts(opts));

        llama_memory_clear(llama_get_memory(model.lctx), true);
        double t_pre = now_ms();
        core::Logger::info("raw prefill toks={}", (int)toks.size());
        if (!decode_chunked(model.lctx, toks)) {
            core::Logger::error("llama_decode prompt failed");
            llama_sampler_free(smpl);
            if (lora_active) clear_lora_adapters(model);
            res.text = ""; res.finish_reason = "error"; return res;
        }
        core::Logger::info("raw prefilled");
        res.prompt_ms = now_ms() - t_pre;
        std::string out;
        int n_decode = 0;
        Utf8Streamer u8;
        double t_dec = now_ms();
        bool pre_think = prompt_prefills_think(prompt);
        // BUG-047 berlaku juga di raw (prompt agent native prefill think):
        // tanpa extension, think kepotong -> calls hilang -> loop sia-sia.
        int lim = max_tokens;
        int ext_total = 0;
        const int ext_cap = 4096;
        bool natural_end = false;
        for (int i = 0; i < lim; i++) {
            llama_token tok = llama_sampler_sample(smpl, model.lctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) { natural_end = true; break; }
            char buf[256];
            int nch = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            std::string piece;
            if (nch > 0) piece.assign(buf, nch);
            piece = u8.push(piece);
            if (!piece.empty()) {
                out += piece;
                bool hit = false;
                for (auto &s : stops) if (!s.empty() && out.find(s) != std::string::npos) { hit = true; break; }
                if (hit) { natural_end = true; break; }
                llama_batch batch = llama_batch_get_one(&tok, 1);
                if (llama_decode(model.lctx, batch) != 0) break;
                n_decode++;
            } else {
                llama_batch batch = llama_batch_get_one(&tok, 1);
                llama_decode(model.lctx, batch);
            }
            if (i+1 >= lim && think_unclosed(out, pre_think) && ext_total < ext_cap) {
                int room = model.config.n_ctx - ((int)toks.size() + i + 1) - 16;
                if (room < 128) break;
                int extra = 512;
                if (extra > ext_cap - ext_total) extra = ext_cap - ext_total;
                if (extra > room) extra = room;
                if (extra < 64) break;
                lim += extra;
                ext_total += extra;
            }
            if (ext_total > 0 && !think_unclosed(out, pre_think) && think_has_answer(out)) {
                int ab = max_tokens > 512 ? max_tokens : 512;
                if (lim > i + ab) lim = i + ab;
            }
        }
        out += u8.flush();
        res.predicted_ms = now_ms() - t_dec;
        llama_sampler_free(smpl);
        if (lora_active) clear_lora_adapters(model);
        touch(model);
        res.text = clean_output(out);
        res.completion_tokens = n_decode;
        res.finish_reason = natural_end ? "stop" : "length";
        return res;
    }

    TextEngine::CompletionResult llama_complete(LoadedModel& model, const std::vector<std::pair<std::string,std::string>>& messages, int max_tokens, float temp, float top_p, const std::vector<std::string>& stops, const GenerationOptions& opts = {}) {
        TextEngine::CompletionResult res;
        if (!model.lmodel || !model.lctx) {
            res.text = "Error: model not loaded with llama";
            res.finish_reason = "error";
            return res;
        }
        // F9-10 auto-compact: ringkas + reset bila prompt >= 85% ctx
        auto msgs = messages;
        if (!auto_compact(model, msgs)) {
            res.text = "Error: auto-compact failed"; res.finish_reason = "error";
            return res;
        }
        // F3-7 speculative decoding via draft model
        if (!opts.draft_model.empty() && opts.draft_model != model.model_id) {
            auto dit = loaded_models_.find(opts.draft_model);
            if (dit != loaded_models_.end() && dit->second.lmodel && dit->second.lctx) {
                touch(dit->second);
                // F9-10: fresh draft context too (target already refreshed by wrapper)
                if (!fresh_context(dit->second)) {
                    core::Logger::warn("Draft context init failed, falling back to standard decode");
                } else {
                    return llama_complete_speculative(model, dit->second, msgs, max_tokens, temp, top_p, stops, opts);
                }
            }
            core::Logger::warn("Draft model '{}' not loaded, falling back to standard decode", opts.draft_model);
        }
        const llama_vocab* vocab = llama_model_get_vocab(model.lmodel);
        std::string prompt = build_chat_prompt(model, msgs, !opts.enable_thinking);
        // tokenize
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) n_prompt = 1;
        std::vector<llama_token> toks(n_prompt);
        int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
        if (n < 0) n = 0;
        else toks.resize(n);
        res.prompt_tokens = (int)toks.size();

        // F3-9: apply requested LoRA adapters for this request
        bool lora_active = apply_lora_adapters(model, opts.lora_adapters) > 0;

        // sampler chain with optional grammar constraint (F3-8)
        std::string grammar = opts.grammar.empty() && opts.json_mode ? json_grammar() : opts.grammar;
        llama_sampler* smpl = make_sampler(vocab, temp, top_p, grammar, opts.grammar_root, sampling_from_opts(opts));

        // clear KV and chunked-prefill the prompt (F3)
        // F9-10 experiment: skip clear (monotonic positions, like llama-server
        // slots) to isolate whether clear() corrupts reused-context state.
        //llama_memory_clear(llama_get_memory(model.lctx), true);
        double t_pre = now_ms();
        if (!decode_chunked(model.lctx, toks)) {
            core::Logger::error("llama_decode prompt failed");
            llama_sampler_free(smpl);
            if (lora_active) clear_lora_adapters(model);
            res.text = "";
            res.finish_reason = "error";
            return res;
        }
        res.prompt_ms = now_ms() - t_pre;
        std::string out;
        int n_decode = 0;
        Utf8Streamer u8;
        double t_dec = now_ms();
        bool pre_think = prompt_prefills_think(prompt);
        // BUG-047: kepotong max_tokens saat think belum tutup -> lanjut
        // sampai think tutup + ada jawaban (batas: +maks(256, max_tokens)).
        int lim = max_tokens;
        int ext_total = 0;
        const int ext_cap = 4096;
        bool natural_end = false;
        for (int i=0;i<lim;i++) {
            llama_token tok = llama_sampler_sample(smpl, model.lctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) { natural_end = true; break; }
            char buf[256];
            int nch = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            std::string piece;
            if (nch > 0) piece.assign(buf, nch);
            piece = u8.push(piece);
            if (!piece.empty()) {
                out += piece;
                bool hit = false;
                for (auto &s : stops) if (!s.empty() && out.find(s) != std::string::npos) { hit=true; break; }
                if (hit) { natural_end = true; break; }
                llama_batch batch = llama_batch_get_one(&tok, 1);
                if (llama_decode(model.lctx, batch) != 0) break;
                n_decode++;
            } else {
                llama_batch batch = llama_batch_get_one(&tok, 1);
                llama_decode(model.lctx, batch);
            }
            if (i+1 >= lim && think_unclosed(out, pre_think) && ext_total < ext_cap) {
                int room = model.config.n_ctx - ((int)toks.size() + i + 1) - 16;
                if (room < 128) break; // tak ada ruang KV: berhenti apa adanya
                int extra = 512;
                if (extra > ext_cap - ext_total) extra = ext_cap - ext_total;
                if (extra > room) extra = room;
                if (extra < 64) break;
                lim += extra;
                ext_total += extra;
                core::Logger::info("think truncated, auto-continue +{} (total +{}, ctx {})",
                    extra, ext_total, model.config.n_ctx);
            }
            if (ext_total > 0 && !think_unclosed(out, pre_think) && think_has_answer(out)) {
                // think selesai + jawaban mulai: beri ruang jawaban lalu stop
                int ab = max_tokens > 512 ? max_tokens : 512;
                if (lim > i + ab) lim = i + ab;
            }
        }
        out += u8.flush();
        res.predicted_ms = now_ms() - t_dec;
        llama_sampler_free(smpl);
        if (lora_active) clear_lora_adapters(model);
        touch(model);
        res.text = clean_output(out); // keep <think> - thinking is necessary per user, strip only if enable_thinking=false
        res.completion_tokens = n_decode;
        res.finish_reason = natural_end ? "stop" : "length";
        return res;
    }

    // F3-7 speculative decoding: draft proposes, target verifies in one batch.
    // Both KVs grow monotonically like normal generation (no mid-request clears);
    // rejected tails are trimmed with seq_rm. Grammar constraints fall back to
    // standard decode (rejected tokens would pollute grammar sampler state).
    TextEngine::CompletionResult llama_complete_speculative(LoadedModel& target, LoadedModel& draft,
        const std::vector<std::pair<std::string,std::string>>& messages, int max_tokens,
        float temp, float top_p, const std::vector<std::string>& stops, const GenerationOptions& opts) {
        TextEngine::CompletionResult res;
        std::string grammar = opts.grammar.empty() && opts.json_mode ? json_grammar() : opts.grammar;
        if (!grammar.empty()) {
            core::Logger::info("speculative disabled under grammar constraint, using standard decode");
            GenerationOptions std_opts = opts;
            std_opts.draft_model.clear();
            return llama_complete(target, messages, max_tokens, temp, top_p, stops, std_opts);
        }
        const llama_vocab* vocab = llama_model_get_vocab(target.lmodel);
        const llama_vocab* vocab_draft = llama_model_get_vocab(draft.lmodel);
        std::string prompt = build_chat_prompt(target, messages, !opts.enable_thinking);
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) n_prompt = 1;
        std::vector<llama_token> toks(n_prompt);
        int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
        if (n < 0) n = 0; else toks.resize(n);
        res.prompt_tokens = (int)toks.size();

        bool lora_active = apply_lora_adapters(target, opts.lora_adapters) > 0;
        llama_sampler* smpl_tgt = make_sampler(vocab, temp, top_p, "", "");
        llama_sampler* smpl_draft = make_sampler(vocab_draft, temp, top_p, "", "");

        auto fail = [&](const char* why) {
            core::Logger::error("speculative failed: {}", why);
            llama_sampler_free(smpl_tgt); llama_sampler_free(smpl_draft);
            if (lora_active) clear_lora_adapters(target);
            res.text = ""; res.finish_reason = "error"; return res;
        };

        int K = std::max(1, std::min(opts.draft_tokens, 8));
        int n_ctx_lim = target.config.n_ctx > 0 ? std::min(target.config.n_ctx, 4096) : 4096;
        auto piece = [&](llama_token tok) -> std::string {
            char buf[256];
            int nch = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            return nch > 0 ? utf8_fix(std::string(buf, nch)) : std::string();
        };
        ExplicitBatch st_tgt, st_draft;  // backing storage for explicit batches
        auto decode_seq = [&](LoadedModel& m, ExplicitBatch& st,
                              const llama_token* tp, int cnt, int start_pos,
                              bool want_logits) -> bool {
            return decode_explicit(m.lctx, tp, cnt, start_pos, want_logits, st);
        };
        auto trim_tail = [&](LoadedModel& m, int keep_from, int kv_end) -> bool {
            if (kv_end <= keep_from) return true;
            return llama_memory_seq_rm(llama_get_memory(m.lctx), 0,
                (llama_pos)keep_from, (llama_pos)kv_end);
        };

        // initial prefill on fresh KVs (single clear each, like standard path)
        llama_memory_clear(llama_get_memory(target.lctx), true);
        llama_memory_clear(llama_get_memory(draft.lctx), true);
        {
            size_t off = 0;
            while (off < toks.size()) {
                size_t n = std::min<size_t>(kPrefillChunk, toks.size() - off);
                if (!decode_seq(target, st_tgt, toks.data() + off, (int)n, (int)off, false)) {
                    return fail("prefill");
                }
                off += n;
            }
        }
        {
            size_t off = 0;
            while (off < toks.size()) {
                size_t n = std::min<size_t>(kPrefillChunk, toks.size() - off);
                if (!decode_seq(draft, st_draft, toks.data() + off, (int)n, (int)off, false)) {
                    return fail("prefill");
                }
                off += n;
            }
        }
        // t0 = target's first-token distribution (post-prefill, idx -1).
        // Verify logits are shifted by one: logits[i] predicts the token AFTER
        // proposals[i], so proposals[i] is checked against logits[i-1] (t0 for i=0).
        llama_token t0 = llama_sampler_sample(smpl_tgt, target.lctx, -1);
        int cur_tgt = (int)toks.size();    // tracked KV lengths (explicit positions)
        int cur_draft = (int)toks.size();
        std::string out;
        int n_decode = 0;
        bool done = false;

        while (!done && (cur_tgt - (int)toks.size()) < max_tokens) {
            if (cur_tgt + K + 4 >= n_ctx_lim || cur_draft + K + 4 >= n_ctx_lim) break;
            // 1. draft proposes up to K tokens at explicit positions
            std::vector<llama_token> proposals;
            for (int k = 0; k < K; ++k) {
                if (cur_draft >= n_ctx_lim - 4) break;
                llama_token dt = llama_sampler_sample(smpl_draft, draft.lctx, -1);
                if (llama_vocab_is_eog(vocab_draft, dt)) break;
                proposals.push_back(dt);
                if (!decode_seq(draft, st_draft, &dt, 1, cur_draft, false)) { proposals.pop_back(); break; }
                cur_draft++;
            }
            if (proposals.empty()) break;  // draft wants to stop
            int P = (int)proposals.size();
            int base = cur_tgt;  // both KVs hold [0, base) + P new tokens after verify
            // 2. target verifies all proposals in ONE forward pass at explicit positions.
            // Request logits for EVERY position so explicit-idx sampling is valid.
            if (!decode_seq(target, st_tgt, proposals.data(), P, cur_tgt, true)) break;
            cur_tgt += P;  // target KV now holds base + P
            // 3. compare each proposal against its own distribution (shifted idx)
            int n_acc = 0;
            llama_token correction = -1;
            for (int i = 0; i < P; ++i) {
                llama_token tt = (i == 0) ? t0
                    : llama_sampler_sample(smpl_tgt, target.lctx, (int32_t)(i - 1));
                if (tt == proposals[i] && !llama_vocab_is_eog(vocab, tt)) {
                    n_acc++;
                } else {
                    correction = tt;
                    break;
                }
            }
            if (n_acc < P) {
                // reject at n_acc: trim rejected tail on both, decode correction
                bool corr_eog = (correction < 0) || llama_vocab_is_eog(vocab, correction);
                if (!trim_tail(target, base + n_acc, base + P)) break;
                if (!trim_tail(draft, base + n_acc, base + P)) break;
                cur_tgt = base + n_acc;
                cur_draft = base + n_acc;
                if (corr_eog) { done = true; break; }
                for (int i = 0; i < n_acc; ++i) {
                    out += piece(proposals[i]); n_decode++;
                    if (llama_vocab_is_eog(vocab, proposals[i])) { done = true; break; }
                }
                if (done) break;
                out += piece(correction); n_decode++;
                if (!decode_seq(target, st_tgt, &correction, 1, cur_tgt, false)) break;
                if (!decode_seq(draft, st_draft, &correction, 1, cur_draft, false)) break;
                cur_tgt++;
                cur_draft++;
                t0 = llama_sampler_sample(smpl_tgt, target.lctx, -1);  // refresh for next round
            } else {
                // full accept: emit proposals + bonus token
                llama_token bonus = llama_sampler_sample(smpl_tgt, target.lctx, (int32_t)(P - 1));
                if (llama_vocab_is_eog(vocab, bonus)) {
                    for (int i = 0; i < P; ++i) {
                        out += piece(proposals[i]); n_decode++;
                    }
                    done = true;
                    break;
                }
                for (int i = 0; i < P; ++i) {
                    out += piece(proposals[i]); n_decode++;
                    if (llama_vocab_is_eog(vocab, proposals[i])) { done = true; break; }
                }
                if (done) break;
                out += piece(bonus); n_decode++;
                if (!decode_seq(target, st_tgt, &bonus, 1, cur_tgt, false)) break;
                if (!decode_seq(draft, st_draft, &bonus, 1, cur_draft, false)) break;
                cur_tgt++;
                cur_draft++;
                t0 = llama_sampler_sample(smpl_tgt, target.lctx, -1);  // refresh for next round
            }
            if (done) break;
            bool hit = false;
            for (auto& s : stops) if (!s.empty() && out.find(s) != std::string::npos) { hit = true; break; }
            if (hit) break;
        }
        llama_sampler_free(smpl_tgt); llama_sampler_free(smpl_draft);
        if (lora_active) clear_lora_adapters(target);
        touch(target); touch(draft);
        res.text = out;
        res.completion_tokens = n_decode;
        res.finish_reason = done ? "stop" : "length";
        return res;
    }

    TextEngine::Timings llama_stream(LoadedModel& model, const std::vector<std::pair<std::string,std::string>>& messages, int max_tokens, float temp, float top_p, const std::vector<std::string>& stops, StreamCallback cb, const GenerationOptions& opts = {}, TextEngine::ProgressCallback progress = {}, const std::string* raw_prompt = nullptr) {
        if (!model.lmodel || !model.lctx) { cb("Error: model not loaded", true); return {}; }
        std::string prompt;
        if (raw_prompt) {
            prompt = *raw_prompt; // tools passthrough: template+tools sudah render di agent
        } else {
            // F9-10 auto-compact: ringkas + reset bila prompt >= 85% ctx
            auto msgs = messages;
            if (!auto_compact(model, msgs)) { cb("Error: auto-compact failed", true); return {}; }
            prompt = build_chat_prompt(model, msgs, !opts.enable_thinking);
        }
        const llama_vocab* vocab = llama_model_get_vocab(model.lmodel);
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) n_prompt = 1;
        std::vector<llama_token> toks(n_prompt);
        int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
        if (n < 0) n = 0; else toks.resize(n);
        // guard keras prompt mentah (sama seperti llama_complete_raw)
        if (raw_prompt) {
            int n_ctx_hard = model.config.n_ctx > 0 ? model.config.n_ctx : context_cap_;
            if ((int)toks.size() > n_ctx_hard - 256 && toks.size() > 512) {
                size_t keep = (size_t)(n_ctx_hard - 256);
                size_t head = keep / 4;
                std::vector<llama_token> nt;
                nt.insert(nt.end(), toks.begin(), toks.begin() + head);
                nt.insert(nt.end(), toks.end() - (keep - head), toks.end());
                toks.swap(nt);
                core::Logger::warn("stream raw prompt truncated to {} toks (ctx {})",
                    (int)toks.size(), n_ctx_hard);
            }
        }
        bool lora_active = apply_lora_adapters(model, opts.lora_adapters) > 0;
        std::string grammar = opts.grammar.empty() && opts.json_mode ? json_grammar() : opts.grammar;
        llama_sampler* smpl = make_sampler(vocab, temp, top_p, grammar, opts.grammar_root, sampling_from_opts(opts));
        llama_memory_clear(llama_get_memory(model.lctx), true);
        double t_pre = now_ms();
        if (!decode_chunked(model.lctx, toks)) { cb("", true); llama_sampler_free(smpl); if (lora_active) clear_lora_adapters(model); return {}; }
        double pre_ms = now_ms() - t_pre;
        // BUG-049: prompt diakhiri <think> terbuka -> output mulai mode think
        if (progress) {
            size_t a = prompt.rfind("<think>");
            size_t b = prompt.rfind("</think>");
            bool pre = (a != std::string::npos && (b == std::string::npos || a > b));
            progress((int)toks.size(), pre_ms, pre); // prompt eksak untuk gauge live
        }
        std::string acc;
        bool finished = false;
        Utf8Streamer u8;
        // F9-10: suppress echoed assistant header live (models regurgitate it);
        // hold the head until it can be cleaned, then stream straight through.
        std::string lead;
        bool lead_done = false;
        auto flush_lead = [&]() {
            if (lead_done) return;
            lead_done = true;
            lead = clean_prefix(lead);
            if (!lead.empty()) cb(lead, false);
            lead.clear();
        };
        int n_decode = 0;
        double t_dec = now_ms();
        bool pre_think = prompt_prefills_think(prompt);
        // BUG-047: sama seperti non-stream - lanjut bila think belum tutup.
        int lim = max_tokens;
        int ext_total = 0;
        const int ext_cap = 4096;
        for (int i=0;i<lim;i++) {
            llama_token tok = llama_sampler_sample(smpl, model.lctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) { flush_lead(); cb("", true); finished=true; break; }
            char buf[256];
            int nch = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            std::string piece;
            if (nch > 0) piece.assign(buf, nch);
            piece = u8.push(piece);
            // keep <think> - thinking is necessary, do not skip
            acc += piece;
            bool hit=false;
            for (auto &s: stops) if (!s.empty() && acc.find(s)!=std::string::npos) hit=true;
            if (hit) { flush_lead(); cb("", true); finished=true; break; }
            if (!piece.empty()) {
                if (!lead_done && lead.size() < 64) {
                    lead += piece;
                    if (lead.size() >= 64) flush_lead();
                } else {
                    flush_lead();
                    cb(piece, false);
                }
            }
            llama_batch b2 = llama_batch_get_one(&tok, 1);
            if (llama_decode(model.lctx, b2)!=0) { flush_lead(); cb("", true); finished=true; break; }
            n_decode++;
            if (i+1 >= lim && think_unclosed(acc, pre_think) && ext_total < ext_cap) {
                int room = model.config.n_ctx - ((int)toks.size() + i + 1) - 16;
                if (room < 128) { flush_lead(); cb("", true); finished=true; break; }
                int extra = 512;
                if (extra > ext_cap - ext_total) extra = ext_cap - ext_total;
                if (extra > room) extra = room;
                if (extra < 64) { flush_lead(); cb("", true); finished=true; break; }
                lim += extra;
                ext_total += extra;
                core::Logger::info("think truncated, auto-continue +{} (total +{}, ctx {})",
                    extra, ext_total, model.config.n_ctx);
            }
            if (ext_total > 0 && !think_unclosed(acc, pre_think) && think_has_answer(acc)) {
                int ab = max_tokens > 512 ? max_tokens : 512;
                if (lim > i + ab) lim = i + ab;
            }
        }
        acc += u8.flush();
        flush_lead();
        if (!finished) cb("", true);
        TextEngine::Timings tm;
        tm.prompt_n = (int)toks.size(); tm.prompt_ms = pre_ms;
        tm.predicted_n = n_decode; tm.predicted_ms = now_ms() - t_dec;
        llama_sampler_free(smpl);
        if (lora_active) clear_lora_adapters(model);
        touch(model);
        return tm;
    }

    // Real embeddings via llama.cpp pooled sequence embeddings (no more zeros)
    std::vector<std::vector<float>> llama_embed(LoadedModel& model, const std::vector<std::string>& inputs) {
        if (!model.lmodel || !model.lctx) return {};
        const llama_vocab* vocab = llama_model_get_vocab(model.lmodel);
        std::vector<std::vector<float>> out;
        int n_embd = llama_model_n_embd(model.lmodel);
        llama_set_embeddings(model.lctx, true);
        for (auto &txt : inputs) {
            int n = -llama_tokenize(vocab, txt.c_str(), txt.size(), nullptr, 0, true, true);
            if (n<=0) n=1;
            std::vector<llama_token> toks(n);
            int nn = llama_tokenize(vocab, txt.c_str(), txt.size(), toks.data(), toks.size(), true, true);
            if (nn>0) toks.resize(nn); else toks.clear();
            std::vector<float> emb(n_embd, 0.0f);
            llama_memory_clear(llama_get_memory(model.lctx), true);
            if (!toks.empty() && decode_chunked(model.lctx, toks)) {
                float* e = llama_get_embeddings_seq(model.lctx, 0);
                if (e) {
                    for (int i = 0; i < n_embd; ++i) emb[i] = e[i];
                } else {
                    // fallback: mean-pool last hidden states is internal; keep zeros only if API yields null
                    core::Logger::warn("llama_get_embeddings_seq returned null, using zero vector");
                }
            }
            out.push_back(std::move(emb));
        }
        llama_set_embeddings(model.lctx, false);
        touch(model);
        return out;
    }
};

TextEngine::TextEngine() : pimpl_(std::make_unique<Impl>()) {}

TextEngine::~TextEngine() = default;

bool TextEngine::initialize(const std::string& models_dir) {
    pimpl_->models_dir_ = models_dir;
    return pimpl_->initialize_backend();
}

void TextEngine::shutdown() {
    pimpl_->shutdown();
}

bool TextEngine::load_model(const std::string& model_id, const std::string& file_path) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);

    if (pimpl_->loaded_models_.find(model_id) != pimpl_->loaded_models_.end()) {
        core::Logger::warn("Model {} already loaded", model_id);
        return true;
    }

    LoadedModel model;
    model.id = core::utils::random_id();
    model.model_id = model_id;

    std::filesystem::path path(file_path);
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    bool success = false;
    if (ext == ".gguf") {
        model.format = ModelFormat::GGUF;
        success = pimpl_->load_gguf_model(file_path, model);
    } else if (ext == ".safetensors") {
        model.format = ModelFormat::SafeTensors;
        success = pimpl_->load_safetensors_model(file_path, model);
    } else {
        core::Logger::error("Unknown model format: {}", ext);
        return false;
    }

    if (!success) {
        return false;
    }

    pimpl_->evict_if_needed(model_id);  // F1-8 hot-swap
    model.last_used_at = pimpl_->now_secs();
    pimpl_->loaded_models_[model_id] = std::move(model);
    return true;
}

bool TextEngine::load_gguf_model(const std::string& model_id, const std::string& file_path) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);

    if (pimpl_->loaded_models_.find(model_id) != pimpl_->loaded_models_.end()) {
        core::Logger::warn("Model {} already loaded", model_id);
        return true;
    }

    LoadedModel model;
    model.id = core::utils::random_id();
    model.model_id = model_id;
    model.format = ModelFormat::GGUF;

    if (!pimpl_->load_gguf_model(file_path, model)) {
        return false;
    }

    pimpl_->evict_if_needed(model_id);  // F1-8 hot-swap
    model.last_used_at = pimpl_->now_secs();
    pimpl_->loaded_models_[model_id] = std::move(model);
    return true;
}

bool TextEngine::load_safetensors_model(const std::string& model_id, const std::string& file_path) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);

    if (pimpl_->loaded_models_.find(model_id) != pimpl_->loaded_models_.end()) {
        core::Logger::warn("Model {} already loaded", model_id);
        return true;
    }

    LoadedModel model;
    model.id = core::utils::random_id();
    model.model_id = model_id;
    model.format = ModelFormat::SafeTensors;

    if (!pimpl_->load_safetensors_model(file_path, model)) {
        return false;
    }

    pimpl_->evict_if_needed(model_id);  // F1-8 hot-swap
    model.last_used_at = pimpl_->now_secs();
    pimpl_->loaded_models_[model_id] = std::move(model);
    return true;
}

bool TextEngine::unload_model(const std::string& model_id) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);

    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end()) {
        return false;
    }

    pimpl_->unload_model_internal(it->second);
    pimpl_->loaded_models_.erase(it);
    return true;
}

bool TextEngine::is_loaded(const std::string& model_id) const {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    return it != pimpl_->loaded_models_.end() && it->second.loaded;
}

TextEngine::CompletionResult TextEngine::complete_raw(
    const std::string& model_id,
    const std::string& prompt,
    int max_tokens,
    float temperature,
    float top_p,
    const std::vector<std::string>& stop,
    const GenerationOptions& opts
) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end() || !it->second.loaded || !it->second.lmodel) {
        CompletionResult r; r.text="Error: Model not loaded"; r.finish_reason="error"; return r;
    }
    // F9-10: fresh context per request (kills reused-context decode hangs)
    if (!pimpl_->fresh_context(it->second)) {
        CompletionResult r; r.text="Error: context init failed"; r.finish_reason="error"; return r;
    }
    return pimpl_->llama_complete_raw(it->second, prompt, max_tokens, temperature, top_p, stop, opts);
}

struct llama_model* TextEngine::get_llama_model(const std::string& model_id) const {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end() || !it->second.loaded) return nullptr;
    return it->second.lmodel;
}

TextEngine::CompletionResult TextEngine::complete(
    const std::string& model_id,
    const std::vector<std::pair<std::string, std::string>>& messages,
    int max_tokens,
    float temperature,
    float top_p,
    const std::vector<std::string>& stop,
    const GenerationOptions& opts
) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end() || !it->second.loaded) {
        CompletionResult r; r.text="Error: Model not loaded"; r.finish_reason="error"; return r;
    }
    // F9-10: fresh context per request (kills reused-context decode hangs)
    if (!pimpl_->fresh_context(it->second)) {
        CompletionResult r; r.text="Error: context init failed"; r.finish_reason="error"; return r;
    }
    // GGUF real via llama.cpp
    if (it->second.lmodel) {
        return pimpl_->llama_complete(it->second, messages, max_tokens, temperature, top_p, stop, opts);
    }
    // fallback safetensors (still placeholder)
    CompletionResult r; r.text="Model type not yet supported for real inference"; r.finish_reason="error"; return r;
}

TextEngine::Timings TextEngine::complete_stream(
    const std::string& model_id,
    const std::vector<std::pair<std::string, std::string>>& messages,
    int max_tokens,
    float temperature,
    float top_p,
    const std::vector<std::string>& stop,
    StreamCallback callback,
    const GenerationOptions& opts,
    ProgressCallback progress
) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end() || !it->second.loaded) {
        callback("Error: Model not loaded", true); return {};
    }
    // F9-10: fresh context per request (kills reused-context decode hangs)
    if (!pimpl_->fresh_context(it->second)) {
        callback("Error: context init failed", true); return {};
    }
    if (it->second.lmodel) {
        return pimpl_->llama_stream(it->second, messages, max_tokens, temperature, top_p, stop, callback, opts, progress);
    }
    callback("Model type not supported", true);
    return {};
}

TextEngine::Timings TextEngine::complete_stream_raw(
    const std::string& model_id,
    const std::string& prompt,
    int max_tokens,
    float temperature,
    float top_p,
    const std::vector<std::string>& stop,
    StreamCallback callback,
    const GenerationOptions& opts,
    ProgressCallback progress
) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end() || !it->second.loaded || !it->second.lmodel) {
        callback("Error: Model not loaded", true); return {};
    }
    // F9-10: fresh context per request (kills reused-context decode hangs)
    if (!pimpl_->fresh_context(it->second)) {
        callback("Error: context init failed", true); return {};
    }
    static const std::vector<std::pair<std::string,std::string>> no_msgs;
    return pimpl_->llama_stream(it->second, no_msgs, max_tokens, temperature, top_p,
                                stop, callback, opts, progress, &prompt);
}

void TextEngine::set_max_loaded_models(int n) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    pimpl_->max_loaded_models_ = n;
}

int TextEngine::max_loaded_models() const {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    return pimpl_->max_loaded_models_;
}

void TextEngine::set_tensor_split(const std::vector<float>& splits) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    pimpl_->tensor_split_ = splits;
}

void TextEngine::set_n_batch(int n) {
    if (n > 0) pimpl_->n_batch_ = n;
}
void TextEngine::set_n_ubatch(int n) {
    if (n > 0) pimpl_->n_ubatch_ = n;
}
void TextEngine::set_n_threads_batch(int n) {
    pimpl_->n_threads_batch_ = n;
}
void TextEngine::set_cache_types(const std::string& k, const std::string& v) {
    pimpl_->cache_type_k_ = k;
    pimpl_->cache_type_v_ = v;
}
void TextEngine::set_use_mmap(bool on) {
    pimpl_->use_mmap_ = on;
}
void TextEngine::set_context_cap(int n) {
    if (n > 0) pimpl_->context_cap_ = n;
}
void TextEngine::set_split_mode(int mode) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    pimpl_->split_mode_ = mode;
}

bool TextEngine::load_lora_adapter(const std::string& model_id, const std::string& adapter_id, const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end() || !it->second.lmodel) {
        core::Logger::error("LoRA load: model '{}' not loaded", model_id);
        return false;
    }
    auto ait = it->second.lora_adapters.find(adapter_id);
    if (ait != it->second.lora_adapters.end()) {
        llama_adapter_lora_free(ait->second);
        it->second.lora_adapters.erase(ait);
    }
    llama_adapter_lora* adapter = llama_adapter_lora_init(it->second.lmodel, path.c_str());
    if (!adapter) {
        core::Logger::error("LoRA load failed: {} -> {}", path, adapter_id);
        return false;
    }
    it->second.lora_adapters[adapter_id] = adapter;
    core::Logger::info("LoRA adapter loaded: {} on model {}", adapter_id, model_id);
    return true;
}

bool TextEngine::unload_lora_adapter(const std::string& model_id, const std::string& adapter_id) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end()) return false;
    auto ait = it->second.lora_adapters.find(adapter_id);
    if (ait == it->second.lora_adapters.end()) return false;
    llama_adapter_lora_free(ait->second);
    it->second.lora_adapters.erase(ait);
    core::Logger::info("LoRA adapter unloaded: {} from model {}", adapter_id, model_id);
    return true;
}

std::vector<std::string> TextEngine::list_lora_adapters(const std::string& model_id) const {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    std::vector<std::string> out;
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end()) return out;
    for (auto& [aid, _] : it->second.lora_adapters) out.push_back(aid);
    return out;
}

std::vector<std::vector<float>> TextEngine::embed(
    const std::string& model_id,
    const std::vector<std::string>& inputs
) {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    auto it = pimpl_->loaded_models_.find(model_id);
    if (it == pimpl_->loaded_models_.end() || !it->second.loaded) return {};
    if (!pimpl_->fresh_context(it->second)) return {};
    if (it->second.lmodel) return pimpl_->llama_embed(it->second, inputs);
    return {};
}

std::vector<TextEngine::ModelInfo> TextEngine::list_loaded_models() const {
    std::lock_guard<std::recursive_mutex> lock(pimpl_->models_mutex_);
    std::vector<ModelInfo> result;
    for (const auto& [id, model] : pimpl_->loaded_models_) {
        ModelInfo info;
        info.id = model.id;
        info.model_id = model.model_id;
        info.config = model.config;
        info.format = model.format;
        info.loaded = model.loaded;
        info.n_gpu_layers_used = model.n_gpu_layers_used;
        info.loaded_at = model.loaded_at;
        info.mem_size = model.mem_size;
        result.push_back(info);
    }
    return result;
}

} // namespace barskuy::engine