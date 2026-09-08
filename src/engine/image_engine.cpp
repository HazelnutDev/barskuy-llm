#include "image_engine.hpp"
#include "core/logger.hpp"
#include "preview_media.hpp"
#include "safetensors_loader.hpp"
#ifdef BARSKUY_WITH_SD
#include "stable-diffusion.h"
#endif
#include <filesystem>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <thread>

namespace barskuy::engine {

// DDPM Scheduler implementation
struct DDPMScheduler : ImageEngine::Scheduler {
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    std::vector<float> alphas_cumprod_prev;
    std::vector<float> sqrt_alphas_cumprod;
    std::vector<float> sqrt_one_minus_alphas_cumprod;
    std::vector<int> timesteps;
    
    void set_timesteps(int steps) override {
        timesteps.clear();
        for (int i = 999; i >= 0; i -= 1000 / steps) {
            timesteps.push_back(i);
        }
    }
    
    std::vector<float> step(const std::vector<float>& noise_pred, 
                           int timestep, 
                           const std::vector<float>& sample) override {
        // DDIM step (simplified)
        float alpha_prod_t = alphas_cumprod[timestep];
        float alpha_prod_t_prev = (timestep > 0) ? alphas_cumprod[timestep - 1] : 1.0f;
        
        float sqrt_alpha_prod_t = sqrt_alphas_cumprod[timestep];
        float sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alphas_cumprod[timestep];
        
        std::vector<float> pred_original_sample(sample.size());
        std::vector<float> pred_sample_direction(sample.size());
        std::vector<float> prev_sample(sample.size());
        
        for (size_t i = 0; i < sample.size(); ++i) {
            pred_original_sample[i] = (sample[i] - sqrt_one_minus_alpha_prod_t * noise_pred[i]) / sqrt_alpha_prod_t;
            pred_sample_direction[i] = std::sqrt(1.0f - alphas_cumprod[timestep - 1]) * noise_pred[i];
            prev_sample[i] = std::sqrt(alphas_cumprod[timestep - 1]) * pred_original_sample[i] + pred_sample_direction[i];
        }
        
        return prev_sample;
    }
    
    void initialize(int num_train_timesteps = 1000) {
        betas.resize(num_train_timesteps);
        alphas.resize(num_train_timesteps);
        alphas_cumprod.resize(num_train_timesteps);
        alphas_cumprod_prev.resize(num_train_timesteps);
        sqrt_alphas_cumprod.resize(num_train_timesteps);
        sqrt_one_minus_alphas_cumprod.resize(num_train_timesteps);
        
        // Linear beta schedule
        float beta_start = 0.00085f;
        float beta_end = 0.012f;
        for (int i = 0; i < num_train_timesteps; ++i) {
            betas[i] = beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1);
            alphas[i] = 1.0f - betas[i];
        }
        
        alphas_cumprod[0] = alphas[0];
        for (int i = 1; i < num_train_timesteps; ++i) {
            alphas_cumprod[i] = alphas_cumprod[i-1] * alphas[i];
        }
        
        alphas_cumprod_prev[0] = 1.0f;
        for (int i = 1; i < num_train_timesteps; ++i) {
            alphas_cumprod_prev[i] = alphas_cumprod[i-1];
        }
        
        for (int i = 0; i < num_train_timesteps; ++i) {
            sqrt_alphas_cumprod[i] = std::sqrt(alphas_cumprod[i]);
            sqrt_one_minus_alphas_cumprod[i] = std::sqrt(1.0f - alphas_cumprod[i]);
        }
    }
};

ImageEngine::ImageEngine(const Config& config) : config_(config) {}

ImageEngine::~ImageEngine() {
    shutdown();
}

bool ImageEngine::initialize() {
    core::Logger::info("Initializing ImageEngine...");
    
    std::filesystem::create_directories(config_.models_dir);
    
    core::Logger::info("ImageEngine initialized (models_dir: {}, max_vram: {} MB)", 
        config_.models_dir, config_.max_vram_bytes / (1024*1024));
    return true;
}

void ImageEngine::shutdown() {
    std::lock_guard<std::recursive_mutex> lock(models_mutex_);
    for (auto& [id, model] : models_) {
        unload_model(id);
    }
    models_.clear();
    current_vram_usage_ = 0;
    core::Logger::info("ImageEngine shutdown complete");
}

bool ImageEngine::load_model(const std::string& model_id, const std::string& model_path) {
    std::lock_guard<std::recursive_mutex> lock(models_mutex_);
    
    if (models_.find(model_id) != models_.end()) {
        core::Logger::warn("Model {} already loaded", model_id);
        return true;
    }
    
    core::Logger::info("Loading diffusion model: {} from {}", model_id, model_path);
    
    if (!std::filesystem::exists(model_path)) {
        core::Logger::error("Model file not found: {}", model_path);
        return false;
    }
    
    // Check VRAM budget
    std::filesystem::file_size(model_path);  // Get file size for estimation
    
    auto model = std::make_unique<DiffusionModel>();
    model->id = model_id;
    model->path = model_path;
    
    // Parse safetensors header to get model info
    SafetensorsLoader loader;
    SafetensorsMetadata metadata;
    if (!loader.parse_header(model_path, metadata)) {
        core::Logger::error("Failed to parse model header: {}", model_path);
        return false;
    }
    
    // Detect architecture from tensor names
    bool is_sdxl = false;
    for (const auto& [name, info] : metadata.tensors) {
        if (name.find("conditioner") != std::string::npos || 
            name.find("text_encoder_2") != std::string::npos) {
            is_sdxl = true;
            break;
        }
    }
    model->arch = is_sdxl ? DiffusionArchitecture::SDXL : DiffusionArchitecture::SD15;
    
    // Create ggml context
    struct ggml_init_params params;
    params.mem_size = 1024 * 1024 * 1024;  // 1GB initial
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    model->ctx = ggml_init(params);
    if (!model->ctx) {
        core::Logger::error("Failed to create ggml context for model");
        return false;
    }
    
    // Load tensors
    if (!loader.load_tensors(model_path, model->ctx, model->tensors, metadata)) {
        core::Logger::error("Failed to load model tensors");
        ggml_free(model->ctx);
        return false;
    }
    
    model->loaded = true;
    model->vram_usage = ggml_used_mem(model->ctx);
    current_vram_usage_ += model->vram_usage;
    
    // Configure model based on architecture
    if (model->arch == DiffusionArchitecture::SDXL) {
        model->text_encoder_max_length = 77;
        model->text_encoder_hidden_size = 2048;  // SDXL uses larger text encoder
    }
    
    models_[model_id] = std::move(model);
    
    core::Logger::info("Model loaded: {} (arch: {}, VRAM: {} MB)", 
        model_id, 
        is_sdxl ? "SDXL" : "SD1.5",
        models_[model_id]->vram_usage / (1024*1024));
    
    // Offload if needed
    maybe_offload();
    
    return true;
}

bool ImageEngine::register_model(const std::string& model_id, const std::string& model_path) {
    if (model_id.empty() || model_path.empty() || !std::filesystem::exists(model_path)) return false;
    std::lock_guard<std::recursive_mutex> lock(models_mutex_);
    auto it = models_.find(model_id);
    if (it != models_.end()) {
        it->second->path = model_path; // re-point, keep resident sd ctx if same file
        return true;
    }
    auto model = std::make_unique<DiffusionModel>();
    model->id = model_id;
    model->path = model_path;
    model->loaded = true; // usable (sd backend lazy-loads weights itself)
    std::string lp = model_path;
    for (auto& c : lp) c = (char)std::tolower((unsigned char)c);
    if (lp.find("xl") != std::string::npos) model->arch = DiffusionArchitecture::SDXL;
    models_[model_id] = std::move(model);
    core::Logger::info("Image model registered: {} -> {}", model_id, model_path);
    return true;
}

bool ImageEngine::unload_model(const std::string& model_id) {
    std::lock_guard<std::recursive_mutex> lock(models_mutex_);
    
    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return false;
    }

#ifdef BARSKUY_WITH_SD
    if (it->second->sd_ctx) {
        free_sd_ctx((sd_ctx_t*)it->second->sd_ctx);
        it->second->sd_ctx = nullptr;
    }
#endif

    current_vram_usage_ -= it->second->vram_usage;
    if (it->second->ctx) {
        ggml_free(it->second->ctx);
    }
    models_.erase(it);
    
    core::Logger::info("Model unloaded: {}", model_id);
    return true;
}

bool ImageEngine::is_loaded(const std::string& model_id) const {
    std::lock_guard<std::recursive_mutex> lock(models_mutex_);
    auto it = models_.find(model_id);
    return it != models_.end() && it->second->loaded;
}

void ImageEngine::maybe_offload() {
    if (current_vram_usage_ <= config_.max_vram_bytes) {
        return;
    }
    
    // Simple LRU offload - unload least recently used model
    // In a real implementation, we'd offload individual components
    core::Logger::warn("VRAM limit exceeded ({} MB / {} MB), consider offloading", 
        current_vram_usage_ / (1024*1024), config_.max_vram_bytes / (1024*1024));
}

std::optional<ImageEngine::ModelInfo> ImageEngine::get_model_info(const std::string& model_id) const {
    std::lock_guard<std::recursive_mutex> lock(models_mutex_);
    auto it = models_.find(model_id);
    if (it == models_.end()) return std::nullopt;
    
    ModelInfo info;
    info.id = it->second->id;
    info.path = it->second->path;
    info.arch = it->second->arch;
    info.loaded = it->second->loaded;
    info.vram_usage = it->second->vram_usage;
    return info;
}

std::vector<ImageEngine::ModelInfo> ImageEngine::list_models() const {
    std::lock_guard<std::recursive_mutex> lock(models_mutex_);
    std::vector<ModelInfo> result;
    for (const auto& [id, model] : models_) {
        ModelInfo info;
        info.id = model->id;
        info.path = model->path;
        info.arch = model->arch;
        info.loaded = model->loaded;
        info.vram_usage = model->vram_usage;
        result.push_back(info);
    }
    return result;
}

std::unique_ptr<ImageEngine::Scheduler> ImageEngine::create_scheduler(ImageGenerationParams::SchedulerType type) {
    auto scheduler = std::make_unique<DDPMScheduler>();
    scheduler->initialize(1000);
    return scheduler;
}

engine::ImageGenerationResult ImageEngine::generate(const ImageGenerationParams& params) {
#ifdef BARSKUY_WITH_SD
    ImageGenerationResult real;
    if (sd_generate(params, real)) return real;
    core::Logger::warn("sd backend unavailable/failed for model={}, preview fallback",
        params.model_id);
#endif
    // PREVIEW mode (Fase 8 honest stub): full diffusion pipeline pending,
    // render deterministic procedural placeholder instead of failing.
    int w = std::clamp(params.width, 64, 2048);
    int h = std::clamp(params.height, 64, 2048);
    int n = std::clamp(params.batch_size, 1, 2);
    ImageGenerationResult result;
    uint64_t seed = (params.seed >= 0 ? (uint64_t)params.seed : (uint64_t)std::random_device{}())
        ^ preview::hash_str(params.prompt);
    for (int i = 0; i < n; ++i) {
        result.images.push_back(preview::render_frame(seed + (uint64_t)i * 7919, w, h));
        result.widths.push_back(w);
        result.heights.push_back(h);
    }
    result.seed_used = params.seed >= 0 ? params.seed : (int)(seed & 0x7FFFFFFF);
    result.steps_used = params.steps;

    core::Logger::info("Image PREVIEW rendered: {}x{} model={} (diffusion pending)", w, h, params.model_id);
    return result;
}

#ifdef BARSKUY_WITH_SD
static std::string sd_to_lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
// true = checkpoint utuh (nama berprefix model./conditioner./cond_stage_/
// first_stage_), false = file terpisah nama polos (perlu prefix internal
// "model.diffusion_model." via diffusion_model_path). Cek ≤128 nama tensor.
static bool sd_is_full_checkpoint(const std::string& path) {
    auto looks_full = [](const std::string& n) {
        return n.rfind("model.", 0) == 0 || n.rfind("conditioner", 0) == 0 ||
               n.rfind("cond_stage", 0) == 0 || n.rfind("first_stage", 0) == 0 ||
               n.rfind("text_encoders", 0) == 0 || n.rfind("clip_", 0) == 0;
    };
    std::string lp = sd_to_lower(path);
    try {
        if (lp.size() >= 5 && lp.compare(lp.size() - 5, 5, ".gguf") == 0) {
            std::ifstream f(path, std::ios::binary);
            if (!f) return true;
            char magic[4];
            f.read(magic, 4);
            if (std::string(magic, 4) != "GGUF") return true;
            uint32_t ver = 0;
            uint64_t nt = 0, nk = 0;
            f.read((char*)&ver, 4);
            f.read((char*)&nt, 8);
            f.read((char*)&nk, 8);
            for (uint64_t i = 0; i < nt && i < 128; ++i) {
                uint64_t ln = 0;
                f.read((char*)&ln, 8);
                if (!f || ln == 0 || ln > 512) break;
                std::string name(ln, '\0');
                f.read(name.data(), (std::streamsize)ln);
                if (!f) break;
                uint32_t nd = 0, tp = 0;
                uint64_t off = 0;
                f.read((char*)&nd, 4);
                f.read((char*)&tp, 4);
                f.read((char*)&off, 8);
                if (!f || nd > 8) break;
                f.seekg((std::streamoff)(8 * nd), std::ios::cur);
                if (looks_full(name)) return true;
            }
            return false;
        }
        if (lp.size() >= 12 && lp.compare(lp.size() - 12, 12, ".safetensors") == 0) {
            std::ifstream f(path, std::ios::binary);
            if (!f) return true;
            uint64_t n = 0;
            f.read((char*)&n, 8);
            if (!f || n == 0 || n > 64 * 1024 * 1024) return true;
            std::string hdr(n, '\0');
            f.read(hdr.data(), (std::streamsize)n);
            if (!f) return true;
            // cari kunci "xxx": tanpa JSON penuh (cepat, cukup)
            for (size_t p = hdr.find('"'); p != std::string::npos; p = hdr.find('"', p + 1)) {
                size_t e = hdr.find('"', p + 1);
                if (e == std::string::npos) break;
                std::string k = hdr.substr(p + 1, e - p - 1);
                if (k != "__metadata__" && looks_full(k)) return true;
                p = e;
                if (e > 65536) break; // cukup sampel awal
            }
            return false;
        }
    } catch (...) {}
    return true;
}
static void sd_log_fwd(enum sd_log_level_t level, const char* text, void* data) {
    (void)data;
    if (!text || !*text) return;
    if (level >= SD_LOG_WARN) core::Logger::warn("[sd] {}", text);
    else core::Logger::info("[sd] {}", text);
}
bool ImageEngine::sd_generate(const ImageGenerationParams& params, ImageGenerationResult& out) {
    std::string model_path;
    {
        std::lock_guard<std::recursive_mutex> lock(models_mutex_);
        auto it = models_.find(params.model_id);
        if (it == models_.end()) {
            core::Logger::warn("sd_generate: model '{}' not registered (register via /v1/models/register)", params.model_id);
            return false;
        }
        model_path = it->second->path;
    }
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        core::Logger::warn("sd_generate: model file missing: {}", model_path);
        return false;
    }
    // Lazy per-model sd context (weights stay resident across requests).
    sd_ctx_t* ctx = nullptr;
    {
        std::lock_guard<std::recursive_mutex> lock(models_mutex_);
        auto it = models_.find(params.model_id);
        if (it == models_.end()) return false;
        if (it->second->sd_ctx) {
            ctx = (sd_ctx_t*)it->second->sd_ctx;
        } else {
            it->second->sd_path_storage = model_path;
            std::string lp = sd_to_lower(model_path);
            bool is_zimage = lp.find("z-image") != std::string::npos ||
                             lp.find("z_image") != std::string::npos;
            // Z-Image ala ComfyUI: diffusion + VAE + LLM terpisah. Cari
            // pasangan di models dir (vae: *ae.safetensors|*vae*; llm: *qwen3-4b*).
            std::string vae_path, llm_path;
            if (is_zimage) {
                std::error_code ec;
                // Cari rekursif di models/image/vae dan models/image/llm (baru),
                // fallback ke flat models/ lama. Prioritaskan subfolder baru.
                std::vector<std::string> search_roots;
                search_roots.push_back(config_.models_dir + "/image/vae");
                search_roots.push_back(config_.models_dir + "/image/llm");
                search_roots.push_back(config_.models_dir + "/image");
                search_roots.push_back(config_.models_dir);
                for (auto& root : search_roots) {
                    if (!vae_path.empty() && !llm_path.empty()) break;
                    for (auto& e : std::filesystem::recursive_directory_iterator(root, ec)) {
                        if (ec) break;
                        if (!e.is_regular_file()) continue;
                        std::string fn = e.path().filename().string();
                        std::string fl = sd_to_lower(fn);
                    bool is_sft = fl.size() >= 12 && fl.compare(fl.size() - 12, 12, ".safetensors") == 0;
                    bool is_gg = fl.size() >= 5 && fl.compare(fl.size() - 5, 5, ".gguf") == 0;
                    if (vae_path.empty() && is_sft &&
                        (fl.find("ae.safetensors") != std::string::npos || fl.find("vae") != std::string::npos))
                        vae_path = e.path().string();
                    // Qwen LLM untuk Z-Image: terima qwen3-4b, qwen2.5-0.5b, atau qwen apapun
                    if (llm_path.empty() && is_gg && fl.find("qwen") != std::string::npos)
                        llm_path = e.path().string();
                        if (!vae_path.empty() && !llm_path.empty()) break;
                    }
                }
                it->second->sd_vae_storage = vae_path;
                it->second->sd_llm_storage = llm_path;
                if (vae_path.empty() || llm_path.empty()) {
                    core::Logger::error("sd z-image needs vae+llm files (vae={}, llm={})",
                        vae_path.empty() ? "-" : vae_path, llm_path.empty() ? "-" : llm_path);
                    return false;
                }
            }
            // File utuh (checkpoint safetensors / paket GGUF) lewat model_path
            // TANPA prefix. diffusion_model_path menambah prefix
            // "model.diffusion_model." dan hanya untuk file UNet terpisah.
            sd_ctx_params_t cp;
            sd_ctx_params_init(&cp);
            // Utuh (prefix model./conditioner./...) -> model_path TANPA prefix
            // tambahan. Terpisah nama polos (DiT Z-Image) -> diffusion_model_path
            // agar sd.cpp bubuhkan prefix "model.diffusion_model." sendiri.
            if (sd_is_full_checkpoint(model_path)) {
                cp.model_path = it->second->sd_path_storage.c_str();
            } else {
                core::Logger::info("sd split-file layout, via diffusion_model_path");
                cp.diffusion_model_path = it->second->sd_path_storage.c_str();
            }
            if (!it->second->sd_vae_storage.empty()) {
                cp.vae_path = it->second->sd_vae_storage.c_str();
                core::Logger::info("sd vae: {}", it->second->sd_vae_storage);
            }
            if (!it->second->sd_llm_storage.empty()) {
                cp.llm_path = it->second->sd_llm_storage.c_str();
                core::Logger::info("sd llm: {}", it->second->sd_llm_storage);
            }
            bool is_gguf = lp.size() >= 5 && lp.compare(lp.size() - 5, 5, ".gguf") == 0;
            if (!is_gguf) {
                // fp16 safetensors (6.5GB) tak muat VRAM 6GB: quant on-load Q8.
                // File GGUF sudah terquant -> wtype dibiarkan default.
                cp.wtype = SD_TYPE_Q8_0;
            }
            cp.n_threads = cpu_threads_ > 0 ? cpu_threads_ : 6;
            cp.flash_attn = true;
            cp.eager_load = true;
#ifdef GGML_USE_VULKAN
            // Hybrid optimal: DiT di RTX (3.8GB, compute-bound), TE+VAE di RAM
            // (3.7GB, sekali pakai). Peak 3.8GB VRAM, no OOM, 1 segmen monolitik
            // ~20 detik. Semua di VRAM (7.4GB) pasti OOM (butuh 1.2GB kontigu
            // tak ada) — buktinya barusan gagal segment 2/5.
            cp.diffusion_flash_attn = true;
            if (is_zimage) {
                cp.backend = "diffusion=vulkan1,te=cpu,vae=cpu";
                cp.params_backend = "te=cpu,vae=cpu";
                cp.disable_prefetch = true;
            } else {
                cp.backend = "diffusion=vulkan1";
                cp.params_backend = "te=vulkan1,vae=vulkan1";
                cp.max_vram = "vulkan1=5";
            }
#endif
            core::Logger::info("sd loading: {} (this takes a while first time)", model_path);
            sd_set_log_callback(sd_log_fwd, nullptr);
            sd_ctx_t* nc = new_sd_ctx(&cp);
            if (!nc) {
                core::Logger::error("sd new_sd_ctx failed for {}", model_path);
                return false;
            }
            it->second->sd_ctx = nc;
            ctx = nc;
        }
    }
    int w = std::clamp(params.width, 64, 2048);
    int h = std::clamp(params.height, 64, 2048);
    int steps = params.steps;
    float guidance = params.guidance_scale;
    enum sample_method_t method = EULER_A_SAMPLE_METHOD;
    // ComfyUI preset (gambar user): Z-Turbo cfg 1.0, res_multistep+simple
    // + flow 3.0, steps 12 (user request, ComfyUI 9 tapi 12 lebih tajam).
    // Turbo default hanya kalau client pakai default 20/512.
    std::string mlp = sd_to_lower(model_path);
    bool is_zturbo = mlp.find("turbo") != std::string::npos &&
        (mlp.find("z-image") != std::string::npos || mlp.find("z_image") != std::string::npos);
    std::string eff_prompt = params.prompt;
    // 3.2.1 revert: unsharp 0.35 bikin halo, user nilai 3.2.0 (12 step)
    // lebih bagus. Balik ke 12 step tanpa post-process — difusi alami.
    if (mlp.find("turbo") != std::string::npos) {
        if (steps < 12) steps = (w * h >= 2048*2048 ? 16 : 12);
        if (guidance == 7.0f) guidance = 1.0f;
        method = EULER_A_SAMPLE_METHOD;
    } else {
        switch (params.scheduler) {
            case ImageGenerationParams::SchedulerType::Euler: method = EULER_SAMPLE_METHOD; break;
            case ImageGenerationParams::SchedulerType::EulerAncestral: method = EULER_A_SAMPLE_METHOD; break;
            case ImageGenerationParams::SchedulerType::DPMpp2M: method = DPMPP2M_SAMPLE_METHOD; break;
            default: method = EULER_A_SAMPLE_METHOD; break;
        }
    }
    steps = std::clamp(steps, 1, 100);
    int n = std::clamp(params.batch_size, 1, 2);
    int64_t seed = params.seed >= 0 ? (int64_t)params.seed
        : (int64_t)(std::random_device{}() & 0x7FFFFFFF);
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.prompt = params.prompt.c_str();
    gp.negative_prompt = params.negative_prompt.c_str();
    gp.width = w;
    gp.height = h;
    // Z-Turbo ComfyUI: res_multistep + simple + shift 3 (proven tajam)
    if (is_zturbo) {
        gp.sample_params.sample_method = RES_MULTISTEP_SAMPLE_METHOD;
        gp.sample_params.scheduler = SIMPLE_SCHEDULER;
        gp.sample_params.flow_shift = 3.0f;
    } else {
        gp.sample_params.sample_method = method;
    }
    gp.sample_params.sample_steps = steps;
    gp.sample_params.guidance.txt_cfg = guidance;
    // VAE z-image 16ch ~6.9GB tanpa tiling di 1024 → OOM / 103s CPU.
    // Tile 512 -> ~1500MB/tile dan ~25s decode (vs 768 -> 6657MB/103s).
    // Vulkan VAE malah failed tile weight prep, jadi tetap cpu + tile kecil.
    if (w >= 1024 || h >= 1024) {
        gp.vae_tiling_params.enabled = true;
        gp.vae_tiling_params.tile_size_x = 512;
        gp.vae_tiling_params.tile_size_y = 512;
    }
    gp.seed = seed;
    gp.batch_count = n;
    sd_image_t* images = nullptr;
    int num_images = 0;
    core::Logger::info("sd txt2img: {}x{} steps={} cfg={} seed={} model={}",
        w, h, steps, guidance, (long long)seed, params.model_id);
    if (!generate_image(ctx, &gp, &images, &num_images) || !images || num_images <= 0) {
        core::Logger::error("sd generate_image failed");
        return false;
    }
    for (int i = 0; i < num_images; ++i) {
        uint32_t iw = images[i].width, ih = images[i].height;
        uint8_t* px = images[i].data;
        if (!px || iw == 0 || ih == 0) continue;
        std::vector<uint8_t> rgba((size_t)iw * ih * 4);
        if (images[i].channel >= 3) {
            for (size_t k = 0; k < (size_t)iw * ih; ++k) {
                rgba[k * 4] = px[k * images[i].channel];
                rgba[k * 4 + 1] = px[k * images[i].channel + 1];
                rgba[k * 4 + 2] = px[k * images[i].channel + 2];
                rgba[k * 4 + 3] = 255;
            }
        } else {
            for (size_t k = 0; k < (size_t)iw * ih; ++k) {
                rgba[k * 4] = rgba[k * 4 + 1] = rgba[k * 4 + 2] = px[k];
                rgba[k * 4 + 3] = 255;
            }
        }
        out.images.push_back(std::move(rgba));
        out.widths.push_back((int)iw);
        out.heights.push_back((int)ih);
    }
    free_sd_images(images, num_images);
    if (out.images.empty()) return false;
    out.seed_used = (int)(seed & 0x7FFFFFFF);
    out.steps_used = steps;
    core::Logger::info("sd txt2img done: {} image(s) model={}", (int)out.images.size(), params.model_id);
    return true;
}
#endif

void ImageEngine::generate_async(const ImageGenerationParams& params, GenerationCallback callback) {
    // Run generation in background thread
    std::thread([this, params, callback]() {
        auto result = generate(params);
        callback(result, true);
    }).detach();
}

} // namespace barskuy::engine