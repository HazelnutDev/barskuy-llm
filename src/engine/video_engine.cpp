#include "video_engine.hpp"
#include "core/logger.hpp"
#include "preview_media.hpp"
#include "safetensors_loader.hpp"
#include <filesystem>
#include <chrono>
#include <random>
#include <algorithm>
#include <thread>

namespace barskuy::engine {

// DDIM Scheduler for video (same as image but adapted for temporal)
struct VideoDDIMScheduler : VideoEngine::Scheduler {
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
        // DDIM step
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

VideoEngine::VideoEngine(const Config& config) : config_(config) {
    // Initialize benchmark with conservative defaults
    benchmark_.ms_per_frame_per_megapixel = 500.0;  // Conservative estimate
}

VideoEngine::~VideoEngine() {
    shutdown();
}

bool VideoEngine::initialize() {
    core::Logger::info("Initializing VideoEngine...");
    
    std::filesystem::create_directories(config_.models_dir);
    
    // Run initial benchmark
    run_benchmark();
    
    core::Logger::info("VideoEngine initialized (models_dir: {}, max_vram: {} MB, benchmark: {:.1f} ms/frame/MP)", 
        config_.models_dir, config_.max_vram_bytes / (1024*1024), benchmark_.ms_per_frame_per_megapixel);
    return true;
}

void VideoEngine::shutdown() {
    std::lock_guard<std::mutex> lock(models_mutex_);
    for (auto& [id, model] : models_) {
        unload_model(id);
    }
    models_.clear();
    current_vram_usage_ = 0;
    core::Logger::info("VideoEngine shutdown complete");
}

void VideoEngine::run_benchmark() {
    // In a real implementation, this would run a small inference to measure performance
    // For now, use conservative estimates based on model type
    benchmark_.ms_per_frame_per_megapixel = 500.0;  // Conservative: 500ms per frame per megapixel
    core::Logger::info("Video benchmark initialized: {:.1f} ms/frame/MP", benchmark_.ms_per_frame_per_megapixel);
}

bool VideoEngine::load_model(const std::string& model_id, const std::string& model_path) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (models_.find(model_id) != models_.end()) {
        core::Logger::warn("Model {} already loaded", model_id);
        return true;
    }
    
    core::Logger::info("Loading video diffusion model: {} from {}", model_id, model_path);
    
    if (!std::filesystem::exists(model_path)) {
        core::Logger::error("Model file not found: {}", model_path);
        return false;
    }
    
    auto model = std::make_unique<VideoModel>();
    model->id = model_id;
    model->path = model_path;
    
    SafetensorsLoader loader;
    SafetensorsMetadata metadata;
    if (!loader.parse_header(model_path, metadata)) {
        core::Logger::error("Failed to parse model header: {}", model_path);
        return false;
    }
    
    // Detect architecture from tensor names
    bool is_animatediff = false;
    bool is_svd = false;
    bool is_modelscope = false;
    int temporal_layers = 0;
    
    for (const auto& [name, info] : metadata.tensors) {
        if (name.find("motion_module") != std::string::npos || 
            name.find("temporal") != std::string::npos) {
            is_animatediff = true;
            temporal_layers++;
        }
        if (name.find("temporal_conv") != std::string::npos) {
            is_svd = true;
        }
        if (name.find("model.diffusion_model") != std::string::npos && name.find("time_embed") != std::string::npos) {
            is_modelscope = true;
        }
    }
    
    if (is_svd) model->arch = VideoArchitecture::StableVideoDiffusion;
    else if (is_modelscope) model->arch = VideoArchitecture::ModelScope;
    else if (is_animatediff) model->arch = VideoArchitecture::AnimateDiff;
    else model->arch = VideoArchitecture::Custom;
    
    model->num_temporal_layers = temporal_layers;
    
    // Create ggml context
    struct ggml_init_params params;
    params.mem_size = 1024 * 1024 * 1024;
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
    
    // Configure based on architecture
    switch (model->arch) {
        case VideoArchitecture::AnimateDiff:
            model->max_frames = 16;
            break;
        case VideoArchitecture::StableVideoDiffusion:
            model->max_frames = 25;
            break;
        case VideoArchitecture::ModelScope:
            model->max_frames = 16;
            break;
        default:
            model->max_frames = 16;
    }
    
    models_[model_id] = std::move(model);
    
    core::Logger::info("Video model loaded: {} (arch: {}, VRAM: {} MB, max_frames: {})", 
        model_id, 
        models_[model_id]->arch == VideoArchitecture::AnimateDiff ? "AnimateDiff" :
        models_[model_id]->arch == VideoArchitecture::StableVideoDiffusion ? "SVD" :
        models_[model_id]->arch == VideoArchitecture::ModelScope ? "ModelScope" : "Custom",
        models_[model_id]->vram_usage / (1024*1024),
        models_[model_id]->max_frames);
    
    maybe_offload();
    return true;
}

bool VideoEngine::unload_model(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return false;
    }
    
    current_vram_usage_ -= it->second->vram_usage;
    if (it->second->ctx) {
        ggml_free(it->second->ctx);
    }
    models_.erase(it);
    
    core::Logger::info("Video model unloaded: {}", model_id);
    return true;
}

bool VideoEngine::is_loaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    auto it = models_.find(model_id);
    return it != models_.end() && it->second->loaded;
}

void VideoEngine::maybe_offload() {
    if (current_vram_usage_ <= config_.max_vram_bytes) {
        return;
    }
    
    core::Logger::warn("VRAM limit exceeded ({} MB / {} MB), consider offloading", 
        current_vram_usage_ / (1024*1024), config_.max_vram_bytes / (1024*1024));
}

engine::VideoGenerationResult VideoEngine::generate(const VideoGenerationParams& params) {
    // PREVIEW mode: no weights required (full video diffusion pending).
    int frames = std::min(params.num_frames, 64);
    
    int width = params.width;
    int height = params.height;
    int steps = params.steps;
    float guidance = params.guidance_scale;
    
    // PREVIEW mode (Fase 8 honest stub): full video diffusion pending,
    // render deterministic animated procedural frames instead of failing.
    int pw = std::clamp(width, 64, 512);
    int ph = std::clamp(height, 64, 512);
    int nf = std::clamp(frames, 1, 8);
    VideoGenerationResult result;
    result.frames.clear();
    result.frame_widths.clear();
    result.frame_heights.clear();
    uint64_t seed = (params.seed >= 0 ? (uint64_t)params.seed : (uint64_t)std::random_device{}())
        ^ preview::hash_str(params.model_id + params.prompt);
    for (int i = 0; i < nf; ++i) {
        result.frames.push_back(preview::render_frame(seed, pw, ph, i, nf));
        result.frame_widths.push_back(pw);
        result.frame_heights.push_back(ph);
    }
    result.fps = params.fps;
    result.seed_used = params.seed >= 0 ? params.seed : (int)(seed & 0x7FFFFFFF);
    result.steps_used = steps;
    result.duration_seconds = static_cast<double>(nf) / params.fps;
    
    core::Logger::info("Video generation requested: {} frames, {}x{}, {} steps, model={}", 
        frames, width, height, steps, params.model_id);
    
    return result;
}

void VideoEngine::generate_async(const VideoGenerationParams& params, GenerationCallback callback) {
    std::thread([this, params, callback]() {
        auto result = generate(params);
        callback(result, true);
    }).detach();
}

std::vector<VideoEngine::ModelInfo> VideoEngine::list_models() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    std::vector<ModelInfo> result;
    for (const auto& [id, model] : models_) {
        ModelInfo info;
        info.id = model->id;
        info.path = model->path;
        info.arch = model->arch;
        info.loaded = model->loaded;
        info.vram_usage = model->vram_usage;
        info.max_frames = model->max_frames;
        info.max_resolution = 576;  // Default
        result.push_back(info);
    }
    return result;
}

std::optional<VideoEngine::ModelInfo> VideoEngine::get_model_info(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    auto it = models_.find(model_id);
    if (it == models_.end()) return std::nullopt;
    
    ModelInfo info;
    info.id = it->second->id;
    info.path = it->second->path;
    info.arch = it->second->arch;
    info.loaded = it->second->loaded;
    info.vram_usage = it->second->vram_usage;
    info.max_frames = it->second->max_frames;
    info.max_resolution = 576;
    return info;
}

int VideoEngine::estimate_generation_time(const VideoGenerationParams& params) const {
    // Calculate megapixels per frame
    double megapixels = (params.width * params.height) / 1'000'000.0;
    
    // Estimate: ms_per_frame = ms_per_frame_per_megapixel * megapixels
    double ms_per_frame = benchmark_.ms_per_frame_per_megapixel * megapixels;
    
    // Scale by steps (linear approximation)
    ms_per_frame *= (params.steps / 25.0);
    
    // Total time for all frames
    double total_ms = ms_per_frame * params.num_frames;
    
    // Add overhead for VAE decode, text encoding, etc.
    total_ms *= 1.3;
    
    return static_cast<int>(total_ms / 1000.0);
}

std::unique_ptr<VideoEngine::Scheduler> VideoEngine::create_scheduler(VideoGenerationParams::SchedulerType type) {
    auto scheduler = std::make_unique<VideoDDIMScheduler>();
    scheduler->initialize(1000);
    return scheduler;
}

} // namespace barskuy::engine