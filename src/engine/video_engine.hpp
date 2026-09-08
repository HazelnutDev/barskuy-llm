#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <optional>
#include <mutex>
#include <ggml.h>

namespace barskuy::engine {

// Video generation parameters
struct VideoGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int num_frames = 16;
    int fps = 8;
    int width = 576;
    int height = 320;
    int steps = 25;
    float guidance_scale = 7.0f;
    int seed = -1;
    std::string model_id;
    
    // Scheduler
    enum class SchedulerType {
        DDIM,
        Euler,
        EulerAncestral,
        DPMpp2M
    } scheduler = SchedulerType::DDIM;
};

// Video generation result
struct VideoGenerationResult {
    std::vector<std::vector<uint8_t>> frames;  // RGB/RGBA frame data
    std::vector<int> frame_widths;
    std::vector<int> frame_heights;
    int fps = 8;
    int seed_used = 0;
    int steps_used = 0;
    double duration_seconds = 0.0;
};

// Video diffusion model architecture
enum class VideoArchitecture {
    AnimateDiff,      // AnimateDiff-style (motion module on SD)
    ModelScope,       // ModelScope T2V
    VideoCrafter,     // VideoCrafter
    StableVideoDiffusion,  // SVD
    Custom
};

// Forward declarations
struct TemporalAttention;
struct VideoUNet;
struct VAEEncoder;
struct VAEDecoder;

// Video Engine - orchestrates video diffusion pipeline
class VideoEngine {
public:
    struct Config {
        std::string models_dir = "./models";
        size_t max_vram_bytes = 6ull * 1024 * 1024 * 1024;  // 6GB default
        bool enable_offload = true;
        int default_steps = 25;
        float default_guidance = 7.0f;
        int max_frames = 64;  // Conservative default for 6GB VRAM
    };

    explicit VideoEngine(const Config& config = {});
    ~VideoEngine();

    bool initialize();
    void shutdown();

    // Load a video diffusion model (safetensors)
    bool load_model(const std::string& model_id, const std::string& model_path);
    bool unload_model(const std::string& model_id);
    bool is_loaded(const std::string& model_id) const;

    // Generate video
    VideoGenerationResult generate(const VideoGenerationParams& params);

    // Async generation with callback
    using GenerationCallback = std::function<void(const VideoGenerationResult&, bool is_final)>;
    void generate_async(const VideoGenerationParams& params, GenerationCallback callback);

    // List loaded models
    struct ModelInfo {
        std::string id;
        std::string path;
        VideoArchitecture arch;
        bool loaded;
        size_t vram_usage = 0;
        int max_frames = 16;
        int max_resolution = 576;
    };
    std::vector<ModelInfo> list_models() const;

    // Get model info
    std::optional<ModelInfo> get_model_info(const std::string& model_id) const;

    // Estimate generation time (eta_seconds)
    int estimate_generation_time(const VideoGenerationParams& params) const;

public:
    // Schedulers
    struct Scheduler {
        virtual ~Scheduler() = default;
        virtual void set_timesteps(int steps) = 0;
        virtual std::vector<float> step(const std::vector<float>& noise_pred, 
                                         int timestep, 
                                         const std::vector<float>& sample) = 0;
    };
    
    std::unique_ptr<Scheduler> create_scheduler(VideoGenerationParams::SchedulerType type);

private:
    Config config_;
    std::unordered_map<std::string, std::unique_ptr<struct VideoModel>> models_;
    mutable std::mutex models_mutex_;

    // VRAM management
    size_t current_vram_usage_ = 0;
    void maybe_offload();

    // Benchmark data for ETA estimation
    struct BenchmarkData {
        double ms_per_frame_per_megapixel = 0.0;
        int benchmark_frames = 16;
        int benchmark_width = 576;
        int benchmark_height = 320;
        int benchmark_steps = 25;
    };
    BenchmarkData benchmark_;
    
    void run_benchmark();

    // Pipeline components (to be implemented)
    // std::unique_ptr<TemporalAttention> create_temporal_attention(const VideoModel& model);
    // std::unique_ptr<VideoUNet> create_video_unet(const VideoModel& model);
    // std::unique_ptr<VAEEncoder> create_vae_encoder(const VideoModel& model);
    // std::unique_ptr<VAEDecoder> create_vae_decoder(const VideoModel& model);
};

// Video model container
struct VideoModel {
    std::string id;
    std::string path;
    VideoArchitecture arch = VideoArchitecture::AnimateDiff;
    bool loaded = false;
    
    // Model components (ggml tensors)
    struct ggml_context* ctx = nullptr;
    std::unordered_map<std::string, struct ggml_tensor*> tensors;
    
    // Model config
    int unet_in_channels = 4;
    int unet_out_channels = 4;
    int unet_sample_size = 64;
    int vae_scale_factor = 8;
    int text_encoder_max_length = 77;
    int text_encoder_hidden_size = 768;
    int temporal_attention_head_dim = 8;
    int num_temporal_layers = 0;
    int max_frames = 16;
    
    size_t vram_usage = 0;
};

} // namespace barskuy::engine