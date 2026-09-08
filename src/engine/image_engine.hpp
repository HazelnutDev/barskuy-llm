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

// Image generation parameters
struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 512;
    int steps = 20;
    float guidance_scale = 7.0f;
    int seed = -1;
    int batch_size = 1;
    
    // Scheduler
    enum class SchedulerType {
        DDIM,
        Euler,
        EulerAncestral,
        DPMpp2M
    } scheduler = SchedulerType::DDIM;
    
    // Model specific
    std::string model_id;
};

// Image generation result
struct ImageGenerationResult {
    std::vector<std::vector<uint8_t>> images;  // RGBA pixel data
    std::vector<int> widths;
    std::vector<int> heights;
    int seed_used = 0;
    int steps_used = 0;
};

// Diffusion model architecture
enum class DiffusionArchitecture {
    SD15,    // Stable Diffusion 1.5
    SDXL,    // Stable Diffusion XL
    FLUX     // FLUX (future)
};

// Forward declarations
struct DiffusionModel;
struct TextEncoder;
struct UNet2DConditionModel;
struct VAEDecoder;

// Image Engine - orchestrates the diffusion pipeline
class ImageEngine {
public:
    struct Config {
        std::string models_dir = "./models";
        size_t max_vram_bytes = 6ull * 1024 * 1024 * 1024;  // 6GB default
        bool enable_offload = true;
        int default_steps = 20;
        float default_guidance = 7.0f;
    };

    explicit ImageEngine(const Config& config = {});
    ~ImageEngine();

    bool initialize();
    void shutdown();
    void set_cpu_threads(int n) { cpu_threads_ = n; }
    // Register id->path without loading tensors (sd backend loads lazily).
    bool register_model(const std::string& model_id, const std::string& model_path);

    // Load a diffusion model (safetensors)
    bool load_model(const std::string& model_id, const std::string& model_path);
    bool unload_model(const std::string& model_id);
    bool is_loaded(const std::string& model_id) const;

    // Generate image
    engine::ImageGenerationResult generate(const ImageGenerationParams& params);

    // Async generation with callback
    using GenerationCallback = std::function<void(const engine::ImageGenerationResult&, bool is_final)>;
    void generate_async(const ImageGenerationParams& params, GenerationCallback callback);

    // List loaded models
    struct ModelInfo {
        std::string id;
        std::string path;
        DiffusionArchitecture arch;
        bool loaded;
        size_t vram_usage = 0;
    };
    std::vector<ModelInfo> list_models() const;

    // Get model info
    std::optional<ModelInfo> get_model_info(const std::string& model_id) const;

private:
    Config config_;
    std::unordered_map<std::string, std::unique_ptr<DiffusionModel>> models_;
    mutable std::recursive_mutex models_mutex_; // recursive: shutdown()->unload_model()

    // VRAM management
    size_t current_vram_usage_ = 0;
    int cpu_threads_ = 6;
    void maybe_offload();

    // Pipeline components
    std::unique_ptr<TextEncoder> create_text_encoder(const DiffusionModel& model);
    std::unique_ptr<UNet2DConditionModel> create_unet(const DiffusionModel& model);
    std::unique_ptr<VAEDecoder> create_vae(const DiffusionModel& model);

    // Real inference via stable-diffusion.cpp (Fase 10, BARSKUY_WITH_SD).
    // False when unavailable -> caller falls back to preview placeholder.
    bool sd_generate(const ImageGenerationParams& params, ImageGenerationResult& out);

    // Schedulers
    public:
    struct Scheduler {
        virtual ~Scheduler() = default;
        virtual void set_timesteps(int steps) = 0;
        virtual std::vector<float> step(const std::vector<float>& noise_pred, 
                                         int timestep, 
                                         const std::vector<float>& sample) = 0;
    };
    
    std::unique_ptr<Scheduler> create_scheduler(ImageGenerationParams::SchedulerType type);
};

// Diffusion model container
struct DiffusionModel {
    std::string id;
    std::string path;
    DiffusionArchitecture arch = DiffusionArchitecture::SD15;
    bool loaded = false;
    
    // Model components (ggml tensors)
    struct ggml_context* ctx = nullptr;
    std::unordered_map<std::string, struct ggml_tensor*> tensors;

    // stable-diffusion.cpp backend (Fase 10): opaque sd_ctx_t*, created
    // lazily on first generate. Storage pins path strings for sd.cpp.
    void* sd_ctx = nullptr;
    std::string sd_path_storage;
    std::string sd_vae_storage;
    std::string sd_llm_storage;
    
    // Model config
    int unet_in_channels = 4;
    int unet_out_channels = 4;
    int unet_sample_size = 64;  // latent size
    int vae_scale_factor = 8;
    int text_encoder_max_length = 77;
    int text_encoder_hidden_size = 768;
    int unet_attention_head_dim = 8;
    std::vector<int> unet_block_out_channels = {320, 640, 1280, 1280};
    std::vector<int> unet_down_block_types = {0, 0, 0, 1};  // 0=CrossAttnDownBlock2D, 1=DownBlock2D
    std::vector<int> unet_up_block_types = {1, 0, 0, 0};
    
    size_t vram_usage = 0;
};

} // namespace barskuy::engine