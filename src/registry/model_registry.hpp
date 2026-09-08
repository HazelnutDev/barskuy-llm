#pragma once

#include <string>
#include <vector>
#include <optional>
#include <sqlite3.h>
#include <nlohmann/json.hpp>

namespace barskuy::registry {

struct Model {
    std::string id;
    std::string path;
    std::string format; // "gguf" or "safetensors"
    std::string architecture;
    std::string quantization;
    std::vector<std::string> capabilities;
    bool loaded = false;
    int64_t created_at = 0;
};

struct ApiKey {
    std::string id;
    std::string key_hash;
    std::string label;
    int64_t created_at = 0;
    std::optional<int64_t> revoked_at;
};

struct GenerationJob {
    std::string id;
    std::string type; // "image" or "video"
    std::string model_id;
    std::string status; // "queued", "processing", "completed", "failed"
    int progress = 0;
    nlohmann::json request_params;
    std::optional<std::string> result_path;
    std::optional<std::string> error;
    int64_t created_at = 0;
    std::optional<int64_t> completed_at;
};

struct UsageLog {
    int64_t id = 0;
    std::string api_key_id;
    std::string endpoint;
    std::optional<std::string> model_id;
    int64_t tokens_or_units = 0;
    int64_t timestamp = 0;
};

struct ArchitectureDefinition {
    std::string id;
    std::string name;
    nlohmann::json tensor_mapping;
    std::string graph_template_ref;
    int64_t created_at = 0;
};

class ModelRegistry {
public:
    explicit ModelRegistry(const std::string& db_path);
    ~ModelRegistry();

    bool initialize();

    // Models
    bool register_model(const Model& model);
    bool unregister_model(const std::string& id);
    std::optional<Model> get_model(const std::string& id);
    std::vector<Model> list_models();
    bool update_model_loaded(const std::string& id, bool loaded);

    // API Keys
    bool create_api_key(const ApiKey& key);
    bool revoke_api_key(const std::string& id);
    std::optional<ApiKey> get_api_key_by_hash(const std::string& key_hash);
    std::vector<ApiKey> list_api_keys();

    // Generation Jobs
    bool create_job(const GenerationJob& job);
    bool update_job_status(const std::string& id, const std::string& status, int progress = -1);
    bool complete_job(const std::string& id, const std::string& result_path);
    bool fail_job(const std::string& id, const std::string& error);
    std::optional<GenerationJob> get_job(const std::string& id);
    std::vector<GenerationJob> list_jobs(const std::optional<std::string>& status = {},
                                          const std::optional<std::string>& type = {});
    bool cleanup_old_jobs(int days = 30);

    // Usage Logs
    bool log_usage(const UsageLog& log);
    std::vector<UsageLog> get_usage_logs(const std::string& api_key_id, int64_t since);

    // Architecture Definitions
    bool register_architecture(const ArchitectureDefinition& def);
    std::optional<ArchitectureDefinition> get_architecture(const std::string& name);
    std::vector<ArchitectureDefinition> list_architectures();

private:
    bool exec(const std::string& sql);
    bool exec_prepared(const std::string& sql, std::function<void(sqlite3_stmt*)> binder);

    std::string db_path_;
    sqlite3* db_ = nullptr;
};

} // namespace barskuy::registry