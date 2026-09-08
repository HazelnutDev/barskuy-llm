#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>
#include <nlohmann/json.hpp>

namespace barskuy::registry {

struct ArchitectureDefinition {
    std::string id;
    std::string name;
    nlohmann::json tensor_mapping;
    nlohmann::json attention_config;
    nlohmann::json moe_config;
    std::string graph_template;
    std::unordered_map<std::string, nlohmann::json> extra_config;
    int64_t created_at = 0;
};

class ArchitectureRegistry {
public:
    ArchitectureRegistry() = default;
    ~ArchitectureRegistry() = default;

    // Load all architecture definitions from a directory
    bool load_from_directory(const std::string& directory_path);

    // Get architecture by name (e.g., "llama", "qwen", "gemma")
    std::optional<ArchitectureDefinition> get_architecture(const std::string& name) const;

    // List all registered architectures
    std::vector<ArchitectureDefinition> list_architectures() const;

    // Get tensor name mapping for a specific layer
    std::string map_tensor_name(const std::string& architecture, 
                                const std::string& template_name, 
                                int layer_idx) const;

    // Get all tensor mappings for an architecture
    std::unordered_map<std::string, std::string> get_all_tensor_mappings(
        const std::string& architecture, int num_layers) const;

    // Check if architecture exists
    bool has_architecture(const std::string& name) const;

private:
    bool load_architecture_file(const std::string& file_path);
    bool validate_definition(const ArchitectureDefinition& def);

    std::unordered_map<std::string, ArchitectureDefinition> architectures_;
};

} // namespace barskuy::registry