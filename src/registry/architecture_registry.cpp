#include "architecture_registry.hpp"
#include "core/logger.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>

namespace barskuy::registry {

bool ArchitectureRegistry::load_from_directory(const std::string& directory_path) {
    core::Logger::info("Loading architecture definitions from: {}", directory_path);

    std::filesystem::path dir(directory_path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        core::Logger::warn("Architecture directory does not exist: {}", directory_path);
        return true;  // Not an error, just no custom architectures
    }

    int loaded_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".json") {
            if (load_architecture_file(entry.path().string())) {
                loaded_count++;
            }
        }
    }

    core::Logger::info("Loaded {} architecture definitions", loaded_count);
    return true;
}

bool ArchitectureRegistry::load_architecture_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        core::Logger::error("Failed to open architecture file: {}", file_path);
        return false;
    }

    try {
        nlohmann::json json;
        file >> json;

        ArchitectureDefinition def;
        def.name = json.value("name", "");
        if (def.name.empty()) {
            core::Logger::error("Architecture definition missing 'name' field: {}", file_path);
            return false;
        }

        def.id = json.value("id", def.name);
        def.tensor_mapping = json.value("tensor_mapping", nlohmann::json::object());
        def.attention_config = json.value("attention", nlohmann::json::object());
        def.moe_config = json.value("moe", nlohmann::json::object());
        def.graph_template = json.value("graph_template", "standard_decoder_only");

        // Store any extra config fields
        for (auto& [key, value] : json.items()) {
            if (key != "name" && key != "id" && key != "tensor_mapping" && 
                key != "attention" && key != "moe" && key != "graph_template") {
                def.extra_config[key] = value;
            }
        }

        def.created_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (!validate_definition(def)) {
            return false;
        }

        architectures_[def.name] = std::move(def);
        core::Logger::info("Loaded architecture: {}", architectures_[def.name].name);
        return true;

    } catch (const std::exception& e) {
        core::Logger::error("Failed to parse architecture file {}: {}", file_path, e.what());
        return false;
    }
}

bool ArchitectureRegistry::validate_definition(const ArchitectureDefinition& def) {
    if (def.name.empty()) {
        core::Logger::error("Architecture name is empty");
        return false;
    }

    if (def.tensor_mapping.is_null() || def.tensor_mapping.empty()) {
        core::Logger::warn("Architecture {} has no tensor_mapping", def.name);
    }

    // Check for required tensor mappings
    const std::vector<std::string> required_mappings = {
        "token_embd",
        "output"
    };

    for (const auto& required : required_mappings) {
        if (!def.tensor_mapping.contains(required)) {
            core::Logger::warn("Architecture {} missing recommended tensor mapping: {}", def.name, required);
        }
    }

    return true;
}

std::optional<ArchitectureDefinition> ArchitectureRegistry::get_architecture(const std::string& name) const {
    auto it = architectures_.find(name);
    if (it != architectures_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<ArchitectureDefinition> ArchitectureRegistry::list_architectures() const {
    std::vector<ArchitectureDefinition> result;
    result.reserve(architectures_.size());
    for (const auto& [name, def] : architectures_) {
        result.push_back(def);
    }
    return result;
}

std::string ArchitectureRegistry::map_tensor_name(const std::string& architecture,
                                                   const std::string& template_name,
                                                   int layer_idx) const {
    auto it = architectures_.find(architecture);
    if (it == architectures_.end()) {
        return "";
    }

    const auto& tensor_mapping = it->second.tensor_mapping;
    if (!tensor_mapping.contains(template_name)) {
        return "";
    }

    std::string pattern = tensor_mapping[template_name].get<std::string>();
    
    // Replace {n} with layer index
    size_t pos = 0;
    while ((pos = pattern.find("{n}", pos)) != std::string::npos) {
        pattern.replace(pos, 3, std::to_string(layer_idx));
        pos += std::to_string(layer_idx).length();
    }

    // Also support {layer} as alternative
    pos = 0;
    while ((pos = pattern.find("{layer}", pos)) != std::string::npos) {
        pattern.replace(pos, 7, std::to_string(layer_idx));
        pos += std::to_string(layer_idx).length();
    }

    return pattern;
}

std::unordered_map<std::string, std::string> ArchitectureRegistry::get_all_tensor_mappings(
    const std::string& architecture, int num_layers) const {
    
    std::unordered_map<std::string, std::string> result;
    
    auto it = architectures_.find(architecture);
    if (it == architectures_.end()) {
        return result;
    }

    const auto& tensor_mapping = it->second.tensor_mapping;
    
    for (auto& [template_name, pattern_json] : tensor_mapping.items()) {
        std::string pattern = pattern_json.get<std::string>();
        
        if (pattern.find("{n}") != std::string::npos || pattern.find("{layer}") != std::string::npos) {
            // Layer-specific tensor - generate for each layer
            for (int i = 0; i < num_layers; ++i) {
                std::string mapped = pattern;
                size_t pos = 0;
                while ((pos = mapped.find("{n}", pos)) != std::string::npos) {
                    mapped.replace(pos, 3, std::to_string(i));
                    pos += std::to_string(i).length();
                }
                pos = 0;
                while ((pos = mapped.find("{layer}", pos)) != std::string::npos) {
                    mapped.replace(pos, 7, std::to_string(i));
                    pos += std::to_string(i).length();
                }
                
                // Create unique key: template_name + layer index
                std::string key = template_name + "." + std::to_string(i);
                result[key] = mapped;
            }
        } else {
            // Global tensor (not layer-specific)
            result[template_name] = pattern;
        }
    }

    return result;
}

bool ArchitectureRegistry::has_architecture(const std::string& name) const {
    return architectures_.find(name) != architectures_.end();
}

} // namespace barskuy::registry