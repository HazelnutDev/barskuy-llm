#include "safetensors_loader.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <cstring>
#include <algorithm>

namespace barskuy::engine {

// C++17 compatible ends_with
static bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Helper to read tensor data from file at specific offset
static bool read_tensor_data(std::ifstream& file, const SafetensorsMetadata::TensorInfo& info, void* buffer) {
    file.seekg(info.offset_begin, std::ios::beg);
    if (!file) return false;
    size_t size = info.offset_end - info.offset_begin;
    file.read(static_cast<char*>(buffer), size);
    return file.good();
}

bool SafetensorsLoader::parse_header(const std::string& file_path, SafetensorsMetadata& metadata) {
    core::Logger::info("Parsing safetensors header: {}", file_path);

    std::vector<char> header_data;
    uint64_t header_size;
    if (!read_file_header(file_path, header_data, header_size)) {
        return false;
    }
    metadata.header_size = header_size;

    if (!parse_json_header(header_data, metadata)) {
        return false;
    }

    metadata.data_size = calculate_data_size(metadata);
    core::Logger::info("Parsed {} tensors, header size: {}, data size: {}",
        metadata.tensors.size(), metadata.header_size, metadata.data_size);

    return true;
}

bool SafetensorsLoader::read_file_header(const std::string& file_path, std::vector<char>& header_data, uint64_t& header_size_out) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        core::Logger::error("Failed to open file: {}", file_path);
        return false;
    }

    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file) {
        core::Logger::error("Failed to read header size");
        return false;
    }

    header_data.resize(header_size);
    file.read(header_data.data(), header_size);
    if (!file) {
        core::Logger::error("Failed to read header data");
        return false;
    }

    header_size_out = header_size + 8;
    return true;
}

bool SafetensorsLoader::parse_json_header(const std::vector<char>& header_data, SafetensorsMetadata& metadata) {
    try {
        std::string json_str(header_data.begin(), header_data.end());
        auto json = nlohmann::json::parse(json_str);

        for (auto& [name, info] : json.items()) {
            if (name == "__metadata__") {
                metadata.metadata = info.get<std::unordered_map<std::string, std::string>>();
                continue;
            }

            SafetensorsMetadata::TensorInfo tensor_info;
            tensor_info.name = name;

            if (info.contains("shape") && info["shape"].is_array()) {
                for (auto& dim : info["shape"]) {
                    tensor_info.shape.push_back(dim.get<int64_t>());
                }
            }

            if (info.contains("dtype") && info["dtype"].is_string()) {
                tensor_info.dtype = info["dtype"].get<std::string>();
            }

            if (info.contains("data_offsets") && info["data_offsets"].is_array() && info["data_offsets"].size() == 2) {
                tensor_info.offset_begin = info["data_offsets"][0].get<uint64_t>();
                tensor_info.offset_end = info["data_offsets"][1].get<uint64_t>();
            }

            metadata.tensors[name] = std::move(tensor_info);
        }

        return true;
    } catch (const std::exception& e) {
        core::Logger::error("Failed to parse JSON header: {}", e.what());
        return false;
    }
}

size_t SafetensorsLoader::calculate_data_size(const SafetensorsMetadata& metadata) {
    size_t max_offset = 0;
    for (const auto& [name, info] : metadata.tensors) {
        if (info.offset_end > max_offset) {
            max_offset = info.offset_end;
        }
    }
    return max_offset;
}

bool SafetensorsLoader::load_tensors(const std::string& file_path,
                                      struct ggml_context* ctx,
                                      std::unordered_map<std::string, struct ggml_tensor*>& tensors,
                                      const SafetensorsMetadata& metadata,
                                      const std::unordered_map<std::string, std::string>& tensor_name_map) {
    core::Logger::info("Loading tensor data from: {}", file_path);

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        core::Logger::error("Failed to open file for tensor data: {}", file_path);
        return false;
    }

    file.seekg(metadata.header_size, std::ios::beg);
    if (!file) {
        core::Logger::error("Failed to seek to tensor data");
        return false;
    }

    std::unordered_map<std::string, struct ggml_tensor*> st_tensor_lookup;
    for (const auto& [canonical_name, tensor] : tensors) {
        std::string st_name = canonical_name;
        auto it = tensor_name_map.find(canonical_name);
        if (it != tensor_name_map.end()) {
            st_name = it->second;
        }
        st_tensor_lookup[st_name] = tensor;
    }

    for (const auto& [name, info] : metadata.tensors) {
        auto it = st_tensor_lookup.find(name);
        if (it == st_tensor_lookup.end()) {
            size_t tensor_size = info.offset_end - info.offset_begin;
            file.seekg(tensor_size, std::ios::cur);
            continue;
        }

        struct ggml_tensor* tensor = it->second;
        size_t tensor_size = info.offset_end - info.offset_begin;
        void* data = ggml_get_data(tensor);

        file.read(static_cast<char*>(data), tensor_size);
        if (!file) {
            core::Logger::error("Failed to read tensor: {} (canonical: {})", name, 
                tensor_name_map.empty() ? name : "mapped");
            return false;
        }

        core::Logger::debug("Loaded tensor: {} ({} bytes)", name, tensor_size);
    }

    core::Logger::info("All tensor data loaded successfully");
    return true;
}

bool SafetensorsLoader::load_tensors_dequantized(const std::string& file_path,
                                                  struct ggml_context* ctx,
                                                  std::unordered_map<std::string, struct ggml_tensor*>& tensors,
                                                  const SafetensorsMetadata& metadata,
                                                  const std::unordered_map<std::string, std::string>& tensor_name_map) {
    core::Logger::info("Loading and dequantizing tensor data from: {}", file_path);

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        core::Logger::error("Failed to open file for tensor data: {}", file_path);
        return false;
    }

    std::unordered_map<std::string, struct ggml_tensor*> st_tensor_lookup;
    for (const auto& [canonical_name, tensor] : tensors) {
        std::string st_name = canonical_name;
        auto it = tensor_name_map.find(canonical_name);
        if (it != tensor_name_map.end()) {
            st_name = it->second;
        }
        st_tensor_lookup[st_name] = tensor;
    }

    struct QuantComponent {
        std::string base_name;
        std::string weight_name;
        std::string scales_name;
        std::string zeros_name;
        std::string g_idx_name;
        QuantFormat format = QuantFormat::None;
    };

    std::unordered_map<std::string, QuantComponent> quant_components;

    for (const auto& [name, info] : metadata.tensors) {
        if (ends_with(name, ".qweight")) {
            std::string base = name.substr(0, name.size() - 8);
            quant_components[base].base_name = base;
            quant_components[base].weight_name = name;
            quant_components[base].format = QuantFormat::AWQ;
        } else if (ends_with(name, ".scales")) {
            std::string base = name.substr(0, name.size() - 7);
            quant_components[base].scales_name = name;
        } else if (ends_with(name, ".qzeros")) {
            std::string base = name.substr(0, name.size() - 7);
            quant_components[base].zeros_name = name;
        } else if (ends_with(name, ".g_idx")) {
            std::string base = name.substr(0, name.size() - 6);
            quant_components[base].g_idx_name = name;
            quant_components[base].format = QuantFormat::GPTQ;
        }
    }

    for (const auto& [name, info] : metadata.tensors) {
        if (ends_with(name, ".qweight") || ends_with(name, ".scales") || 
            ends_with(name, ".qzeros") || ends_with(name, ".g_idx")) {
            size_t tensor_size = info.offset_end - info.offset_begin;
            file.seekg(tensor_size, std::ios::cur);
            continue;
        }

        auto it = st_tensor_lookup.find(name);
        if (it == st_tensor_lookup.end()) {
            size_t tensor_size = info.offset_end - info.offset_begin;
            file.seekg(tensor_size, std::ios::cur);
            continue;
        }

        struct ggml_tensor* tensor = it->second;
        void* data = ggml_get_data(tensor);

        auto qc_it = quant_components.find(name);
        if (qc_it != quant_components.end()) {
            const QuantComponent& qc = qc_it->second;
            
            auto weight_it = metadata.tensors.find(qc.weight_name);
            auto scales_it = metadata.tensors.find(qc.scales_name);
            auto zeros_it = qc.zeros_name.empty() ? metadata.tensors.end() : metadata.tensors.find(qc.zeros_name);
            auto g_idx_it = qc.g_idx_name.empty() ? metadata.tensors.end() : metadata.tensors.find(qc.g_idx_name);

            if (weight_it == metadata.tensors.end() || scales_it == metadata.tensors.end()) {
                core::Logger::error("Missing weight or scales for quantized tensor: {}", name);
                return false;
            }

            size_t weight_size = weight_it->second.offset_end - weight_it->second.offset_begin;
            size_t scales_size = scales_it->second.offset_end - scales_it->second.offset_begin;
            
            std::vector<uint8_t> weight_buf(weight_size);
            std::vector<uint8_t> scales_buf(scales_size);
            std::vector<uint8_t> zeros_buf;
            std::vector<uint8_t> g_idx_buf;

            if (!read_tensor_data(file, weight_it->second, weight_buf.data())) return false;
            if (!read_tensor_data(file, scales_it->second, scales_buf.data())) return false;

            if (zeros_it != metadata.tensors.end()) {
                size_t zeros_size = zeros_it->second.offset_end - zeros_it->second.offset_begin;
                zeros_buf.resize(zeros_size);
                if (!read_tensor_data(file, zeros_it->second, zeros_buf.data())) return false;
            }

            if (g_idx_it != metadata.tensors.end()) {
                size_t g_idx_size = g_idx_it->second.offset_end - g_idx_it->second.offset_begin;
                g_idx_buf.resize(g_idx_size);
                if (!read_tensor_data(file, g_idx_it->second, g_idx_buf.data())) return false;
            }

            int rows = static_cast<int>(info.shape[0]);
            int cols = info.shape.size() > 1 ? static_cast<int>(info.shape[1]) : 1;
            
            bool success = false;
            if (qc.format == QuantFormat::AWQ) {
                success = dequantize_awq(weight_buf.data(), scales_buf.data(), 
                                         zeros_buf.empty() ? nullptr : zeros_buf.data(),
                                         data, rows, cols, 128, !zeros_buf.empty());
            } else if (qc.format == QuantFormat::GPTQ) {
                success = dequantize_gptq(weight_buf.data(), scales_buf.data(),
                                          zeros_buf.empty() ? nullptr : zeros_buf.data(),
                                          g_idx_buf.empty() ? nullptr : g_idx_buf.data(),
                                          data, rows, cols, 128, 4, !zeros_buf.empty());
            }

            if (!success) {
                core::Logger::error("Dequantization failed for: {}", name);
                return false;
            }
            
            core::Logger::debug("Dequantized tensor: {} ({}x{})", name, rows, cols);
            continue;
        }

        size_t tensor_size = info.offset_end - info.offset_begin;
        file.read(static_cast<char*>(data), tensor_size);
        if (!file) {
            core::Logger::error("Failed to read tensor: {}", name);
            return false;
        }
        core::Logger::debug("Loaded tensor: {} ({} bytes)", name, tensor_size);
    }

    core::Logger::info("All tensor data loaded and dequantized successfully");
    return true;
}

enum ggml_type SafetensorsLoader::dtype_to_ggml_type(const std::string& dtype) {
    static const std::unordered_map<std::string, enum ggml_type> dtype_map = {
        {"F32", GGML_TYPE_F32},
        {"F16", GGML_TYPE_F16},
        {"BF16", GGML_TYPE_BF16},
        {"F64", GGML_TYPE_F64},
        {"I8", GGML_TYPE_I8},
        {"I16", GGML_TYPE_I16},
        {"I32", GGML_TYPE_I32},
        {"I64", GGML_TYPE_I64},
        {"U8", GGML_TYPE_I8},
        {"U16", GGML_TYPE_I16},
        {"U32", GGML_TYPE_I32},
        {"U64", GGML_TYPE_I64},
        {"BOOL", GGML_TYPE_I8},
        {"f32", GGML_TYPE_F32},
        {"f16", GGML_TYPE_F16},
        {"bf16", GGML_TYPE_BF16},
        {"f64", GGML_TYPE_F64},
        {"i8", GGML_TYPE_I8},
        {"i16", GGML_TYPE_I16},
        {"i32", GGML_TYPE_I32},
        {"i64", GGML_TYPE_I64},
        {"bool", GGML_TYPE_I8},
    };

    auto it = dtype_map.find(dtype);
    if (it != dtype_map.end()) {
        return it->second;
    }

    core::Logger::warn("Unknown dtype: {}, defaulting to F32", dtype);
    return GGML_TYPE_F32;
}

std::string SafetensorsLoader::ggml_type_to_dtype(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_F64: return "F64";
        case GGML_TYPE_I8: return "I8";
        case GGML_TYPE_I16: return "I16";
        case GGML_TYPE_I32: return "I32";
        case GGML_TYPE_I64: return "I64";
        default: return "F32";
    }
}

} // namespace barskuy::engine