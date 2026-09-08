#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <optional>
#include <ggml.h>
#include "dequantize.hpp"

namespace barskuy::engine {

struct SafetensorsMetadata {
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        std::string dtype;  // e.g., "F32", "F16", "BF16", "I8", etc.
        uint64_t offset_begin = 0;
        uint64_t offset_end = 0;
    };

    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_map<std::string, std::string> metadata;  // __metadata__ JSON
    size_t header_size = 0;
    size_t data_size = 0;
};

enum class SafetensorsDType {
    Unknown,
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    BOOL
};

class SafetensorsLoader {
public:
    SafetensorsLoader() = default;
    ~SafetensorsLoader() = default;

    // Parse header only (no tensor data loading)
    bool parse_header(const std::string& file_path, SafetensorsMetadata& metadata);

    // Load tensor data into ggml context
    bool load_tensors(const std::string& file_path, 
                      struct ggml_context* ctx,
                      std::unordered_map<std::string, struct ggml_tensor*>& tensors,
                      const SafetensorsMetadata& metadata,
                      const std::unordered_map<std::string, std::string>& tensor_name_map = {});

    // Load and dequantize tensor data (for AWQ/GPTQ/FP8/NF4)
    bool load_tensors_dequantized(const std::string& file_path,
                                  struct ggml_context* ctx,
                                  std::unordered_map<std::string, struct ggml_tensor*>& tensors,
                                  const SafetensorsMetadata& metadata,
                                  const std::unordered_map<std::string, std::string>& tensor_name_map);

    // Get ggml_type from dtype string
    static enum ggml_type dtype_to_ggml_type(const std::string& dtype);

    // Get dtype string from ggml_type
    static std::string ggml_type_to_dtype(enum ggml_type type);

private:
    bool read_file_header(const std::string& file_path, std::vector<char>& header_data, uint64_t& header_size);
    bool parse_json_header(const std::vector<char>& header_data, SafetensorsMetadata& metadata);
    size_t calculate_data_size(const SafetensorsMetadata& metadata);
};

} // namespace barskuy::engine