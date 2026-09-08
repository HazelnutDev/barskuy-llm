#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <ggml.h>

namespace barskuy::engine {

// Quantization format detection from tensor names
enum class QuantFormat {
    None,       // No quantization (FP16/BF16/F32)
    AWQ,        // Activation-aware Weight Quantization
    GPTQ,       // GPTQ quantization
    FP8,        // FP8 (E4M3/E5M2)
    NF4,        // BitsAndBytes 4-bit NormalFloat
    Unknown
};

// Quantization parameters extracted from safetensors
struct QuantParams {
    QuantFormat format = QuantFormat::None;
    int group_size = 128;           // For AWQ/GPTQ
    bool has_zero_point = true;     // For AWQ/GPTQ
    bool is_symmetric = false;      // For FP8
    std::string scale_tensor;       // Name of scale tensor
    std::string zero_tensor;        // Name of zero-point tensor
    std::string weight_tensor;      // Name of packed weight tensor
};

// Detect quantization format from tensor names in the model
QuantParams detect_quant_format(const std::unordered_map<std::string, std::string>& tensor_name_map,
                                const std::string& base_name);

// Dequantize AWQ packed int4 weights to F16
// weight_data: packed int4 weights (uint32_t array, 2 weights per uint32_t)
// scales: FP16 scales [num_groups]
// zeros: packed int4 zero points (uint32_t array) or nullptr for symmetric
// output: F16 output buffer [rows * cols]
// rows: output rows (out_features)
// cols: output cols (in_features)
bool dequantize_awq(const void* weight_data, const void* scales, const void* zeros,
                    void* output, int rows, int cols, int group_size, bool has_zero_point);

// Dequantize GPTQ packed weights to F16
bool dequantize_gptq(const void* weight_data, const void* scales, const void* zeros,
                     const void* g_idx, void* output, int rows, int cols, 
                     int group_size, int bits, bool has_zero_point);

// Dequantize FP8 to F16
// weight_data: FP8 weights (uint8_t array)
// scales: FP16 scales (per-tensor or per-channel)
// output: F16 output buffer
bool dequantize_fp8(const void* weight_data, const void* scales, void* output,
                    int rows, int cols, bool is_e4m3, bool per_channel);

// Dequantize NF4 (BitsAndBytes 4-bit NormalFloat) to F16
// weight_data: NF4 packed weights (uint8_t array, 2 weights per byte)
// absmax: FP16 absolute max values per block
// quantile: NF4 quantile constants (16 values)
// output: F16 output buffer
bool dequantize_nf4(const void* weight_data, const void* absmax, const void* quantile,
                    void* output, int rows, int cols, int block_size);

// Helper: unpack int4 from uint32_t (2 weights per uint32_t)
inline void unpack_int4(const uint32_t* packed, int8_t* unpacked, int count) {
    for (int i = 0; i < count; i += 2) {
        uint32_t val = packed[i / 2];
        unpacked[i] = static_cast<int8_t>(val & 0xF);
        if (i + 1 < count) {
            unpacked[i + 1] = static_cast<int8_t>((val >> 4) & 0xF);
        }
    }
}

// Helper: unpack int4 zero points from uint32_t
inline void unpack_int4_zeros(const uint32_t* packed, int8_t* unpacked, int count) {
    unpack_int4(packed, unpacked, count);
}

} // namespace barskuy::engine