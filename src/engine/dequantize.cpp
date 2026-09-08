#include "dequantize.hpp"
#include "core/logger.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>

// C++17 compatible ends_with
static bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// FP16 conversion helpers (using uint16_t raw bits)
static inline float fp16_to_fp32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1 : 1) * std::ldexp(mant / 1024.0f, -14);
    } else if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    } else {
        return (sign ? -1 : 1) * std::ldexp(1.0f + mant / 1024.0f, exp - 15);
    }
}

static inline uint16_t fp32_to_fp16(float f) {
    if (std::isnan(f)) return 0x7E00;
    if (std::isinf(f)) return (f < 0) ? 0xFC00 : 0x7C00;
    if (f == 0.0f) return 0;
    
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    int sign = (bits >> 31) & 1;
    int exp = ((bits >> 23) & 0xFF) - 127;
    int mant = bits & 0x7FFFFF;
    
    if (exp > 15) return (sign << 15) | 0x7C00;
    if (exp < -14) {
        if (exp < -24) return sign << 15;
        mant |= 0x800000;
        mant >>= (-exp - 14);
        return (sign << 15) | (mant >> 13);
    }
    
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
}

namespace barskuy::engine {

QuantParams detect_quant_format(const std::unordered_map<std::string, std::string>& tensor_name_map,
                                const std::string& base_name) {
    QuantParams params;
    
    std::string qweight_name = base_name + ".qweight";
    std::string scales_name = base_name + ".scales";
    std::string qzeros_name = base_name + ".qzeros";
    
    auto find_mapped = [&](const std::string& canonical) -> std::string {
        for (const auto& [canon, st] : tensor_name_map) {
            if (canon == canonical) return st;
        }
        return canonical;
    };
    
    std::string mapped_qweight = find_mapped(qweight_name);
    std::string mapped_scales = find_mapped(scales_name);
    std::string mapped_qzeros = find_mapped(qzeros_name);
    
    if (base_name.find("qweight") != std::string::npos ||
        base_name.find("qzeros") != std::string::npos) {
        params.format = QuantFormat::None;
        return params;
    }
    
    return params;
}

// AWQ dequantization: int4 packed weights + FP16 scales + int4 zero points
bool dequantize_awq(const void* weight_data, const void* scales, const void* zeros,
                    void* output, int rows, int cols, int group_size, bool has_zero_point) {
    const uint32_t* qweight = static_cast<const uint32_t*>(weight_data);
    const uint16_t* scale_data = static_cast<const uint16_t*>(scales);
    const uint32_t* qzeros = has_zero_point ? static_cast<const uint32_t*>(zeros) : nullptr;
    uint16_t* out = static_cast<uint16_t*>(output);
    
    int num_groups = (cols + group_size - 1) / group_size;
    
    std::vector<int8_t> weight_unpacked(rows * cols);
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int pack_idx = (c / 8) * rows + r;
            uint32_t packed = qweight[pack_idx];
            int shift = (c % 8) * 4;
            int8_t w = static_cast<int8_t>((packed >> shift) & 0xF);
            weight_unpacked[r * cols + c] = w;
        }
    }
    
    std::vector<int8_t> zero_unpacked;
    if (has_zero_point && qzeros) {
        zero_unpacked.resize(rows * num_groups);
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < num_groups; ++g) {
                int pack_idx = (g / 8) * rows + r;
                uint32_t packed = qzeros[pack_idx];
                int shift = (g % 8) * 4;
                int8_t z = static_cast<int8_t>((packed >> shift) & 0xF);
                zero_unpacked[r * num_groups + g] = z;
            }
        }
    }
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int g = c / group_size;
            float scale = fp16_to_fp32(scale_data[r * num_groups + g]);
            int8_t w = weight_unpacked[r * cols + c];
            float val;
            if (has_zero_point) {
                int8_t z = zero_unpacked[r * num_groups + g];
                val = (w - z) * scale;
            } else {
                val = w * scale;
            }
            out[r * cols + c] = fp32_to_fp16(val);
        }
    }
    
    return true;
}

// GPTQ dequantization
bool dequantize_gptq(const void* weight_data, const void* scales, const void* zeros,
                     const void* g_idx, void* output, int rows, int cols,
                     int group_size, int bits, bool has_zero_point) {
    int pack_factor = 32 / bits;
    const uint32_t* qweight = static_cast<const uint32_t*>(weight_data);
    const uint16_t* scale_data = static_cast<const uint16_t*>(scales);
    const uint32_t* qzeros = has_zero_point ? static_cast<const uint32_t*>(zeros) : nullptr;
    const int32_t* g_idx_data = static_cast<const int32_t*>(g_idx);
    uint16_t* out = static_cast<uint16_t*>(output);
    
    int num_groups = (cols + group_size - 1) / group_size;
    
    std::vector<int8_t> weight_unpacked(rows * cols);
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int pack_idx = (c / pack_factor) * rows + r;
            uint32_t packed = qweight[pack_idx];
            int shift = (c % pack_factor) * bits;
            int mask = (1 << bits) - 1;
            int8_t w = static_cast<int8_t>((packed >> shift) & mask);
            if (bits == 4 && (w & 0x8)) w |= 0xF0;
            if (bits == 8 && (w & 0x80)) w |= 0xFF00;
            weight_unpacked[r * cols + c] = w;
        }
    }
    
    std::vector<int8_t> zero_unpacked;
    if (has_zero_point && qzeros) {
        zero_unpacked.resize(rows * num_groups);
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < num_groups; ++g) {
                int pack_idx = (g / pack_factor) * rows + r;
                uint32_t packed = qzeros[pack_idx];
                int shift = (g % pack_factor) * bits;
                int mask = (1 << bits) - 1;
                int8_t z = static_cast<int8_t>((packed >> shift) & mask);
                zero_unpacked[r * num_groups + g] = z;
            }
        }
    }
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int g = g_idx_data ? g_idx_data[c] : (c / group_size);
            if (g < 0) g = 0;
            if (g >= num_groups) g = num_groups - 1;
            
            float scale = fp16_to_fp32(scale_data[r * num_groups + g]);
            int8_t w = weight_unpacked[r * cols + c];
            float val;
            if (has_zero_point) {
                int8_t z = zero_unpacked[r * num_groups + g];
                val = (w - z) * scale;
            } else {
                val = w * scale;
            }
            out[r * cols + c] = fp32_to_fp16(val);
        }
    }
    
    return true;
}

// FP8 dequantization (E4M3 or E5M2)
bool dequantize_fp8(const void* weight_data, const void* scales, void* output,
                    int rows, int cols, bool is_e4m3, bool per_channel) {
    const uint8_t* fp8_data = static_cast<const uint8_t*>(weight_data);
    const uint16_t* scale_data = static_cast<const uint16_t*>(scales);
    uint16_t* out = static_cast<uint16_t*>(output);
    
    static uint16_t fp8_e4m3_to_f16[256];
    static uint16_t fp8_e5m2_to_f16[256];
    static bool tables_init = false;
    
    if (!tables_init) {
        for (int i = 0; i < 256; ++i) {
            uint8_t v = static_cast<uint8_t>(i);
            int sign = (v >> 7) & 1;
            int exp = (v >> 3) & 0xF;
            int mant = v & 0x7;
            
            float f16_val;
            if (exp == 0) {
                f16_val = (sign ? -1 : 1) * mant * std::pow(2.0f, -9);
            } else if (exp == 15) {
                f16_val = sign ? -INFINITY : INFINITY;
            } else {
                f16_val = (sign ? -1 : 1) * (1.0f + mant / 8.0f) * std::pow(2.0f, exp - 8);
            }
            fp8_e4m3_to_f16[i] = fp32_to_fp16(f16_val);
            
            exp = (v >> 2) & 0x1F;
            mant = v & 0x3;
            
            if (exp == 0) {
                f16_val = (sign ? -1 : 1) * mant * std::pow(2.0f, -12);
            } else if (exp == 31) {
                f16_val = sign ? -INFINITY : INFINITY;
            } else {
                f16_val = (sign ? -1 : 1) * (1.0f + mant / 4.0f) * std::pow(2.0f, exp - 16);
            }
            fp8_e5m2_to_f16[i] = fp32_to_fp16(f16_val);
        }
        tables_init = true;
    }
    
    const uint16_t* table = is_e4m3 ? fp8_e4m3_to_f16 : fp8_e5m2_to_f16;
    
    if (per_channel) {
        for (int r = 0; r < rows; ++r) {
            float scale = fp16_to_fp32(scale_data[r]);
            for (int c = 0; c < cols; ++c) {
                float fp8_val = fp16_to_fp32(table[fp8_data[r * cols + c]]);
                float val = fp8_val * scale;
                out[r * cols + c] = fp32_to_fp16(val);
            }
        }
    } else {
        float scale = fp16_to_fp32(scale_data[0]);
        for (int i = 0; i < rows * cols; ++i) {
            float fp8_val = fp16_to_fp32(table[fp8_data[i]]);
            float val = fp8_val * scale;
            out[i] = fp32_to_fp16(val);
        }
    }
    
    return true;
}

// NF4 dequantization (BitsAndBytes 4-bit NormalFloat)
bool dequantize_nf4(const void* weight_data, const void* absmax, const void* quantile,
                    void* output, int rows, int cols, int block_size) {
    const uint8_t* nf4_data = static_cast<const uint8_t*>(weight_data);
    const uint16_t* absmax_data = static_cast<const uint16_t*>(absmax);
    const uint16_t* quantile_data = static_cast<const uint16_t*>(quantile);
    uint16_t* out = static_cast<uint16_t*>(output);
    
    static const float nf4_levels[16] = {
        -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
        -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
        0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
        0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
    };
    
    int num_blocks = (rows * cols + block_size - 1) / block_size;
    
    for (int b = 0; b < num_blocks; ++b) {
        int block_start = b * block_size;
        int block_end = std::min(block_start + block_size, rows * cols);
        float block_scale = fp16_to_fp32(absmax_data[b]);
        
        for (int i = block_start; i < block_end; ++i) {
            uint8_t byte = nf4_data[i / 2];
            int idx = (i % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
            float val = nf4_levels[idx] * block_scale;
            out[i] = fp32_to_fp16(val);
        }
    }
    
    return true;
}

} // namespace barskuy::engine