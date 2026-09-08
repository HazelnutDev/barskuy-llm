#include "preview_media.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace barskuy::engine::preview {

bool write_png(const std::string& path, const std::vector<uint8_t>& rgba, int w, int h) {
    if ((int)rgba.size() < w * h * 4 || w <= 0 || h <= 0) return false;
    return stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4) != 0;
}

} // namespace barskuy::engine::preview
