#pragma once

// Procedural PREVIEW media (Fase 8 honest stub).
// Full diffusion (UNet/VAE/temporal) is still pending, so generate() fills
// deterministic placeholder pixels: seeded gradient + moving shapes.
// Labeled "preview" in API responses until the real pipeline lands.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace barskuy::engine::preview {

inline uint64_t hash_str(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

// Deterministic RGBA frame. frame_idx/n_frames animate shapes (video).
inline std::vector<uint8_t> render_frame(uint64_t seed, int w, int h, int frame_idx = 0, int n_frames = 1) {
    std::vector<uint8_t> px((size_t)w * h * 4);
    uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    auto rnd = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 33) & 0xFF; };
    uint8_t r0 = rnd(), g0 = rnd(), b0 = rnd();
    uint8_t r1 = rnd(), g1 = rnd(), b1 = rnd();
    double t = n_frames > 1 ? (double)frame_idx / n_frames : 0.0;
    double cx = w * (0.2 + 0.6 * t), cy = h * (0.3 + 0.4 * (1.0 - t));
    double rad = std::min(w, h) * 0.18;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double f = (double)y / (h > 1 ? h - 1 : 1);
            double dx = x - cx, dy = y - cy;
            bool in_circle = (dx * dx + dy * dy) < rad * rad;
            bool bar = ((x + (int)(t * w)) % (w / 8 + 1)) < 3;
            uint8_t r = (uint8_t)(r0 + (r1 - r0) * f);
            uint8_t g = (uint8_t)(g0 + (g1 - g0) * f);
            uint8_t b = (uint8_t)(b0 + (b1 - b0) * f);
            if (in_circle) { r = 255 - r; g = 255 - g; }
            if (bar) b = (uint8_t)std::min(255, b + 60);
            size_t i = ((size_t)y * w + x) * 4;
            px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = 255;
        }
    }
    return px;
}

// PNG writer via stb (for real diffusion output, user wants PNG)
// BMP kept for fallback. Browsers display both fine.
#include <string>
#include <vector>
bool write_png(const std::string& path, const std::vector<uint8_t>& rgba, int w, int h);
// Minimal BMP writer (no deps). Browsers display .bmp fine.
inline bool write_bmp(const std::string& path, const std::vector<uint8_t>& rgba, int w, int h) {
    if ((int)rgba.size() < w * h * 4 || w <= 0 || h <= 0) return false;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    int row_pad = (4 - (w * 3) % 4) % 4;
    uint32_t row_size = (uint32_t)(w * 3 + row_pad);
    uint32_t img_size = row_size * (uint32_t)h;
    uint32_t file_size = 54 + img_size;
    uint8_t hdr[54] = {0};
    hdr[0] = 'B'; hdr[1] = 'M';
    hdr[2] = file_size & 0xFF; hdr[3] = (file_size >> 8) & 0xFF;
    hdr[4] = (file_size >> 16) & 0xFF; hdr[5] = (file_size >> 24) & 0xFF;
    hdr[10] = 54;
    hdr[14] = 40;
    hdr[18] = w & 0xFF; hdr[19] = (w >> 8) & 0xFF; hdr[20] = (w >> 16) & 0xFF; hdr[21] = (w >> 24) & 0xFF;
    hdr[22] = h & 0xFF; hdr[23] = (h >> 8) & 0xFF; hdr[24] = (h >> 16) & 0xFF; hdr[25] = (h >> 24) & 0xFF;
    hdr[26] = 1; hdr[28] = 24;
    hdr[34] = img_size & 0xFF; hdr[35] = (img_size >> 8) & 0xFF;
    hdr[36] = (img_size >> 16) & 0xFF; hdr[37] = (img_size >> 24) & 0xFF;
    fwrite(hdr, 1, 54, f);
    uint8_t pad[3] = {0, 0, 0};
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            size_t i = ((size_t)y * w + x) * 4;
            uint8_t bgr[3] = {rgba[i + 2], rgba[i + 1], rgba[i]};
            fwrite(bgr, 1, 3, f);
        }
        if (row_pad) fwrite(pad, 1, row_pad, f);
    }
    fclose(f);
    return true;
}

} // namespace barskuy::engine::preview
