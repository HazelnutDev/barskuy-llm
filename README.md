# barskuy-llm — Single Binary AI Server (Text + Image + Video)

> **v3.2.1 | Windows Vulkan | GGUF + Safetensors | OpenAI Compatible | Embedded llama.cpp WebUI**

`barskuy-llm` adalah inference server **satu binary C++ (MSVC)** berbasis `ggml` + `llama.cpp` + `stable-diffusion.cpp (sd.cpp)`. Satu exe melayani **chat LLM (GGUF), image (Z-Image-Turbo SDXL-Turbo), video, embeddings, tools/MCP agent, dan WebUI** — tanpa Python, Redis, atau microservices.

![WebUI](https://img.shields.io/badge/WebUI-embedded-green) ![Vulkan](https://img.shields.io/badge/Vulkan-RTX%203050%206GB-blue) ![Version](https://img.shields.io/badge/version-3.2.1-orange)

---

## 1. Fitur Utama

| Area | Status | Detail |
|------|--------|--------|
| **Text** | ✅ 3.2.1 | GGUF Q4_K_M/Q6_K, safetensors AWQ/GPTQ/FP8/NF4, streaming SSE sejati, `<think>` → `reasoning_content`, auto-compact, `max_tokens=-1`, `n_predict` alias |
| **Image** | ✅ 3.2.1 | Z-Image-Turbo `Q2_K` (2.93GB) + `ae.safetensors` + `Qwen3-4B` LLM, 12 steps / `RES_MULTISTEP+SIMPLE shift 3.0` cfg 1.0, 1024² ~166s (VAE CPU 104s) — **tajam natural**, tiling 512 jaga OOM 6.9GB |
| **Video** | 🚧 Next | Pipeline + API ada (preview stub), difusi video asli next fase |
| **Agent/Tools** | ✅ | 8 built-in (`get_datetime`, `read/write/edit`, `glob`, `grep`, `shell`) + MCP (`context7`, `exa`), `POST /tools` + passthrough SSE |
| **WebUI** | ✅ | `llama.cpp/tools/ui` SvelteKit **di-embed ke binary** (serve dari memori), capability-gated (Text/Image/Video), model swap tanpa restart |
| **GPU** | Vulkan ON, CUDA off (nvcc 12.8 vs MSVC 19.44 blocked) | RTX 3050 6GB: 692MB VRAM diffusion, 179 tok/s decode LLM, VAE CPU (tune next) |

---

## 2. Download — Pilih Salah Satu

### Opsi A — Release ZIP (Tanpa Build, Recommended)
1. Buka **Releases** di https://github.com/HazelnutDev/barskuy-llm
2. Download `barskuy-llm-3.2.1-windows-vulkan.zip` (≈ 80MB)
3. Extract → dapat `barskuy-llm.exe` + `stable-diffusion.dll` + `libcrypto-3-x64.dll` + `libssl-3-x64.dll`
4. Lanjut ke **Quick Start**

> ZIP **tidak** berisi `models/` (hemat 7GB). Taruh model kamu di `models/` sebelah exe.

### Opsi B — Build dari Source
```powershell
git clone --recursive https://github.com/HazelnutDev/barskuy-llm.git
cd barskuy-llm
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build --config Release
.\build\bin\Release\barskuy-llm.exe --port 8080
```

---

## 3. Quick Start (ZIP / Build Sama)

```powershell
# 1. taruh model (contoh)
# models/qwen3-4b-q6_k.gguf
# models/z_image_turbo-Q2_K.gguf + models/image/vae/ae.safetensors + models/image/llm/qwen3-4b-*.gguf

# 2. jalankan (auth OPEN bila tanpa key, seperti llama-server)
.\barskuy-llm.exe --port 8080
# atau dengan key
.\barskuy-llm.exe --port 8080 --api-key sk-barskuy

# 3. buka WebUI
start http://127.0.0.1:8080
# cek versi
curl http://127.0.0.1:8080/health
# {"version":"3.2.1","build":"Sep  8 2026 ..."}
```

**Single-instance lock**: port dikunci via `%TEMP%/barskuy-llm-<port>.lock` — instance kedua exit `FATAL`, mencegah double-bind Windows `SO_REUSEADDR`.

---

## 4. API

### 4.1 Health & Props
```bash
GET /health
GET /props            # n_ctx, params default (temp 0.6 top_k 20 top_p 0.9 repeat 1.1)
GET /v1/models        # list + status loaded/unloaded + active on top
```

### 4.2 Chat (OpenAI Compatible)
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{
   "model":"qwen3-4b",
   "messages":[{"role":"user","content":"halo"}],
   "stream":true,
   "temperature":0.6,
   "max_tokens":-1
 }'
# stream SSE: data: {"choices":[{"delta":{"content":"...","reasoning_content":"..."}}]} + data: [DONE]
# non-stream: {"choices":[{"message":{"content":"...","reasoning_content":"..."}}],"usage":{...}}

# Alias: POST /v1/complete , POST /v1/completions
```

**Khusus:**
- `reasoning_content` = isi `<think>` yang di-split live (UI collapsible). Kosong → tidak dikirim (anti header kosong).
- `max_tokens=-1` / `n_predict=-1` = sampai EOS dalam batas `n_ctx` (8192 default). Nilai > `n_ctx` di-clamp.
- `chat_template_kwargs.enable_thinking=false` = jawab langsung tanpa think (cepat).

### 4.3 Embeddings
```bash
POST /v1/embeddings { "model":"...", "input":"teks" } # 896-dim
```

### 4.4 Image — Z-Image-Turbo
```bash
# submit async (UI pakai ini)
curl -X POST http://127.0.0.1:8080/v1/generate/image \
 -H "Content-Type: application/json" \
 -d '{
   "model":"z-image-turbo",
   "prompt":"Seorang wanita muda di padang rumput saat sunset, close up dekat frame, cinematic warm, detail tajam",
   "width":1024,"height":1024
 }'
# {"id":"img_...","status":"queued"}

# poll
curl http://127.0.0.1:8080/v1/generate/image/img_...
# {"status":"completed","result":{"data":[{"url":"/image/img_..._f0.png"}]}}
# file: ./image/img_..._f0.png (PNG 1.7-2.0MB)

# param opsional: steps (default 12 turbo), guidance_scale (default 1.0), seed, negative_prompt
```

**Catatan 3.2.1:**
- 1024² = `12 steps + RES_MULTISTEP/SIMPLE shift 3.0`, ~166s (log `vae compute 6657 MB → 104s CPU` + diffusion 60s). 512² tetap cepat. Tiling 512 aktif tapi pada VAE 16ch ini belum efektif (tetap 1 segment) — next optimize.
- Jangan naik ke 20 steps (terbukti 2 menit+ hang prefill sebelum fix).
- Prompt framing `close up dekat frame` paling ngaruh untuk tajam portrait, bukan post-process.

### 4.5 Video (Preview Stub)
```bash
POST /v1/generate/video + GET /v1/generate/video/{id}  # 4 frame BMP preview (difusi video asli next)
```

### 4.6 Tools / MCP
```bash
GET /tools  # bare array ServerToolInfo (8 built-in + mcp__context7__*)
POST /tools { "tool":"get_datetime", "params":{} } # {"plain_text_response":"..."} atau SSE {chunk}/{done}
POST /tools  # juga terima "arguments" STRING JSON ala OpenAI (auto parse)
# Chat passthrough: kirim "tools":[...] tanpa "agent":true → balikan tool_calls (client/MCP WebUI yang eksekusi)
# Agent loop: kirim "agent":true dengan tools → server eksekusi built-in+MCP + loop sampai jawaban
```

### 4.7 Auth
- **Tanpa key di DB** = `Auth: OPEN` (log startup) → semua endpoint lolos tanpa Bearer (WebUI langsung pakai).
- **Dengan key** (`--api-key` atau `BARSKUY_API_KEY` atau `POST /v1/api-keys`) = `Auth: REQUIRED` → wajib `Authorization: Bearer <key>`.

---

## 5. Model

Letak: `models/` (auto `mkdir`). Daftarkan:

```bash
curl -X POST http://127.0.0.1:8080/v1/models/register \
 -H "Content-Type: application/json" \
 -d '{"path":"models/qwen3-4b-q6_k.gguf","alias":"qwen3-4b","capabilities":["text"]}'
curl -X POST http://127.0.0.1:8080/v1/models/register \
 -d '{"path":"models/z_image_turbo-Q2_K.gguf","alias":"z-image-turbo","capabilities":["image"]}'
```

**Rekomendasi 3.2.1:**
- Text: `qwen3-4b-q6_k` / `qwen2.5-0.5b-q4_k_m` / `ling-3-tiny-q4_k_m` (semua diverifikasi load)
- Image: `z_image_turbo-Q2_K.gguf` (2.93) + `z_vae_ae.safetensors` (160MB) + `qwen3-4b-q4_k_m.gguf` — SDXL-Turbo lama (6.5GB fp16) sudah hapus, tidak muat VRAM 6GB.

---

## 6. Konfigurasi Flags

```
-m/--model <path>          preload satu model (capabilities auto)
--host 0.0.0.0 --port 8080
--api-key <key>            seed key (bersama BARSKUY_API_KEY env)
--tools all | ""            default all (disable: --no-tools)
--agent / --no-agent        default ON
--mcp / --no-mcp + --mcp-servers-config mcp.json  default ON
--tools-root <dir>         sandbox file tools (default .)
--max-agent-steps 5
--n-cpu-moe 1 (default, 38 bikin race prefill)
--batch-size 2048 --ubatch-size 1024 --cache-type-k auto --cache-type-v auto
--context-size 8192 / -ctk (default 8192, fallback props 8192)
--flash-attn, --jinja (selalu ON), --performance
```

---

## 7. Build Detail

- **CMake** `project(barskuy-llm VERSION 3.2.1)` → `BARSKUY_VERSION` ke `/health` + `/props.build_info`
- **Deps**: `third_party/llama.cpp` (ggml 0.22.0 pin), `sd.cpp` (stable-diffusion.dll), `fmt`, `sqlite3`, `httplib`, `nlohmann/json`, `sqlite` DB (`barskuy.db`)
- **Vulkan** `GGML_VULKAN=ON` + `target_link ggml-vulkan` → dua device (Intel iGPU + RTX), offload 99 layers, decode 179 tok/s. CUDA `OFF` (blocked nvcc 12.8 vs MSVC 19.44).

---

## 8. Struktur Repo

```
barskuy-llm/
├── src/{core,api,registry,queue,engine,utils,agent}
├── architectures/   # JSON tensor mapping (Llama/Qwen/Gemma)
├── third_party/     # llama.cpp / sd.cpp / ggml / fmt / sqlite
├── scripts/embed_webui.py  # embed tools/ui dist ke binary
├── CMakeLists.txt
├── task.md          # log bug & fase (sumber kebenaran)
└── build/bin/Release/barskuy-llm.exe  # hasil build (tidak di-push, via Release)
```

Frontend lama `Next.js` sudah hapus — sekarang `llama.cpp/tools/ui` SvelteKit di-embed.

---

## 9. Troubleshooting

- `FATAL port 8080 is held by another instance` → `taskkill /F /IM barskuy-llm.exe`, cek `%TEMP%/barskuy-llm-8080.lock`
- `401 Invalid or missing API key` padahal fresh → `barskuy.db` berisi key lama → hapus `barskuy.db` atau isi key di Settings
- `MODEL_LOAD_FAILED` → path di `models/` salah, cek `/v1/models`
- Image `queued` lama → 1024 memang ~2.5 menit (VAE CPU). Cek `BARSKUY_LOG_FILE` atau `srv.log`. Preview 44KB = fallback gagal; file 1.7MB = real.

---

## 10. Roadmap

- **Next**: Video diffusion asli (Anima/AnimateDiff) ganti preview stub
- **Image**: VAE tiling efektif + Vulkan VAE (hilangkan 104s CPU), hires-fix untuk zoom lebih tajam
- **Text**: Speculative draft same-vocab (Qwen3-0.6B), Q4_K_M benchmark ~55 tok/s
- Lihat `task.md §1 Fase & Milestone` untuk histori lengkap 50+ BUG/Fase.

---

## 11. Lisensi & Kredit

MIT. Kredit: `ggerganov/llama.cpp`, `leejet/stable-diffusion.cpp`, `ggml`. Dibuat untuk RTX 3050 6GB single-machine — jujur di `task.md` bila ada stub/limit.
