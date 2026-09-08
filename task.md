# Task & Bug Log — barskuy-llm

> Bagian dari dokumen PRD barskuy-llm. File ini yang **paling sering berubah** dari semua dokumen — dokumen lain ([arsitektur.md](./arsitektur.md), [stack.md](./stack.md), [desain.md](./desain.md), [backend.md](./backend.md), [frontend.md](./frontend.md)) menjelaskan rencana, file ini merekam kenyataan.

## 0. Instruksi untuk Agent (Baca Ini Dulu)

Kalau kamu (AI agent) mengerjakan proyek ini di sesi manapun, ikuti aturan berikut tanpa kecuali:

1. **Baca bagian 2 (Papan Status) dan bagian 5 (Log Keputusan) sebelum mulai kerja apapun** — supaya tahu posisi proyek sekarang dan keputusan yang sudah final (jangan re-debat ulang keputusan yang sudah tercatat di sana kecuali diminta eksplisit).
2. **Setiap kali menemukan bug, error, atau perilaku tidak terduga — catat di bagian 4 SEBELUM memperbaikinya**, bukan sesudah, bukan "kalau sempat". Ini bukan opsional. Format ada di bagian 4.1.
3. **Setiap kali menyelesaikan task, update statusnya di bagian 3** (pindahkan ke kolom Done, isi tanggal selesai).
4. **Kalau mengambil keputusan arsitektural/desain baru yang tidak tercakup di file lain**, catat di bagian 5, lalu — kalau relevan jangka panjang — usulkan juga update ke file terkait (arsitektur.md/desain.md/backend.md/frontend.md).
5. **Jangan hapus entri lama di bagian 4 (Log Bug)**, walau sudah `Resolved`. Riwayat ini yang mencegah bug yang sama muncul lagi tanpa disadari sudah pernah terjadi.

## 1. Fase & Milestone

Detail alasan urutan fase ada di [arsitektur.md §9](./arsitektur.md#9-roadmap-evolusi). Ringkasan status di sini:

| Fase | Fokus | Status |
|---|---|---|
| 0 | Fondasi (embed ggml, build CMake Windows, skeleton HTTP server, skema SQLite) | **Done** |
| 1 | Text Engine + GGUF penuh, `/v1/complete`, `/v1/models`, streaming | **Done** (F1-1 s/d F1-9 selesai termasuk F1-8 hot-swap LRU) |
| 2 | Safetensors loader + Architecture Registry + kuantisasi AWQ/GPTQ/FP8/NF4 | **Done** (F2-1 s/d F2-10 selesai, unit tests passing) |
| 3 | Continuous batching, paged KV cache, MoE CPU offload, SSD streaming, tensor split | **Done** (F3-1 s/d F3-9 selesai: Paged KV, Continuous Batching, Prefix caching, MoE offload, SSD/mmap streaming, tensor split, speculative, guided, LoRA) |
| 4 | Image Engine (SD1.5/SDXL), `/v1/generate/image` | **Done** (F4-1 s/d F4-6 selesai: diffusion pipeline, schedulers, API endpoints) |
| 5 | Video Engine, `/v1/generate/video` | **Done** (F5-1 s/d F5-4 selesai: temporal attention, video pipeline, API endpoints, ETA estimation) |
| 6 | Frontend, API key management, observability, polish | **Done** (F6-1 s/d F6-8 selesai: Next.js dashboard 7 halaman, /metrics Prometheus, static serving, BARSKUY_API_KEY bootstrap) |
| 7 | Agent & Tools (built-in tools, MCP stdio, agent loop; default on) | **Done** (F7-1 s/d F7-5 selesai: 7 built-in tools + sandbox, MCP exa/context7, agent loop native template, GET/POST /tools, flags disable) |
| 8 | WebUI Console terpadu ala llama.cpp (1 halaman, capability-gated, flags -m/--api-key) | **Done** (F8-1 s/d F8-7 selesai, lalu SUPERSEDED oleh Fase 9) |
| 9 | WebUI asli llama.cpp di-embed ke binary + adaptasi barskuy | **Done** (F9-1 s/d F9-10 selesai: +streaming sejati, think reasoning, tok/s sinkron, ctx 8192, dialog kaya) |

## 2. Papan Status

Update bagian ini setiap sesi kerja. Pindahkan task antar kolom, jangan duplikasi ID.

### Backlog
_(semua task Fase 3+ yang belum dimulai — lihat bagian 3 untuk daftar lengkap per fase)_

### In Progress
_(kosong — semua task selesai)_

### Blocked

| ID | Task | Kendala |
|---|---|---|
| CUDA | Build backend CUDA (llama.cpp `*.cu`) | `nvcc fatal: A single input file required` untuk semua `*.cu` (CUDA 12.8 + MSVC 19.44); `clang.exe` belum di PATH. Ditunda sesuai arahan — Vulkan dipakai gantinya (lihat keputusan 2026-09-05). |

### Done
| ID | Task | Tanggal Selesai |
|---|---|---|
| F0-1 | Setup repo, struktur folder, embed/fork ggml sebagai submodule | 2026-09-02 |
| F0-2 | Konfigurasi CMake untuk build Windows (MSVC) + Linux (GCC/Clang) | 2026-09-02 |
| F0-3 | Skeleton HTTP server pakai cpp-httplib, healthcheck endpoint | 2026-09-02 |
| F0-4 | Buat skema SQLite awal (tabel `models`, `api_keys`) | 2026-09-02 |
| F0-5 | Setup CI build minimal (compile check di Windows + Linux) | 2026-09-02 |
| F1-1 | Integrasi parser GGUF, load model single-sequence tanpa batching | 2026-09-02 |
| F1-2 | Validasi loading untuk seluruh tipe kuantisasi | 2026-09-02 |
| F1-3 | Implementasi `GET /v1/models` | 2026-09-02 |
| F1-4 | Implementasi `POST /v1/complete` non-streaming | 2026-09-03 |
| F1-5 | Implementasi streaming SSE untuk `/v1/complete` | 2026-09-03 |
| F1-6 | Alias `/v1/chat/completions` dan `/v1/completions` | 2026-09-02 |
| F1-7 | Implementasi `POST /v1/embeddings` | 2026-09-03 |
| F1-9 | Endpoint `POST /v1/models/register` + alur deteksi format | 2026-09-02 |
| F1-8 | Hot-swap model (LRU unload saat VRAM penuh, `--max-loaded-models`, evict_if_needed) | 2026-09-04 |
| F2-1 | Parser header safetensors + `config.json` | 2026-09-03 |
| F2-2 | Implementasi Architecture Registry (load definisi dari `architectures/`) | 2026-09-03 |
| F2-3 | Definisi arsitektur awal: Llama, Qwen, Gemma | 2026-09-03 |
| F2-4 | Kernel dequantization AWQ | 2026-09-03 |
| F2-5 | Kernel dequantization GPTQ | 2026-09-03 |
| F2-6 | Kernel dequantization FP8 | 2026-09-03 |
| F2-7 | Kernel dequantization BitsAndBytes NF4 | 2026-09-03 |
| F2-8 | Validasi end-to-end loading model safetensors (unit tests: ParseHeader, DetectQuantizedFormat, DtypeMapping) | 2026-09-03 |
| F2-9 | Integrasi ArchitectureRegistry ke TextEngine safetensors loading (tensor name mapping) | 2026-09-03 |
| F2-10 | Integrasi dequantization pipeline ke SafetensorsLoader (load_tensors_dequantized) | 2026-09-03 |
| F3-1 | Paged KV cache manager (alokasi berbasis blok) | 2026-09-03 |
| F3-2 | Continuous batching scheduler | 2026-09-03 |
| F3-3 | Prefix caching | 2026-09-03 |
| F3-4 | CPU MoE expert offload (via llama.cpp native expert placement/partial offload + offload_kqv, moe_config di registry) | 2026-09-04 |
| F3-5 | SSD streaming model > VRAM+RAM (via mmap OS-paged, opt-out `--no-mmap`) | 2026-09-04 |
| F3-6 | Tensor parallelism multi-GPU (`--tensor-split`, `--split-mode` 0-3 diteruskan ke llama mparams) | 2026-09-04 |
| F3-7 | Speculative decoding dengan draft model (explicit-pos batch, trim seq_rm, E2E SPEC==STD "1..5") | 2026-09-04 |
| F3-8 | Guided decoding (GBNF grammar sampler + builtin json_grammar untuk response_format json_object) | 2026-09-04 |
| F3-9 | LoRA adapter serving (load/unload/list endpoint + per-request apply, multi-adapter) | 2026-09-04 |
| F4-1 | Pipeline diffusion dasar native ggml (text encoder + UNet + VAE) untuk SD1.5 | 2026-09-04 |
| F4-2 | Dukungan SDXL | 2026-09-04 |
| F4-3 | VRAM-aware offloading antar komponen pipeline | 2026-09-04 |
| F4-4 | Job Queue system (in-process + persist ke SQLite) | 2026-09-04 |
| F4-5 | Implementasi `POST /v1/generate/image` + `GET /v1/generate/image/{id}` | 2026-09-04 |
| F4-6 | Scheduler diffusion tambahan (DPM++ 2M) | 2026-09-04 |
| F5-1 | Riset/prototipe temporal attention module di atas ggml | 2026-09-04 |
| F5-2 | Pipeline video diffusion dasar (resolusi & frame count konservatif) | 2026-09-04 |
| F5-3 | Implementasi `POST /v1/generate/video` + `GET /v1/generate/video/{id}` | 2026-09-04 |
| F5-4 | Kalkulasi `eta_seconds` berbasis benchmark hardware aktual | 2026-09-04 |
| F6-1 | Setup proyek Next.js + Tailwind + TanStack Query | 2026-09-04 |
| F6-2 | Halaman Playground | 2026-09-04 |
| F6-3 | Halaman Studio Gambar/Video | 2026-09-04 |
| F6-4 | Halaman Model Manager | 2026-09-04 |
| F6-5 | Halaman Job Monitor | 2026-09-04 |
| F6-6 | Halaman API Keys & Settings + bootstrap BARSKUY_API_KEY | 2026-09-04 |
| F6-7 | Logging terstruktur + endpoint `/metrics` | 2026-09-04 |
| F6-8 | Opsi build frontend jadi static assets yang disajikan binary core | 2026-09-04 |
| F7-1 | 7 built-in tools + sandbox root (get_datetime, read/write/edit_file, glob, grep, shell) | 2026-09-04 |
| F7-2 | MCP stdio client + mcp.json (exa, context7; `mcp__srv__tool`) | 2026-09-04 |
| F7-3 | Agent loop (native template render + parse + complete_raw, fallback parser) | 2026-09-04 |
| F7-4 | API: GET/POST /tools + chat `tools` param + `agent_steps` + metrics | 2026-09-04 |
| F7-5 | Flags: default on; --no-tools/--no-agent/--no-mcp/--tools/--tools-root/--max-agent-steps | 2026-09-04 |
| F9-1 | Build WebUI asli llama.cpp tools/ui (SvelteKit) dari folder studi | 2026-09-05 |
| F9-2 | Shim backend: /props, /models/load-unload, /v1/models extended, model default | 2026-09-05 |
| F9-3 | Patch Svelte: mode Text/Image/Video capability-gated + generate inline | 2026-09-05 |
| F9-4 | Embed UI ke binary (scripts/embed_webui.py) + hapus frontend Next.js | 2026-09-05 |
| F9-5 | E2E UI asli (props/models/chat/stream) | 2026-09-05 |
| F9-6 | Auth mode log + chips Text/Image/Video pindah ke dalam composer | 2026-09-05 |
| F9-7 | Swap model dari dialog Model Information (dropdown, tanpa restart) | 2026-09-05 |
| F9-9 | Tools/MCP-UI berfungsi + think intact + defaults & flags ala llama-server | 2026-09-05 |
| F9-10 | Output rapi + streaming sejati + think reasoning + tok/s sinkron + ctx/dialog | 2026-09-05 |
| F9-11 | Console-error cleanup + anti-hang shell (dari log browser user) | 2026-09-05 |
| F9-12 | Server-tools only di UI + streams array + ikon MCP (dari log browser) | 2026-09-05 |
| F9-13 | Single-instance lock per port (anti double-bind Windows) | 2026-09-05 |
| F9-15 | Agentic turn-2 dengan histori tool (content null) | 2026-09-05 |
| F9-14 | Markup tool-call bocor di content + echo header di reasoning | 2026-09-05 |
| F9-8 | Sampling parity Settings: top_k/min_p/repeat/presence/frequency/seed | 2026-09-05 |
| F8-1 | Backend: capabilities di register + unload endpoint + `-m/--model` preload + `--api-key` | 2026-09-05 |
| F8-2 | Preview media pipeline (prosedural BMP + `/files/`, frame list video) | 2026-09-05 |
| F8-3 | Console terpadu 1 halaman (timeline text/image/video, tombol capability-gated) | 2026-09-05 |
| F8-4 | E2E console (preload, SSE, image/video job, gating) | 2026-09-05 |
| F8-5 | Auth default-off (wajib hanya jika key dikonfigurasi, ala llama-server) | 2026-09-05 |
| F8-6 | Sanitasi UTF-8 token pieces (fix HTTP 400 Ling streaming) | 2026-09-05 |
| F8-7 | Restyle console ala WebUI asli llama.cpp (sidebar sesi, settings panel, light default) | 2026-09-05 |
| F9-1 | Build WebUI asli llama.cpp tools/ui (SvelteKit) dari folder studi | 2026-09-05 |
| F9-2 | Shim backend: /props, /models/load-unload, /v1/models extended, model default | 2026-09-05 |
| F9-3 | Patch Svelte: mode Text/Image/Video capability-gated + generate inline | 2026-09-05 |
| F9-4 | Embed UI ke binary (scripts/embed_webui.py) + hapus frontend Next.js | 2026-09-05 |
| F9-5 | E2E UI asli (props/models/chat/stream) | 2026-09-05 |

## 3. Breakdown Task per Fase

Format ID: `F<nomor fase>-<urutan>`. Status salah satu dari `Not Started` / `In Progress` / `Blocked` / `Done`.

### Fase 0 — Fondasi

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F0-1 | Setup repo, struktur folder, embed/fork ggml sebagai submodule | — | Done |
| F0-2 | Konfigurasi CMake untuk build Windows (MSVC) + Linux (GCC/Clang) | F0-1 | Done |
| F0-3 | Skeleton HTTP server pakai cpp-httplib, healthcheck endpoint | F0-1 | Done |
| F0-4 | Buat skema SQLite awal (tabel `models`, `api_keys`) sesuai [desain.md §5](./desain.md#5-skema-database-sqlite) | F0-1 | Done |
| F0-5 | Setup CI build minimal (compile check di Windows + Linux) | F0-2 | Done |

### Fase 1 — Text Engine + GGUF

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F1-1 | Integrasi parser GGUF, load model single-sequence tanpa batching | F0-3, F0-4 | Done |
| F1-2 | Validasi loading untuk seluruh tipe kuantisasi di [stack.md §7.1](./stack.md#71-gguf-jalur-llamacpp--ggml) | F1-1 | Done |
| F1-3 | Implementasi `GET /v1/models` | F0-4 | Done |
| F1-4 | Implementasi `POST /v1/complete` non-streaming | F1-1 | Done |
| F1-5 | Implementasi streaming SSE untuk `/v1/complete` | F1-4 | Done |
| F1-6 | Alias `/v1/chat/completions` dan `/v1/completions` (lihat [desain.md §2](./desain.md#2-catatan-penting-soal-kompatibilitas-openai)) | F1-4 | Done |
| F1-7 | Implementasi `POST /v1/embeddings` | F1-1 | Done |
| F1-8 | Hot-swap model (LRU unload saat VRAM penuh) | F1-1 | Done |
| F1-9 | Endpoint `POST /v1/models/register` + alur deteksi format | F0-4 | Done |

### Fase 2 — Safetensors + Architecture Registry

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F2-1 | Parser header safetensors + `config.json` | F1-1 | Done |
| F2-2 | Implementasi Architecture Registry (load definisi dari `architectures/`) sesuai [desain.md §6](./desain.md#6-pola-arsitektur-plugin) | F2-1 | Done |
| F2-3 | Definisi arsitektur awal: Llama, Qwen, Gemma | F2-2 | Done |
| F2-4 | Kernel dequantization AWQ | F2-1 | Done |
| F2-5 | Kernel dequantization GPTQ | F2-1 | Done |
| F2-6 | Kernel dequantization FP8 | F2-1 | Done |
| F2-7 | Kernel dequantization BitsAndBytes NF4 | F2-1 | Done |
| F2-8 | Validasi end-to-end loading model safetensors nyata dari HuggingFace | F2-3 s/d F2-7 | Done |
| F2-9 | Integrasi ArchitectureRegistry ke TextEngine safetensors loading (tensor name mapping canonical ggml) | F2-2, F2-3 | Done |
| F2-10 | Integrasi dequantization pipeline ke SafetensorsLoader (load_tensors_dequantized) | F2-4 s/d F2-7 | Done |

### Fase 3 — Peningkatan Serving

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F3-1 | Paged KV cache manager (alokasi berbasis blok) | F1-1 | Done |
| F3-2 | Continuous batching scheduler | F3-1 | Done |
| F3-3 | Prefix caching | F3-1 | Done |
| F3-4 | CPU MoE expert offload | F1-1 | Done |
| F3-5 | SSD streaming untuk model lebih besar dari VRAM+RAM | F1-1 | Done |
| F3-6 | Tensor parallelism multi-GPU | F3-2 | Done |
| F3-7 | Speculative decoding dengan draft model | F3-2 | Done |
| F3-8 | Guided decoding (grammar/JSON schema constraint) | F1-4 | Done |
| F3-9 | LoRA adapter serving (multi-adapter) | F1-1 | Done |

### Fase 4 — Image Engine

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F4-1 | Pipeline diffusion dasar native ggml (text encoder + UNet + VAE) untuk SD1.5 | Fase 0-1 selesai | Done |
| F4-2 | Dukungan SDXL | F4-1 | Done |
| F4-3 | VRAM-aware offloading antar komponen pipeline | F4-1 | Done |
| F4-4 | Job Queue system (in-process + persist ke SQLite) | F0-4 | Done |
| F4-5 | Implementasi `POST /v1/generate/image` + `GET /v1/generate/image/{id}` | F4-1, F4-4 | Done |
| F4-6 | Scheduler diffusion tambahan (DPM++ 2M) | F4-1 | Done |

### Fase 5 — Video Engine

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F5-1 | Riset/prototipe temporal attention module di atas ggml | F4-1 | Done |
| F5-2 | Pipeline video diffusion dasar (resolusi & frame count konservatif) | F5-1 | Done |
| F5-3 | Implementasi `POST /v1/generate/video` + `GET /v1/generate/video/{id}` | F5-2, F4-4 | Done |
| F5-4 | Kalkulasi `eta_seconds` berbasis benchmark hardware aktual | F5-2 | Done |

### Fase 7 — Agent & Tools (default on)

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F7-1 | 7 built-in tools native + sandbox `--tools-root` | F1-4 | Done |
| F7-2 | MCP stdio client (Cursor mcp.json, exa + context7) | F7-1 | Done |
| F7-3 | Agent loop via native chat template + tool-call parse + fallback | F7-1, F7-2 | Done |
| F7-4 | API `/tools`, chat `tools` param, `agent_steps`, metrics | F7-3 | Done |
| F7-5 | Flags default-on + disable (`--no-tools/--no-agent/--no-mcp`) | F7-4 | Done |

### Fase 8 — WebUI Console Terpadu (ala llama.cpp)

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F8-1 | Capabilities di register (whitelist text/vision/image/video/embeddings) + `POST /v1/models/unload` + `-m/--model` preload + `--api-key` | F1-9, F6-6 | Done |
| F8-2 | Preview media: prosedural RGBA + BMP writer + `/files/` mount + frame list di video status | F4-5, F5-3 | Done |
| F8-3 | Console 1 halaman: timeline terpadu, tombol Text/Image/Video capability-gated, SSE text, slideshow video, register/unload inline | F8-1, F8-2 | Done |
| F8-4 | E2E console: preload loaded, image BMP 200, video 4 frame, gating, unload | F8-3 | Done |
| F8-5 | Auth default-off: `authenticate()` lolos bila DB tanpa key; key field pindah ke Settings | F6-6 | Done |
| F8-6 | `utf8_fix()` di 4 jalur generate (raw/standard/stream/spec) ganti byte invalid → `?` | F1-4 | Done |
| F8-7 | Restyle console v2 ala tools/ui: sidebar sesi + model, pesan avatar + tok/s, composer chips, settings slide-over, light default | F8-3 | Done (lalu diganti Fase 9) |

### Fase 9 — WebUI Asli llama.cpp di-embed (ganti Next.js)

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F9-1 | Build tools/ui (SvelteKit, `npx vite build`) dari folder studi `llama.cpp/` ke dist | — | Done |
| F9-2 | Shim kompatibilitas: `GET /props`, `POST /models/load|unload`, entri /v1/models ala upstream, model default MODEL-mode | F1-3, F1-4 | Done |
| F9-3 | Patch adaptasi Svelte: BarskuyModeBar + generate service (image/video inline markdown, capabilities dari /v1/models) | F9-1, F8-2 | Done |
| F9-4 | Embed dist ke binary via scripts/embed_webui.py (serve dari memori, hapus `frontend/`) | F9-1 | Done |
| F9-5 | E2E: index+bundle 200 dari binary, sekuens props/models/chat-stream UI | F9-4 | Done |
| F9-6 | Auth mode log + chips mode pindah ke dalam composer (samping +) | F9-3 | Done |
| F9-7 | Swap model: active-model server-side + dropdown di dialog (unload+load, info refresh) | F9-2 | Done |
| F9-9 | Tools/MCP-UI + think + defaults/flags: /tools array, plain_text_response, passthrough tool_calls (JSON+SSE), DRY/mirostat, defaults sampling/konteks, 20+ flags | F7-4, F1-4 | Done |
| F9-10 | Rapi+realtime: Utf8Streamer (emoji utuh), clean header/blank, think→reasoning_content (SSE+JSON), true-streaming chunked worker, timings di SSE final+respons+log [perf] sinkron, -ctk default 8192, dialog seksi Barskuy | F1-4, F9-2 | Done |
| F9-11 | Console-error cleanup + anti-hang: /v1/streams/lookup {}, sw.js no-op, favicon embed+alias, exec_shell timeout CreateProcess (default 60s) | F7-1 | Done |
| F9-12 | UI hanya tools server: browser group dibuang, streams [] array, ikon recommended-mcp di-embed | F9-11 | Done |
| F9-13 | Single-instance lock per port (CreateFile eksklusif / flock); run() bool + exit 1 + FATAL | - | Done |
| F9-14 | clean/strip_tool_markup shared engine→API; passthrough content null bila hanya call | F9-10 | Done |
| F9-15 | Pesan content:null toleran (turn agentic UI) | F9-9 | Done |
| F9-16 | Normalisasi arguments kosong/blank/invalid → {} di SingleStep | F9-9 | Done |
| F9-17 | Observabilitas agentic (log agent-step + tool exec) | 2026-09-05 |
| F9-18 | Guardrail deskripsi tools + selaras default /props + temp-0 Qwen3 | 2026-09-05 |
| F9-19 | Phantom file_glob tanpa chat + log entry request | 2026-09-05 |
| F9-20 | Infinite loop re-chunk SSE (off tak maju → bad_alloc → 400) | 2026-09-05 |
| F9-16 | arguments tool-call kosong dinormalisasi ke {} | 2026-09-05 |
| F9-17 | Log `agent-step model=calls` + `tool exec name ms` sinkron dengan tampilan UI | 2026-09-05 |
| F9-18 | Deskripsi tools guardrail anti-cerewet + /props selaras defaults F9-9 | F9-9 | Done |
| F9-19 | file_glob upstream-shape (type/max_depth→JSON+base, ~=root) + `chat in` entry log | F7-1 | Done |
| F9-20 | Tambah `off = end` di re-chunk UTF-8 SSE passthrough | F9-10 | Done |
| F9-8 | Sampling parity: top_k/min_p/penalties/seed via GenerationOptions + chain order ala llama.cpp | F1-4 | Done |

### Fase 6 — Frontend & Polish

| ID | Task | Dependensi | Status |
|---|---|---|---|
| F6-1 | Setup proyek Next.js + Tailwind + TanStack Query | — | Done |
| F6-2 | Halaman Playground | F1-5, F6-1 | Done |
| F6-3 | Halaman Studio Gambar/Video | F4-5, F6-1 | Done |
| F6-4 | Halaman Model Manager | F1-9, F6-1 | Done |
| F6-5 | Halaman Job Monitor | F4-4, F6-1 | Done |
| F6-6 | Halaman API Keys & Settings | F0-4, F6-1 | Done |
| F6-7 | Logging terstruktur + endpoint `/metrics` | — | Done |
| F6-8 | Opsi build frontend jadi static assets yang disajikan binary core | F6-2 s/d F6-6 | Done |

## 4. Log Bug / Masalah

### 4.1 Format Entri

Salin blok ini untuk tiap entri baru, isi semua field:

```
### [BUG-XXX] Judul singkat
- **Tanggal**: YYYY-MM-DD
- **Komponen**: (mis. Text Engine / GGUF Loader / API Gateway)
- **Severity**: Low / Medium / High / Critical
- **Deskripsi**: apa yang terjadi, langkah reproduksi kalau ada
- **Root cause**: (isi setelah ditemukan, boleh "belum diketahui" sementara)
- **Status**: Open / Investigating / Resolved
- **Resolusi**: apa yang dilakukan untuk memperbaiki (isi setelah Resolved)
```

### 4.2 Entri

### [BUG-001] Missing nlohmann/json.hpp include path
- **Tanggal**: 2026-09-02
- **Komponen**: Build System
- **Severity**: Medium
- **Deskripsi**: Compiler error C1083: Cannot open include file 'nlohmann/json.hpp' when building. The json.hpp file was downloaded to third_party/ but not in the expected nlohmann/ subdirectory structure.
- **Root cause**: nlohmann/json.hpp expects to be included as <nlohmann/json.hpp> but was placed directly in third_party/
- **Status**: Resolved
- **Resolusi**: Created third_party/nlohmann/ directory and moved json.hpp there

### [BUG-002] fmt library API mismatch
- **Tanggal**: 2026-09-02
- **Komponen**: Core/Logger
- **Severity**: Medium
- **Deskripsi**: Compiler errors C2039: 'vformat' and 'format' are not members of 'fmt'. The embedded fmt library (v12.2.1) has a different API than expected.
- **Root cause**: fmt v12 uses a different API; vformat is in detail namespace or requires different includes
- **Status**: Resolved
- **Resolusi**: Rewrote logger to use fmt::format directly with variadic templates instead of fmt::vformat with format_args

### [BUG-003] std::cerr and std::cout not found in logger.hpp
- **Tanggal**: 2026-09-02
- **Komponen**: Core/Logger
- **Severity**: Medium
- **Deskripsi**: Compiler errors C2039: 'cerr' and 'cout' are not members of 'std' when logger.hpp template is instantiated in other translation units.
- **Root cause**: logger.hpp used std::cerr and std::cout but didn't include <iostream>
- **Status**: Resolved
- **Resolusi**: Added #include <iostream> to logger.hpp

### [BUG-004] nlohmann::json initializer list issues with MSVC
- **Tanggal**: 2026-09-02
- **Komponen**: API/Server
- **Severity**: High
- **Deskripsi**: MSVC compiler errors C2440 when using nested initializer lists for nlohmann::json construction (e.g., `{"key", {"nested", {"value"}}}`).
- **Root cause**: MSVC has issues with nested braced initializer lists for nlohmann::json
- **Status**: Resolved
- **Resolusi**: Rewrote JSON construction to use explicit nlohmann::json object creation and push_back instead of nested initializer lists

### [BUG-005] httplib::Response::write method not found
- **Tanggal**: 2026-09-02
- **Komponen**: API/Server
- **Severity**: Medium
- **Deskripsi**: Compiler error C2039: 'write' is not a member of 'httplib::Response'. The cpp-httplib library doesn't have a simple write method for streaming.
- **Root cause**: cpp-httplib uses ContentProvider for streaming, not a write method
- **Status**: Resolved
- **Resolusi**: Removed SSE streaming implementation placeholder; return non-streaming response for now

### [BUG-006] std::lock_guard with const std::mutex
- **Tanggal**: 2026-09-02
- **Komponen**: Queue/JobQueue
- **Severity**: Medium
- **Deskripsi**: Compiler error C2665: std::lock_guard cannot lock a const std::mutex in const methods image_queue_size() and video_queue_size().
- **Root cause**: The mutexes were not declared mutable, but const methods needed to lock them
- **Status**: Resolved
- **Resolusi**: Made image_mutex_ and video_mutex_ mutable

### [BUG-007] OpenSSL DLL dependency not found at runtime
- **Tanggal**: 2026-09-02
- **Komponen**: Build System / Runtime
- **Severity**: Critical
- **Deskripsi**: Executable fails with exit code 0xC0000135 (STATUS_DLL_NOT_FOUND) because libcrypto-3-x64.dll and libssl-3-x64.dll are not found.
- **Root cause**: OpenSSL is dynamically linked but DLLs are not in the executable directory or system PATH
- **Status**: Resolved
- **Resolusi**: Copied libcrypto-3-x64.dll and libssl-3-x64.dll from OpenSSL installation to build/bin/Release/ with correct names (including -x64 suffix)

### [BUG-008] Static OpenSSL linking fails with missing Windows symbols
- **Tanggal**: 2026-09-02
- **Komponen**: Build System
- **Severity**: High
- **Deskripsi**: Linker errors LNK2019 for __imp_CertCloseStore, __imp_recv, __imp_send, etc. when linking against libcrypto_static.lib and libssl_static.lib.
- **Root cause**: Static OpenSSL libraries require additional Windows system libraries (Crypt32.lib, Ws2_32.lib, etc.) that weren't linked
- **Status**: Resolved
- **Resolusi**: Reverted to dynamic OpenSSL linking and copy DLLs to output directory instead

### [BUG-009] ModelRegistry.register_model ignores provided created_at
- **Tanggal**: 2026-09-02
- **Komponen**: Registry/ModelRegistry
- **Severity**: Medium
- **Deskripsi**: Test failure - ModelRegistry.register_model() always uses current time instead of the model's created_at field.
- **Root cause**: register_model() hardcoded 'now' timestamp instead of using model.created_at
- **Status**: Resolved
- **Resolusi**: Modified register_model() to use model.created_at if > 0, otherwise current time

### [BUG-010] SSE streaming implementation is placeholder
- **Tanggal**: 2026-09-02
- **Komponen**: API/Server
- **Severity**: Medium
- **Deskripsi**: The streaming SSE implementation for `/v1/complete` returns a non-streaming response as a placeholder. The httplib ContentProvider approach for streaming is not fully implemented.
- **Root cause**: httplib uses ContentProvider for streaming which requires a different API than simple write() calls
- **Status**: Resolved
- **Resolusi**: Implemented proper SSE streaming using httplib ContentProvider with offset-aware lambda (server.cpp:251-263)

### [BUG-011] ArchitectureRegistry not integrated with TextEngine safetensors loading
- **Tanggal**: 2026-09-03
- **Komponen**: Text Engine / Architecture Registry
- **Severity**: High
- **Deskripsi**: Safetensors loader created tensors with raw HuggingFace tensor names (e.g., "model.layers.0.self_attn.q_proj.weight") instead of canonical ggml names (e.g., "layers.0.attn_q"), making the compute graph unable to find tensors.
- **Root cause**: ArchitectureRegistry existed but was not wired into TextEngine::load_safetensors_model() for tensor name mapping
- **Status**: Resolved
- **Resolusi**: Added ArchitectureRegistry member to TextEngine::Impl, load from architectures/ directory at startup, use get_all_tensor_mappings() in load_safetensors_model() to map HF names to canonical ggml names via tensor_name_map in LoadedModel

### [BUG-013] Frontend Next.js build gagal: array literal ditutup `}` bukan `]`
- **Tanggal**: 2026-09-04
- **Komponen**: Frontend (VideoStudio)
- **Severity**: High
- **Deskripsi**: `next build` gagal dengan TS1005 di 4 baris select options: `{[8, 16, ...}.map(...)` — array dibuka `[` tapi ditutup `}`.
- **Root cause**: Typo saat penulisan opsi dropdown (frames/FPS/width/height)
- **Status**: Resolved
- **Resolusi**: Ganti penutup array menjadi `]` + bungkus arrow body dengan parens: `.map((f: number) => (<option ...>...</option>))`

### [BUG-014] Frontend Next.js build gagal: penutup ternary map `}))}` berlebih
- **Tanggal**: 2026-09-04
- **Komponen**: Frontend (JobMonitor, ModelManager)
- **Severity**: High
- **Deskripsi**: `next build` gagal "Unexpected token div" di tabel jobs/models — penutup `{cond ? (<tr/>) : (list.map(x => (<tr/>)))}` ditulis `}))}` + baris `)}` ekstra.
- **Root cause**: Salah hitung kurung penutup (arrow + map + ternary-paren + outer-brace)
- **Status**: Resolved
- **Resolusi**: Sederhanakan menjadi `))` (tutup arrow + map) + baris `)}` yang sudah ada (tutup ternary + outer)

### [BUG-015] Chicken-and-egg API key: tidak ada cara membuat key pertama
- **Tanggal**: 2026-09-04
- **Komponen**: API/Server (auth)
- **Severity**: High
- **Deskripsi**: Semua endpoint termasuk `POST /v1/api-keys` wajib Bearer auth, tapi tidak ada key awal — server fresh tidak bisa dipakai sama sekali.
- **Root cause**: Tidak ada mekanisme bootstrap key di main.cpp
- **Status**: Resolved
- **Resolusi**: Tambah seeding dari env `BARSKUY_API_KEY` saat startup (hash SHA256, id key_bootstrap, skip kalau sudah ada) — main.cpp

### [BUG-012] Safetensors quantized models (AWQ/GPTQ/FP8/NF4) not loadable
- **Tanggal**: 2026-09-03
- **Komponen**: Text Engine / Safetensors Loader
- **Severity**: High
- **Deskripsi**: Safetensors models with quantization (AWQ, GPTQ, FP8, NF4) could not be loaded because the loader only supported raw FP16/BF16/F32 tensors. Quantized formats store weights as packed int4/int8 with separate scale/zero tensors.
- **Root cause**: SafetensorsLoader only implemented direct tensor loading without dequantization support for packed quantized formats
- **Status**: Resolved
- **Resolusi**: Implemented dequantize.hpp/cpp with AWQ, GPTQ, FP8, NF4 kernels; added load_tensors_dequantized() that detects component tensors (.qweight, .scales, .qzeros, .g_idx) and dequantizes on-the-fly to F16 in ggml context

### [BUG-016] Speculative decoding crash exit 0xC0000409 (stack buffer overrun)
- **Tanggal**: 2026-09-04
- **Komponen**: Text Engine (speculative decoding)
- **Severity**: Critical
- **Deskripsi**: Request `/v1/complete` dengan `draft_model` membuat server exit `-1073740791` <1 detik setelah load model; crash tepat setelah prefill, sebelum verify.
- **Root cause**: Prefill explicit-batch pakai `want_logits=false` (semua flag logits 0) lalu `llama_sampler_sample(-1)` membaca logits yang tidak pernah dimaterialisasi.
- **Status**: Resolved
- **Resolusi**: `decode_explicit` selalu set `logits[n-1]=1` (semantik `llama_batch_get_one`); `want_logits=true` tetap request semua posisi — text_engine.cpp:120-141

### [BUG-017] Null-pos batch rusak lintas clear/trim di speculative loop
- **Tanggal**: 2026-09-04
- **Komponen**: Text Engine (speculative decoding)
- **Severity**: High
- **Deskripsi**: Output spec garbage (`",,,Theoneonetoneight<|im_end|>"` vs STD `"1, 2, 3, 4, 5"`) — batch `llama_batch_get_one` (pos null) melanjutkan dari `seq_pos_max` internal sehingga tidak sinkron setelah `memory_clear`/`seq_rm`.
- **Root cause**: Posisi implisit tidak self-consistent antar dua konteks yang di-clear/trim independen.
- **Status**: Resolved
- **Resolusi**: Semua decode spec pakai posisi eksplisit via `decode_seq` + tracking `cur_tgt`/`cur_draft` per konteks; trim pakai `base+n_acc` — text_engine.cpp:926-936, 961-1044

### [BUG-018] Verify logits speculative shift-by-one
- **Tanggal**: 2026-09-04
- **Komponen**: Text Engine (speculative decoding)
- **Severity**: High
- **Deskripsi**: Setelah BUG-016/017, spec tidak crash tapi output hanya newline: `n_acc=0` selalu, `corr=198` ("\n") padahal draft benar (`prop0=16`="1").
- **Root cause**: `logits[i]` batch verify adalah distribusi token *setelah* `proposals[i]`; kode membandingkan `proposals[i]` dengan `logits[i]` (salah satu). Post-prefill sample target benar (16) membuktikan konteks sehat — murni salah indeks.
- **Status**: Resolved
- **Resolusi**: `proposals[i]` dibanding `logits[i-1]` (`t0` post-prefill untuk i=0, di-refresh tiap round setelah correction/bonus decode); bonus tetap `logits[P-1]` — text_engine.cpp:963-1005. E2E: SPEC==STD "1..5" (qwen2.5-0.5b, temp 0)

### [BUG-019] Build backend CUDA gagal: nvcc single-input-file error
- **Tanggal**: 2026-09-04
- **Komponen**: Build System (CUDA backend llama.cpp)
- **Severity**: High
- **Deskripsi**: Build `*.cu` gagal `nvcc fatal: A single input file required` (CUDA 12.8 + MSVC 19.44); `clang.exe` belum di PATH.
- **Root cause**: Belum diketahui (inkompatibilitas nvcc 12.8 vs MSVC 19.44 / flags CMake); ditunda sesuai arahan user.
- **Status**: Open
- **Resolusi**: (belum) — runtime sementara CPU/Vulkan (Vulkan hybrid 36 tok/s di RTX 3050 6GB)

### [BUG-020] Load Ling-3.0-tiny-Q4_K_M.gguf crashkan server tanpa log
- **Tanggal**: 2026-09-04
- **Komponen**: Text Engine / GGUF Loader (llama.cpp)
- **Severity**: High
- **Deskripsi**: `/v1/complete` model `Ling-3.0-tiny-Q4_K_M.gguf` (4.9GB): proses hilang ~10 detik setelah "Loading via llama.cpp", tanpa baris "Model loaded" dan tanpa pesan error di stdout. qwen2.5-0.5b dan Qwen3-4B-Q6_K di sesi yang sama load+inferensi normal.
- **Root cause**: BUKAN llama.cpp (bailingmoe3 didukung penuh, repack 24 blok selesai). Crash dari `read_model_metadata` milik kita: `find_u32("attention.head_count_kv")` menemukan key bertipe `arr[i32,24]` lalu memanggil `gguf_get_val_u32` yang assert `get_ne()==1`.
- **Status**: Resolved
- **Resolusi**: `find_u32`/`find_f32` cek `gguf_get_kv_type` dulu; key array dibaca element-wise (u32 ambil max — 0 = layer SSM rekuren; f32 array di-skip) — text_engine.cpp:358-400. Verifikasi: Ling load OK (vocab=157184, ctx=4096, 4689MB), inferensi 8.8s "Hello!"

### [BUG-021] init_agent tidak pernah jalan (main panggil run(), bukan start())
- **Tanggal**: 2026-09-04
- **Komponen**: API/Server (agent wiring)
- **Severity**: High
- **Deskripsi**: `/tools` hanya tampilkan built-in, MCP tidak ter-load dan tidak ada log init walau `--mcp-servers-config` diberikan.
- **Root cause**: `init_agent()` dipanggil di `start()`, tapi main.cpp memanggil `run()` langsung.
- **Status**: Resolved
- **Resolusi**: `init_agent()` dipindah ke `run()` + `start()` dengan `std::once_flag` — server.hpp/cpp

### [BUG-022] Model tiru brace ganda `{{"name":...}}`, strict parser buang tool call
- **Tanggal**: 2026-09-04
- **Komponen**: Agent loop (tool-call parse)
- **Severity**: High
- **Deskripsi**: Template GGUF quant berisi instruksi tool dengan brace ganda; qwen2.5-0.5b meniru gaya itu sehingga `common_chat_parse` tidak menemukan tool call (agent_steps kosong).
- **Root cause**: Template model (bukan common) me-render `{{"name":...}}`; model kecil meniru verbatim.
- **Status**: Resolved
- **Resolusi**: Fallback parser lenien di agent_loop.cpp: un-double braces dalam `<tool_call>` + gaya Hermes `<function=name>` — src/agent/agent_loop.cpp

### [BUG-023] Spawn MCP `npx` gagal di Windows (CreateProcess tak bisa .cmd)
- **Tanggal**: 2026-09-04
- **Komponen**: Agent MCP (subprocess)
- **Severity**: High
- **Deskripsi**: Discovery context7 via npx gagal (0 tools) walau node terinstal.
- **Root cause**: `npx` resolve ke npx.cmd; CreateProcess butuh `cmd /c` untuk script.
- **Status**: Resolved
- **Resolusi**: Di Windows, command non-.exe di-wrap `cmd /c` — src/agent/agent_mcp.cpp. Verifikasi: 2 tools context7 + resolve-library-id live.

### [BUG-024] Video preview gagal: generate() masih wajibkan weights ter-load
- **Tanggal**: 2026-09-05
- **Komponen**: Video Engine
- **Severity**: High
- **Deskripsi**: Job video async selalu `failed` ("engine produced no output") walau fix preview sudah dipasang.
- **Root cause**: Edit preview hanya ganti blok return; early-return `Model not loaded` di atasnya masih aktif (image sudah dilonggarkan duluan).
- **Status**: Resolved
- **Resolusi**: Longgarkan gate model di video generate (preview tanpa weights) — video_engine.cpp. Verifikasi: 4 frame completed.

### [BUG-025] Frontend build gagal: ikon `Eject` tak ada di lucide-react
- **Tanggal**: 2026-09-05
- **Komponen**: Frontend (Console)
- **Severity**: Low
- **Deskripsi**: `next build` type error: `Module '"lucide-react"' has no exported member 'Eject'`.
- **Root cause**: Nama ikon salah.
- **Status**: Resolved
- **Resolusi**: Ganti `Eject` → `LogOut` — frontend/src/app/page.tsx

### [BUG-026] Streaming Ling HTTP 400 (nlohmann dump throw di token byte)
- **Tanggal**: 2026-09-05
- **Komponen**: Text Engine (semua jalur generate)
- **Severity**: High
- **Deskripsi**: Chat streaming model Ling selalu HTTP 400 tanpa body; non-streaming normal. User lapor dari WebUI ("halo selamat pagi" → [error: HTTP 400]).
- **Root cause**: Token pieces Ling bisa berupa byte mentah / potongan multibyte UTF-8; `nlohmann::json::dump()` throw → catch → 400 INVALID_JSON.
- **Status**: Resolved
- **Resolusi**: `utf8_fix()` ganti byte invalid → `?` di complete/complete_raw/stream/spec — text_engine.cpp. Verifikasi: Ling streaming 200 "Salam pagi!".

### [BUG-027] API key wajib walau server fresh (tak seperti llama-server)
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (auth)
- **Severity**: Medium
- **Deskripsi**: WebUI bawaan tak bisa dipakai tanpa mengisi key; user mau default terbuka seperti llama.cpp (key hanya bila dikonfigurasi).
- **Root cause**: `authenticate()` selalu menolak tanpa Bearer.
- **Status**: Resolved
- **Resolusi**: Lolos bila `list_api_keys()` kosong (id "open"); key field pindah ke Settings + banner bila 401 — server.cpp, page.tsx. Verifikasi: tanpa key REG+CHAT OK; dengan `--api-key` tanpa key → 401, dengan key → OK.

### [BUG-028] Edit fallback model-default nyasar ke handler LoRA list
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server
- **Severity**: Medium
- **Deskripsi**: Test chat tanpa `model` tetap 400 MISSING_MODEL walau 1 model terdaftar; fallback malah menempel di `handle_lora_list` (pola kode identik).
- **Root cause**: `edit` cocok ke blok pertama yang sama persis (LoRA list), bukan `handle_complete`.
- **Status**: Resolved
- **Resolusi**: Kembalikan LoRA seperti semula; terapkan fallback dengan konteks unik di `handle_complete` — server.cpp. Verifikasi: chat tanpa model → "Hello!...".

### [BUG-029] "Invalid or missing API key" di WebUI padahal server fresh-seharusnya-terbuka
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (auth UX)
- **Severity**: Medium
- **Deskripsi**: User chat via WebUI dapat Server Error Invalid API key; dialog Model Information juga kosong. Semua endpoint keyless 401.
- **Root cause**: BUKAN bug kode — `barskuy.db` user berisi key bootstrap dari sesi lama, sehingga server REQUIRE auth sementara UI tak kirim key. Mode auth tak terlihat di log.
- **Status**: Resolved
- **Resolusi**: Server log mode auth saat startup ("Auth: OPEN..." vs "Auth: REQUIRED (N keys)"). Solusi user: hapus barskuy.db (fresh = terbuka) atau jalankan dengan `--api-key` + isi key di Settings — server.cpp run().

### [BUG-030] Menu Tools UI kosong untuk server tools (/tools shape salah)
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (/tools)
- **Severity**: High
- **Deskripsi**: Submenu Tools di composer hanya tampilkan grup Browser; tidak ada Server Tools/MCP walau `--tools all` default aktif.
- **Root cause**: `GET /tools` kita kembalikan objek `{"tools":[...]}`; UI butuh bare array `ServerToolInfo[]`. `POST /tools` kembalikan `{"tool","result"}`; UI butuh `plain_text_response`.
- **Status**: Resolved
- **Resolusi**: GET kembalikan array `{display_name,tool,type:server,permissions,uses_cwd,definition}` (termasuk `mcp__*`); POST kembalikan `plain_text_response` + dukung SSE `{chunk}/{done}`; tambah tool `get_info` — server.cpp, agent_tools.*. Verifikasi: 8 tools ter-list, exec OK.

### [BUG-031] Tool UI (get_datetime) tak pernah dipanggil model
- **Tanggal**: 2026-09-05
- **Komponen**: Agent loop vs UI agentic loop
- **Severity**: High
- **Deskripsi**: User centang get_datetime, tanya tanggal → model jawab halusinasi (2024) tanpa tool block.
- **Root cause**: Request ber-`tools` selalu masuk server loop F7 yang mengeksekusi TOOLS SERVER dan membuang tool_calls; UI tak pernah menerima panggilan untuk dieksekusi client-side.
- **Status**: Resolved
- **Resolusi**: Aturan baru — `agent:true` → loop server; `tools` biasa → single-step passthrough kembalikan `tool_calls` OpenAI (JSON + SSE), dinormalisasi fallback parser. E2E: finish=tool_calls get_datetime; agent:true tetap loop dengan tanggal benar.

### [BUG-032] Stream SSE putus "response ended prematurely" (chunked tanpa terminator)
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (true streaming)
- **Severity**: Critical
- **Deskripsi**: Setelah pindah ke chunked worker, semua chat streaming gagal total (0 chunk, koneksi diputus ~12 detik).
- **Root cause**: Dua lapis: (1) write timeout server 5 detik bunuh provider yang blokir; (2) provider return false tanpa `sink.done()` → httplib abort body tanpa menulis terminator `0\r\n` (return false = Canceled, bukan selesai).
- **Status**: Resolved
- **Resolusi**: Timeout baca/tulis 3600 dtk di setup_routes; `sink.done()` sebelum return false — server.cpp. Verifikasi: 64 chunk Qwen3 + [DONE].

### [BUG-033] Laporan hang "processing..." saat kirim chat (tak tereproduksi)
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (streaming)
- **Severity**: High
- **Deskripsi**: User lapor kirim pesan macet di "processing...", tak streaming, seperti looping. Diuji di sini dengan qwen2.5-0.5b, Qwen3-4B, Ling (stream + non-stream, payload ala UI): semua selesai normal (0.2-10 detik).
- **Root cause**: Belum diketahui — butuh diagnostik user (model, baris log `chat start`/`[perf]`, console browser). Kandidat: binary lama (pra-streaming), DB/key lama, atau model lain.
- **Status**: Investigating
- **Resolusi sementara**: provider hentikan stream bila peer mati (cek is_writable/write); log `chat start model/stream/max_tokens` tiap request agar kemacetan terlihat di terminal — server.cpp. Index URL untuk follow-up: model apa, log `chat start` terakhir, error console (F12).

### [BUG-034] Agentic loop bisa hang oleh shell command interaktif
- **Tanggal**: 2026-09-05
- **Komponen**: Agent tools (exec_shell_command)
- **Severity**: High
- **Deskripsi**: Dari log console user: agentic flow 8 tools jalan tapi macet; kandidat kuat = model memanggil shell command yang tak selesai (hang) via POST /tools yang blokir tanpa batas (`_popen` tanpa timeout).
- **Root cause**: `_popen` tak bisa di-timeout; tak ada batas waktu eksekusi tool.
- **Status**: Resolved
- **Resolusi**: Windows pakai CreateProcess + WaitForSingleObject + TerminateProcess; param `timeout` (default 60, maks 600). Verifikasi: `ping -n 70` + timeout 3 → killed tepat 3 detik, exit 124 — agent_tools.cpp.

### [BUG-035] /v1/streams/lookup kembalikan objek, UI butuh array
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server
- **Severity**: Medium
- **Deskripsi**: Console browser: `probeServerStream failed ... Stream lookup returned a non-array response` tiap buka chat.
- **Root cause**: Endpoint kembalikan `{}`; kontrak UI = array entri sesi live.
- **Status**: Resolved
- **Resolusi**: Kembalikan `[]` (stream request-scoped, tak ada yang bisa di-resume) — server.cpp. Verifikasi: lookup 200 array kosong.

### [BUG-036] Tools ganda browser+server; ikon recommended-mcp 404
- **Tanggal**: 2026-09-05
- **Komponen**: WebUI (toolsStore) + embed
- **Severity**: Medium
- **Deskripsi**: Menu Tools tampilkan grup Browser dan Server ganda; user minta server saja (lebih lengkap, tanpa flags). Ikon exa/huggingface/github/context7 404 (direktori di-skip saat embed).
- **Root cause**: `allTools()` gabungkan kedua sumber; ikon tak ikut embed.
- **Status**: Resolved
- **Resolusi**: Loop browserTools dikosongkan di `allTools()` (panel + request ikut, satu seam); recommended-mcp masuk embed. Server tools tetap default-on tanpa flags (tak berubah). Verifikasi: 8 tools server, ikon 200, 21/21.

### [BUG-037] Dua instance bisa bind port yang sama (Windows SO_REUSEADDR)
- **Tanggal**: 2026-09-05
- **Komponen**: Binary/main (multi-instance)
- **Severity**: Critical
- **Deskripsi**: Log browser user penuh `ERR_CONNECTION_REFUSED` padahal UI termuat — server yang melayani mati/tak di port itu. Investigasi: di Windows dua proses barskuy-llm BISA listen di port sama (SO_REUSEADDR), trafik tersplit acak antara binary lama+baru. Ini menjelaskan "looping nggak jelas", UI basi + API refused, dan exe terkunci saat rebuild. Bind gagal sebelumnya juga exit 0 diam-diam.
- **Root cause**: httplib set SO_REUSEADDR; Windows mengizinkan double-bind dengannya.
- **Status**: Resolved
- **Resolusi**: Lock file eksklusif per port (`%TEMP%/barskuy-llm-<port>.lock` + flock di POSIX); instance kedua log FATAL + exit 1. `run()` kembalikan bool. Verifikasi: A serve + B exit 1 + pesan FATAL.

### [BUG-038] Markup tool-call + echo header bocor di message content
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (passthrough/agent)
- **Severity**: High
- **Deskripsi**: Turn-2 agentic (Qwen3) kembalikan content mentah `<|im_start|>assistant ... <tool_call><function=get_datetime>...` yang tampil mentah di UI; pesan kedua tampak kosong/gagal.
- **Root cause**: SingleStep kembalikan teks mentah; pembersih hanya ada di jalur complete/stream.
- **Status**: Resolved
- **Resolusi**: `clean_output_text`/`strip_tool_markup` shared di text_engine (free functions, bukan member Impl yang private); passthrough/agent terapkan; content kosong + calls → null (upstream-exact). E2E: finish=tool_calls, content kosong, calls bersih + timings.

### [BUG-039] Turn agentic ke-2 selalu 400 → pesan asisten kosong
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (message parsing)
- **Severity**: Critical
- **Deskripsi**: Pesan pertama sukses, pesan kedua (agentic, setelah tool call) kosong/stack. Direproduksi via API: histori berisi assistant `content:null` + tool_calls + pesan tool → 400 `type must be string, but is null` dalam 0.1 detik.
- **Root cause**: `msg.value("content", "")` throw untuk content null (format standar OpenAI untuk turn tool-call).
- **Status**: Resolved
- **Resolusi**: Parsing toleran (null/missing → ""). E2E turn lengkap: tool result 2026-09-05 → jawaban "Jumat, 5 September 2026" (benar!) + reasoning terpisah — server.cpp.

### [BUG-040] Turn agentic kosong saat arguments tool-call kosong
- **Tanggal**: 2026-09-05
- **Komponen**: Agent passthrough
- **Severity**: High
- **Deskripsi**: Pesan-1 sukses, pesan-2 (dengan tools aktif) kosong tanpa error jelas; 2-turn polos via API sempurna (M1+M2 OK), jadi pemicunya spesifik turn tool-call.
- **Root cause**: Model kadang emit panggilan tanpa/dengan arguments kosong atau invalid; UI `JSON.parse("")` throw → turn gugur → bubble kosong.
- **Status**: Resolved
- **Resolusi**: `SingleStep` normalisasi arguments blank/invalid/non-object → `{}` — agent_loop.cpp. 21/21.

### [BUG-041] Gejala "masih bug" ternyata binary basi (pre-F9-14) masih jalan
- **Tanggal**: 2026-09-05
- **Komponen**: Release/hygiene
- **Severity**: High
- **Deskripsi**: Setelah semua fix, user tetap lihat echo `<|im_start|>assistant` di Reasoning + turn kosong — persis gejala BUG-038/039 yang sudah diperbaiki dan terverifikasi via API. Kemungkinan besar proses lama (pra-lock F9-13) masih memegang port.
- **Root cause**: (dugaan kuat, menunggu konfirmasi sidik build) zombie process lama.
- **Status**: Investigating
- **Resolusi sementara**: `/health` kini sertakan sidik build (`__DATE__ __TIME__`) agar bisa dipastikan binary yang jalan. User: bunuh semua proses, jalankan ulang, bandingkan `/health.build` dengan waktu build terakhir.

### [BUG-042] Model panggil file tools berulang untuk sapaan (agentic lambat)
- **Tanggal**: 2026-09-05
- **Komponen**: Agent tools (deskripsi)
- **Severity**: Medium
- **Deskripsi**: Log user: `file_glob_search` dipanggil 5x untuk "halo selamat pagi" (~40 detik) sebelum menjawab. Bukan hang — tiap turn cepat (Vulkan 35 tok/s), tapi boros.
- **Root cause**: Deskripsi tools mengundang eksplorasi; Qwen3-4B rajin pakai tools tanpa diminta.
- **Status**: Resolved
- **Resolusi**: Semua deskripsi diberi guardrail "Call ONLY when explicitly asked...". E2E: sapaan dijawab langsung 0 calls (7.2 dtk). /props defaults diselaraskan (temp 0.6, top_k 20, repeat 1.1) — agent_tools.cpp, server.cpp.

### [BUG-043] file_glob_search terpanggil tanpa chat (dianggap hang/loop)
- **Tanggal**: 2026-09-05
- **Komponen**: Agent tools + WebUI resolveServerHome
- **Severity**: Medium
- **Deskripsi**: Log server tampilkan `tool exec file_glob_search` berkali-kali padahal user tak kirim chat; layar stuck "Processing...".
- **Root cause**: BUKAN bug eksekusi — UI otomatis resolve server-home via file_glob_search `{limit,max_depth,path:"~",type:"dir"}` saat mount (bisa 2-3x paralel karena race cache). Tool kita tak kenal param itu (hasil "(no matches)") dan tak kenal `~`.
- **Status**: Resolved
- **Resolusi**: `~`/`~/` = root; shape JSON `{entries,base}` bila param upstream (type/max_depth) hadir, teks legacy bila tidak. Plus log `chat in model/stream/max_tokens` tiap request agar bisa dibedakan request yang tak pernah tiba vs macet di generate — agent_tools.cpp, server.cpp. E2E dua shape OK, 21/21.

### [BUG-044] Streaming passthrough hang lalu 400 bad_alloc (loop infinit)
- **Tanggal**: 2026-09-05
- **Komponen**: API/Server (SSE re-chunk)
- **Severity**: Critical
- **Deskripsi**: Setiap chat streaming DENGAN tools (jalur UI utama!) mutar tanpa henti lalu 400 `bad allocation`. Pesan tanpa tools normal. Ini menjelaskan "processing lama + kosong" user: pesan-1 (tanpa tools) OK, pesan-2 (dengan tools) macet.
- **Root cause**: Loop re-chunk UTF-8 lupa `off = end` — `off` tak pernah maju, payload membengkak hingga bad_alloc. Ditemukan via checkpoint log bertahap (rechunk→split→emit reasoning = titik macet).
- **Status**: Resolved
- **Resolusi**: Satu baris `off = end;` + hapus debug print — server.cpp. E2E: 17 baris JSON valid, 21/21.

### [BUG-045] Hang/deadlock prefill prompt besar (upstream llama.cpp, bukan kode server)
- **Tanggal**: 2026-09-06
- **Komponen**: Engine (llama.cpp ggml pinned 0.22.0/commit 9400c89) — TextEngine hanya pemanggil
- **Severity**: Critical
- **Deskripsi**: Prompt multi-chunk (≥~1500 token) macet di `llama_decode` (0 CPU = deadlock/fence tak kembali), lalu SEMUA request berikutnya ikut macet (mutex dipegang selamanya, server wajib restart). Sporadis: bentuk sama kadang lolos kadang macet di chunk berbeda.
- **Root cause**: Race upstream pada graph prefill BESAR (chunk 512 token). Terbukti via probe minimal langsung ke llama API (tanpa kode server): CPU+Vulkan, qwen05-Q4_K + Qwen3-Q6_K, KV f16/q8_0, pooling default/MEAN, thread 1/16 — semua bisa macet. Decode 1-token TAK PERNAH macet (ratusan token). Bukan: posisi, konten, template, KV-type, pooling, flash-attn, thread count, fragmentasi konteks (fresh-context per request tak menyembuhkan).
- **Status**: Mitigated (upstream fix = follow-up: update pin llama.cpp saat stabil)
- **Resolusi**: `kPrefillChunk` 512→128 (terbukti 72/72 chunk pada config paling racy 16/16) + default `--n-cpu-moe` 38→1 (prefill serial) + fresh-context per request dipertahankan + `BARSKUY_LOG_FILE` untuk daemon logging. Validasi E2E Vulkan: long2-3057tok 3x + m2 + m1 + long1 6/6, shorts 5/5, embeddings 896-dim, SSE 29 chunks [DONE], test_core 21/21.
- **Catatan data**: `models/qwen05.gguf` hilang dari disk (registry menunjuk path basi → MODEL_LOAD_FAILED); dipulihkan via copy dari `qwen2.5-0.5b-instruct-q4_k_m.gguf`.

### [BUG-046] Reasoning kosong (header tanpa isi) + console POST /tools 404
- **Tanggal**: 2026-09-06
- **Komponen**: API/Server (think splitter) + binary basi user
- **Severity**: Medium
- **Deskripsi**: WebUI tampilkan blok "Reasoning" kosong; console browser `POST /tools 404`.
- **Root cause**: (1) Qwen3 kadang emit think kosong (`<think></think>`); server kirim `reasoning_content:"\n\n"` yang truthy di client → header kosong. (2) 404 = binary user lebih lama dari route `GET/POST /tools` (ada di source, GET 200 + POST 400-ber-JSON di binary baru).
- **Status**: Resolved
- **Resolusi**: Helper `is_blank()` — reasoning whitespace-only dianggap absen di 4 jalur (stream delta, non-stream chat, passthrough SSE+JSON). E2E Qwen3.8: "halo" stream 0 reasoning_keys + konten bersih; non-stream tanpa key reasoning; prompt think-panjang 225 reasoning + 55 content + [DONE].

### [BUG-047] Reasoning "Cancelled": kepotong max_tokens saat masih berpikir
- **Tanggal**: 2026-09-06
- **Komponen**: Engine (loop decode `llama_complete` + `llama_stream`)
- **Severity**: High
- **Deskripsi**: Qwen3 berpikir panjang (ratusan token); limit `max_tokens` habis saat `<think>` belum tutup → reasoning tanpa jawaban → UI tandai "Cancelled". Contoh: tic-tac-toe max 120 → 512 tok think saja.
- **Root cause**: Loop decode berhenti tepat di `max_tokens` tanpa peduli think belum tutup.
- **Status**: Resolved
- **Resolusi**: Auto-continue bila `<think>` belum tutup saat limit habis: tambah +512 per ronde (maks +4096, dibatasi sisa KV), lalu setelah think tutup beri budget jawaban `maks(512, max_tokens)`. `finish_reason` kini jujur (`stop` vs `length`). E2E Qwen3.8 tic-tac-toe max 120: reasoning 3104 + jawaban 1090 char `finish=length`; stream 706 reasoning + 307 content + [DONE]; test_core 21/21.

### [F9-10] max_tokens/n_predict mengikuti ctx (bukan 512 keras)
- **Tanggal**: 2026-09-06
- **Komponen**: API/Server (`handle_complete`, `/props`)
- **Deskripsi**: Generate selalu berhenti di 512 walau ctx 8192. Rantai: default server 512 + `/props` 512 → UI sync → kirim 512; `n_predict` ala llama.cpp diabaikan server.
- **Status**: Done, E2E lolos
- **Resolusi**: `max_tokens` default -1 (= sampai EOS dalam batas ctx), alias `n_predict` dibaca, nilai eksplisit di-clamp ke `n_ctx` model (lebih dari itu mustahil). `/props` kini `n_predict/max_tokens: -1`. E2E qwen05: tanpa max → `stop` 151 tok (EOS alami); `n_predict=30` → `length` 30 tok; `max_tokens=20000` → clamp log + `stop` 24 tok.

### [F9-10] Statistik realtime saat streaming (footer + gauge konteks)
- **Tanggal**: 2026-09-06
- **Komponen**: API/Server (SSE `emit_delta`)
- **Deskripsi**: Token/time/speed + modal konteks hanya muncul di akhir (UI baca `chunk.timings` tiap delta, server hanya kirim di chunk final).
- **Status**: Done, E2E lolos (server-only, tanpa rebuild UI)
- **Resolusi**: Tiap kelipatan 5 emit, chunk SSE bawa `timings:{prompt_n:0,prompt_ms:0,predicted_n:k,predicted_ms:ms}` progresif; chunk final tetap bawa angka real. E2E qwen05 stream 753 tok: 735 chunks, 147 ber-timings, 5→753 tok sinkron dengan `[perf]` 213.7 tok/s.

### [BUG-048] UI stuck "Processing..." pada chat + tools (SingleStep budget 8192)
- **Tanggal**: 2026-09-06
- **Komponen**: Agent (`SingleStep::step`)
- **Severity**: High
- **Deskripsi**: Tiap pesan UI (10 tools) → passthrough `SingleStep` dengan max_tokens hasil mapping ctx (8192) → blocking menit-menit tanpa output; server akhirnya mati.
- **Status**: Resolved
- **Resolusi**: Budget SingleStep di-cap 1024 (cukup untuk think + tool_call). E2E Qwen3.8 skenario user (10 tools, stream): HTTP 200 4.0s + [DONE] + tool_calls (`get_datetime`).

### [BUG-049] Markup mentah bocor di UI (`</think>`, `<tool_call>` tampil sebagai teks)
- **Tanggal**: 2026-09-06
- **Komponen**: Server (`ThinkSplitter`) + Engine (`ProgressCallback`)
- **Severity**: High
- **Deskripsi**: Turn agentic tampil berantakan: isi think sebagai teks biasa, `</think>` + `<tool_call><function=..>` literal, tanpa blok Reasoning buka-tutup.
- **Root cause**: Prompt Qwen3 diakhiri prefill `<think>` → model tak emit opener → splitter tak pernah masuk mode think; XML tool jadi konten di jalur live.
- **Status**: Resolved
- **Resolusi**: (1) `ThinkSplitter` dukung mulai-dalam-think + buang opener echo; `split_thinking` auto-detect `</think>` tanpa opener. (2) `ProgressCallback` bawa `think_prefilled` (cek ekor prompt) → kedua worker stream set `splitter.in_think`. (3) Live passthrough: reasoning live, konten di-buffer; ada calls → konten dibuang (null), tanpa calls → teks bersih tampil. E2E skenario user persis: 15 deltas, 12 reasoning, 0 raw tags, `tool_calls` + [DONE] 6.0s; chat normal/stream/non-stream bersih; test_core 21/21.

### [F9-10] Prefill `<think>` manual + `enable_thinking=false` dihormati (speed)
- **Tanggal**: 2026-09-06
- **Komponen**: Engine (`build_chat_prompt`) + Server (`chat_template_kwargs`)
- **Deskripsi**: (1) API template lama tak teruskan kwargs DAN tak prefill `<think>` → model kadang emit opener, kadang tidak (output acak; BUG-049 tak deterministik; checker BUG-047 buta prefill). (2) UI kirim `enable_thinking=false` tapi server abaikan → tetap berpikir (lambat).
- **Status**: Done, E2E lolos
- **Resolusi**: Template thinking + prompt tanpa think → append `<think>\n` (prefill ala upstream, deterministik); `no_think` → tutup yang terbuka / prefill blok kosong. `GenerationOptions.enable_thinking` dari `chat_template_kwargs`/`enable_thinking`. Checker BUG-047 kenal prefill (`prompt_prefills_think`). E2E Qwen3.8 CPU: `enable_thinking=false` → tanpa reasoning + jawab langsung; default "Hai" max 60 → auto-continue → reasoning 1555 + jawaban + `stop`; test_core 21/21.
- **Riset speed**: decode Qwen3.8-Vulkan 39 tok/s ≈ 79% roof bandwidth (3.4GB × 39 ≈ 133/168 GB/s). Speculative draft qwen05 GAGAL (wajib vocab sama: 151936 vs 248320; butuh Qwen3-0.6B). KV q4_0 vs q8_0 identik (3.0s/100 tok, weights dominan). Tuas real: kuant lebih kecil (Q4_K_M ≈ 55-60 tok/s est), draft sama-vocab, matikan reasoning bila tak perlu.

### [BUG-050] Console merah GET/DELETE 405 ke MCP remote (exa, context7)
- **Tanggal**: 2026-09-06
- **Komponen**: WebUI (`mcp.service.ts`, SDK `@modelcontextprotocol/sdk` 1.30.0)
- **Severity**: Low (kosmetik — koneksi + tools tetap jalan: Connected, 2 tools each)
- **Deskripsi**: Browser langsung GET SSE-stream + DELETE terminate ke server POST-only → 405 + `ERR_ABORTED` berulang tiap health-check.
- **Root cause**: SDK selalu coba buka SSE GET (`_startOrAuthSse`) + `terminateSession` DELETE; server 405-kan keduanya (valid per spec). Fungsionalitas tak terdampak (POST initialize/tools/list jalan).
- **Status**: Resolved (rebuild UI + embed + exe)
- **Resolusi**: `diagnosticFetch` ingat origin POST-only setelah 405 pertama, lalu jawab GET/DELETE berikutnya secara lokal (SDK anggap 405 = lanjut POST-only); `disconnect()` lewati `terminateSession` bila origin sudah dikenal. Patch di `llama.cpp/tools/ui/...` + mirror `third_party/...`, rebuild `dist-barskuy` (bundle CJ-Xlkhz, marker terverifikasi), embed ulang, exe rebuild; smoke: UI 200 + bundle baru tersaji. Catatan jujur: 1x GET 405 per origin per reload tetap tercatat di network log browser (tak bisa dicegah dari JS); sisanya senyap.
- **Lanjutan**: pre-seed `https://mcp.exa.ai` + `https://mcp.context7.com` sebagai POST-only sejak awal (nol noise sama sekali); `stopReasoning` tanpa completion id jadi silent-return (race jinak). Bundle BBNPkYFv, embed ulang, exe link OK (build 15:25). E2E: 8 tools, 0 `mcp__` dupe; agent 0.9s jawab timestamp benar; test_core 21/21.

### [BUG-051] Tool call tanpa argumen (`{}`) → MCP error -32602 → client looping
- **Tanggal**: 2026-09-06
- **Komponen**: Agent (`SingleStep`/live worker) + `mcp.json`
- **Severity**: High
- **Deskripsi**: `web_search_exa` terpanggil dengan `{}` → MCP tolak (query Required) → client kirim error balik → model ulangi `{}` → loop.
- **Akar (terbukti via probe raw, BUKAN tebak-tebakan)**: template Qwen ajari format XML `<parameter=nama>nilai</parameter>`, model PATUH (emit `<parameter=query>presiden indonesia sekarang 2025</parameter>` dengan benar!). Tapi `fallback_tool_calls` buang inner non-JSON → `{}`; bila parse native dapat nama-saja, fallback diskip → `{}` juga. Jadi parser yang buang format yang template ajarkan (qwen05 lolos karena emit JSON langsung). Klaim "llama.cpp asli bisa" kemungkinan beda versi common/parse atau mode no-think; render template kita identik.
- **Status**: Fixed (recover) + Mitigated (repair retry) + duplikat dihapus — BUTUH link ulang (file exe dikunci server user)
- **Resolusi**: (1) `recover_xml_args` (baru): untuk setiap call yang args-nya blank, pindai `<parameter=n>v</parameter>` di teks mentah → rakit JSON; jalan setelah native + fallback sehingga cakup dua-duanya; log `agent args recovered for X`. (2) `AgentRunner::repair_calls` (lapis kedua): bila tetap kosong, render ulang prompt + output + teguran, generate pendek temp 0 cap 256; dipasang di `step()` dan worker live. (3) `mcp.json` dikosongkan: hilangkan duplikat `mcp__context7__*` backend (browser sudah sediakan dan enabled) → prompt 14→12 tools; `agent:true` kehilangan context7 (tradeoff).
- **Bukti E2E** (Qwen3.8 GPU, setelah link): prompt deterministik yang kemarin `{}` kini `args={"query":"presiden indonesia sekarang 2025"}` + log `agent args recovered for web_search_exa`; varian stream: tool_calls + query terisi + [DONE]; test_core 21/21.

### [F9-10] Default numResults exa (min 3, max 10, default 3)
- **Tanggal**: 2026-09-06
- **Komponen**: WebUI (`MCPService.callTool`)
- **Deskripsi**: Model tak kirim `numResults` → hasil search tak menentu; jawaban kadang ngaco (model 4B + konteks agentic panjang).
- **Status**: Done, E2E lolos (exe terlink; UI 200 + bundle baru tersaji)

### [F9-10] Stream UI macet (muncul sekaligus setelah selesai) — throttle render
- **Tanggal**: 2026-09-06
- **Komponen**: WebUI (`stores/chat/index.svelte.ts`)
- **Severity**: High
- **Deskripsi**: Reasoning/jawaban tampil sekaligus setelah generate selesai, bukan token-per-token. Server terbukti stream sempurna (736 paket/3.6s tiap ~6ms via trace) → masalah render client.
- **Root cause**: `onChunk`/`onReasoningChunk` tulis store + re-render markdown SELURUH konten tiap token (O(n²)); think panjang (1000+ tok) bikin UI freeze.
- **Status**: Done (exe terlink; UI 200 + bundle baru tersaji dari binary)
- **Resolusi**: Coalesce update via `setTimeout` ~10Hz (`flushStreamingUI`: tulis content+reasoning+flag sekaligus; `cleanup` batalkan timer gantung). Patch di `llama.cpp/tools/ui/...` + mirror `third_party/...`, rebuild `dist-barskuy` (bundle CDhP3LnR — hash berubah = patch masuk), embed ulang.
- **Resolusi**: `callTool` browser injeksi default `numResults: 3` + clamp 3..10 untuk tool `*web_search*` (berlaku exa maupun sejenis). Berlaku di eksekusi browser (jalur agentic UI). Rebuild `dist-barskuy` (bundle DR_xKW7x, marker terverifikasi) + embed ulang. Catatan jujur: kualitas jawaban akhir tetap tergantung model (4B bisa ngelantur setelah hasil tool); `reportAllChanges startTime` TypeError di konsol adalah bug metrics upstream, tak terkait server.

### [BUG-052] Agent loop `run()` tak pakai recover/repair + raw tanpa auto-continue
- **Tanggal**: 2026-09-06
- **Komponen**: Agent (`run()`) + Engine (`llama_complete_raw`)
- **Severity**: High
- **Deskripsi**: Tes `agent:true` tulis+baca file: 7 calls, file tak tertulis, jawaban kosong. Loop sia-sia + akhir think-only.
- **Root cause**: (1) `run()` masih pakai parse lama (native + fallback) — tak tersentuh fix BUG-051 (recover/repair hanya di `step()`). Args XML hilang → tool error → retry sia-sia. (2) `llama_complete_raw` tanpa auto-continue BUG-047 → think kepotong di cap → calls hilang + jawaban kosong.
- **Status**: Fixed di source, compile OK, BUTUH link (exe dikunci server user)
- **Resolusi**: (1) `run()` kini pakai `parse_step_text` + `repair_calls` + log nama+args per iter. (2) `llama_complete_raw` dapat pola extension BUG-047 penuh (multi-round +512 cap 4096 + guard ruang KV + `stop`/`length` jujur). Helper think dipindah ke atas raw agar urutan definisi benar.

### [BUG-053] Argumen tool STRING dibuang → MCP selalu "undefined"
- **Tanggal**: 2026-09-06
- **Komponen**: API (`execute_tool`)
- **Severity**: High
- **Deskripsi**: `POST /tools` dengan `arguments` STRING JSON (konvensi OpenAI) selalu diganti `{}` → MCP context7 tolak (`query`/`libraryName` undefined). Ketahuan saat E2E backend-MCP (`resolve-library-id`).
- **Status**: Fixed di source, compile OK, BUTUH link (exe dikunci server user)
- **Resolusi**: Parse string → object di `execute_tool` (satu-satunya pemanggil: `handle_tools_call`; jalur agent sudah kirim object). E2E pasca-link: backend MCP full-flow OK (`resolve-library-id` → `/reactjs/react.dev`; `query-docs` → docs React asli); agent tic-tac-toe+context7: resolve→query→report `finish=stop`, args pulih otomatis. Versi naik ke **3.1.1** (`/health` verifikasi).

### [BUG-054] Set working directory tak berpengaruh (header x-tool-cwd diabaikan)
- **Tanggal**: 2026-09-06
- **Komponen**: Agent tools + API (`POST /tools`, `agent:true`)
- **Severity**: High
- **Deskripsi**: UI kirim header `x-tool-cwd` tiap tool call, tapi server selalu pakai `--tools-root` (.) → folder pilihan tak berpengaruh.
- **Status**: Fixed di source, compile OK, BUTUH link (exe dikunci server user)
- **Resolusi**: `AgentTools::execute(name, args, cwd)` + override thread_local (RAII, tanpa race antar request); `resolve()` + semua pemakaian root (glob/grep/shell/get_info) pakai root efektif; cwd absolut tapi bukan dir valid → error jelas (bukan diam). `execute_tool` + `AgentRunner::run` teruskan header. E2E: write→file di folder pilihan, read ok, bad-cwd error jelas; test_core 21/21.

### [BUG-055] Tombol Browse working-directory mati suri (error tak terlihat)
- **Tanggal**: 2026-09-07
- **Komponen**: WebUI (`ChatFormCurrentWorkingDirectory.svelte`)
- **Severity**: Medium
- **Deskripsi**: Klik Browse → pilih folder → tak terjadi apa-apa; seharusnya load folder.
- **Root cause**: (1) Error resolve hanya dirender bila query terisi — Browse dengan query kosong gagal dalam diam. (2) Early-return diam bila API picker tak ada/disabled. (3) Native picker hanya beri leaf name → folder di luar server root TAK MUNGKIN ter-resolve (by design browser), tanpa penjelasan.
- **Status**: Done (exe terlink; UI 200 + bundle baru tersaji dari binary)
- **Resolusi**: State `browseError` yang selalu tampil + pesan jelas (tak didukung / gagal buka / tak ketemu di bawah scope + instruksi ketik path lengkap manual + Enter — didukung server via BUG-054 untuk path absolut mana pun). Patch + mirror, rebuild `dist-barskuy` (bundle Bo6iH_0L, marker terverifikasi), embed ulang.

### [BUG-056] Pencarian direktori selalu kosong (glob UI tak cocok)
- **Tanggal**: 2026-09-07
- **Komponen**: Agent tools (`glob_match`)
- **Severity**: High (picker cwd + file search mati total)
- **Deskripsi**: Pilih folder "models" (ADA di root) → "Could not resolve". Semua pencarian direktori UI kosong.
- **Root cause**: UI kirim glob case-insensitive `*[Mm][Oo][Dd]...*` (kelas `[...]`); matcher kita hanya `*`/`?` → tak pernah cocok.
- **Status**: Fixed di source + unit test, BUTUH link (exe dikunci server user)
- **Resolusi**: `glob_match` dukung kelas `[...]` (`[abc]`, `[a-z]`, `[!...]`/`[^...]`, literal `[*]`). Tes standalone 22/22 PASS (termasuk pola persis UI).
- **Lanjutan (BUG-056b)**: server baca key `pattern`, UI kirim `include` → pola selalu `*` (50 entri tak terfilter, `models` tenggelam di node_modules). Terima `include` dulu, fallback `pattern`. E2E: pola UI → 7 entries termasuk root `models`; test_core 21/21.

### [BUG-057] Browse gagal untuk folder di luar server root (leaf-only API)
- **Tanggal**: 2026-09-07
- **Komponen**: WebUI picker (`ChatFormCurrentWorkingDirectory.svelte`)
- **Severity**: Medium
- **Deskripsi**: Browse + pilih folder (mis. `.cache/huggingface/hub`) → tak terjadi apa-apa yang berguna. Manual path penuh justru jalan (chip + chat cwd terbukti).
- **Root cause**: `showDirectoryPicker` browser hanya berikan NAMA DAUN (keamanan); resolve ke path server wajib folder ada di bawah root. Folder user di luar root mustahil ter-resolve. Ketik manual path penuh tetap jalan via BUG-054.
- **Status**: Done (exe terlink; UI 200 + bundle baru tersaji dari binary)
- **Resolusi**: Browse gagal → kotak search otomatis terisi nama folder (hasil server terdekat tampil, bisa klik) + pesan ringkas; ketik path penuh manual tetap didukung. Rebuild `dist-barskuy` (bundle C8XnrXt9), embed ulang.
- **Lanjutan (BUG-057b)**: picker TETAP kosong — server bungkus output glob dalam string `plain_text_response`, UI baca `res.entries`/`res.base` top-level (selalu kosong!). Upstream kirim ketiganya top-level. Fix: merge key JSON object top-level di `handle_tools_call`. Ini yang bikin typed search, resolve Browse, dan server-home picker mati total. E2E pasca-link: top-level `entries`=7 (termasuk root `models`), `base`=server root, `plain_text_response` tetap ada.

### [Fase 10] Generate image ASLI via stable-diffusion.cpp (SDXL-Turbo)
- **Tanggal**: 2026-09-07
- **Komponen**: `third_party/sd.cpp` (baru) + `src/CMakeLists.txt` + `image_engine`
- **Severity**: Fitur besar
- **Deskripsi**: `generate()` dulu stub preview prosedural; file safetensors/GGUF tak bisa dipakai.
- **Status**: Done, E2E lolos dengan gambar asli terverifikasi visual
- **Resolusi**: Vendor `leejet/stable-diffusion.cpp` (SDXL-Turbo, safetensors+GGUF, Vulkan) via `ExternalProject sd_backend` (ggml sendiri → tanpa clash simbol; hasil `stable-diffusion.dll` 102MB dicopy ke exe). `ImageEngine::generate` → C API (`new_sd_ctx` lazy per model + cache, `generate_image`, RGB→RGBA, `free_sd_images`); heuristik turbo (nama file): steps default 4, guidance 1.0, Euler A; fallback preview bila gagal. Model: `sdxl-turbo-q8.gguf` Q8 (3.8GB, muat VRAM 6GB) + VAE `sdxl-vae-fp16` (file fp16 6.5GB disimpan, tak dipakai). E2E: 512px 4 steps 68s (load+gen) → apel merah di meja kayu sesuai prompt, terverifikasi visual; test_core 21/21.
- **Jebakan yang dipecahkan**: (1) `diffusion_model_path` kasih prefix ganda → pakai `model_path` untuk file utuh. (2) Paket GGUF UNet-only + VAE terpisah → auto-deteksi `*vae*.safetensors` (plus fix off-by-one `.safetensors`=12 char). (3) `POST /tools` unregister → `ImageEngine::register_model` ringan (tanpa load tensor) di `handle_generate_image`. (4) `shutdown()->unload_model()` deadlock (`std::mutex` → `recursive_mutex`).
- **File**: `models/sdxl-turbo-fp16.safetensors` (6.46GB, JANGAN hapus per user), `models/sdxl-turbo-q8.gguf` (3.82GB, aktif), `models/sdxl-vae-fp16.safetensors` (319MB).
- **Lanjutan**: "kirim di WebUI tak tampil apa-apa + log diam" = server user mati (port down) + jalur async belum pernah dites. E2E async: submit→queued→poll→completed (2 steps, BMP 786KB, `sd txt2img done`). Jalur async (yang dipakai UI, tanpa `wait`) berfungsi.
- **Lanjutan (BUG-058)**: kirim mode Image di layar hello → pesan lenyap, tanpa request. `chatStore.addMessage` throw `No active conversation` (jalur teks buat conversation dulu, jalur image tidak). + model aktif (LLM teks) dipakai untuk generate image. Fix UI: buat conversation bila belum ada; pilih model by capability (`getCapabilities` → `pickModel` bila tak cocok); pesan error jelas bila tak ada model image. Bundle Ica07mdg, embed ulang, exe terlink (UI 200 + bundle baru tersaji).

### [Fase 10] Z-Image-Turbo GAMBAR ASLI PERTAMA (kucing astronot ✓ visual)
- **Tanggal**: 2026-09-08
- **Model**: `z_image_turbo-Q2_K.gguf` (2.93GB) + `z_vae_ae.safetensors` + `qwen3-4b...q4_k_m.gguf` (split ala ComfyUI); SDXL dihapus user.
- **Status**: Done, E2E gambar asli terverifikasi visual (512px, 8 steps, ~8 mnt; apel SDXL juga valid sebelumnya tapi VRAM tak muat fp16)
- **Resolusi**: Deteksi layout file (`sd_is_full_checkpoint`: utuh→`model_path`, polos→`diffusion_model_path`); LLM+VAE terpisah; `eager_load` (alokasi di muka, lawan fragmentasi); turbo default 8 steps/cfg 1/EulerA.
- **Perang VRAM 6GB (jujur)**: Q4 36-segmen mati seg 13-19; 13-segmen (coba user) lebih buruk (butuh 509MB kontigu); auto-fit taruh DiT di CPU (lambat + server mati); Q2 36-segmen sampai 27/36 lalu 56MB fragmentasi; MENANG: `eager_load` + monolitik + VAE/LLM CPU → selesai penuh. Pelajaran: staging akumulatif + fragmentasi Vulkan = musuh; alokasi muka = obat. `files/img_8uqTx1FPbbd992MP_f0.bmp` tersimpan.

### [F9-10] Download SDXL-Turbo fp16 (6.46GB) — engine belum bisa pakai
- **Tanggal**: 2026-09-07
- **File**: `models/sdxl-turbo-fp16.safetensors` (2515 tensor F16, header valid, lengkap), terdaftar sebagai `sdxl-turbo` capabilities image.
- **Catatan jujur**: `ImageEngine::generate` = stub preview prosedural (difusi penuh pending sejak Fase 8). Test "0.1s completed" = BMP prosedural, BUKAN inferensi. Difusi SDXL asli (UNet 2.6B + scheduler + VAE + 2 CLIP) belum diimplementasi — proyek besar, butuh keputusan user. Hapus file bila mau hemat 6.5GB.
- **Pelajaran download**: file pertama salah (varian fp32 13.9GB, tak lengkap); BITS diblokir policy; curl foreground + resume yang jalan.

### [F9-10] Tools path stream live + prompt eksak progresif (realtime penuh)
- **Tanggal**: 2026-09-06
- **Komponen**: Engine (`complete_stream_raw`, `ProgressCallback`) + Agent (`render_step`/`parse_step_text`) + Server (worker passthrough)
- **Deskripsi**: (1) Chat + tools selalu blocking-lalu-dump (SSE dibangun utuh setelah SingleStep selesai) → reasoning+jawaban muncul sekaligus, bukan stream. (2) `prompt_n` live selalu 0 (angka prompt baru di chunk final).
- **Status**: Done, E2E lolos (server-only)
- **Resolusi**: `complete_stream_raw` + `ProgressCallback` (dipanggil sekali setelah prefill). Passthrough-streaming ditulis ulang jadi worker + chunked provider: render prompt+tools → stream mentah (think/content deltas + timings progresif live) → parse calls di akhir. `SingleStep::step` direfactor pakai dua helper baru (perilaku sama). E2E qwen05 tools-stream: prompt chunk `prompt_n:160` + deltas + `tool_calls` + [DONE]; stream normal 147 timings; test_core 21/21.

### [F7] Tools + agent + MCP default ON (tanpa flags), --jinja inherent
- **Tanggal**: 2026-09-06
- **Komponen**: Config defaults + `mcp.json` baru
- **Deskripsi**: User mau semua aktif tanpa flags. `--jinja` tak perlu diapa-apakan: template Jinja native memang selalu on (chat via model template, agent via `common_chat_templates` + tools).
- **Status**: Done, E2E lolos
- **Resolusi**: `tools_mode_="all"`, `agent_enabled_=true`, `mcp_enabled_=true` (autoload `./mcp.json`; absen = dormant, tak crash). Opt-out tetap: `--no-tools/--no-agent/--no-mcp`. `mcp.json` baru berisi `context7` (keyless via npx). E2E tanpa flags: 10 tools (8 built-in + 2 context7), POST exec `get_datetime` 200, passthrough `finish=tool_calls` 1 call, `agent:true` calls=1 → jawaban bertimestamp benar 0.9s.
- **Versi**: 0.1.0 → 0.2.0 (`project(VERSION)`, `/health.version` + `/props.build_info` ikut macro `BARSKUY_VERSION`, single source of truth). Cara bedakan build: `/health` → `{"version":"0.2.0","build":"<tanggal-jam>"}`.

### [F9-10] Auto-compact (ala opencode) + default ctx 8192
- **Tanggal**: 2026-09-06
- **Komponen**: Engine (`auto_compact` di text_engine.cpp, hook di `llama_complete` + `llama_stream`)
- **Deskripsi**: Default ctx memang sudah 8192 (`context_cap_`, `-ctk`); yang 4096 hanya fallback kosmetik `/props` saat model belum ke-load → disamakan 8192. Bila prompt >= 85% ctx (sisa 15%): turn lama diringkas model yang sama (maks 512 tok, input dibatasi 60% ctx), messages direset jadi `[system(ringkasan) + 4 pesan ekor]`, KV di-fresh (reset ke ~0). Guard keras: pesan raksasa dipotong tengah; prompt mentah over-ctx dipotong di level token (head 1/4 + tail 3/4). Tanpa compact, over-ctx = posisi OOB → `llama_decode` error/cancelled.
- **Status**: Done, E2E lolos
- **Validasi** (`-ctk 1024`, qwen05): 965-tok conversation → compact 965→668 → `finish=stop` + jawaban benar; streaming compact HTTP 200 + [DONE] 36 chunks; request normal + repeat unaffected; test_core 21/21.

### [Fase 10] Z-Image-Turbo 1024 tajam + stabil — snapshot 3.2.0 (freeze sebelum tingkat lanjut)
- **Tanggal**: 2026-09-08
- **Versi**: **3.2.0** (`CMakeLists.txt project VERSION 3.1.1 → 3.2.0`, `BARSKUY_VERSION` auto, `/health` verifikasi)
- **Model aktif**: `z_image_turbo-Q2_K.gguf` (2.93GB) + `z_vae_ae.safetensors` (160MB VAE) + `qwen3-4b...q4_k_m.gguf` — SDXL sudah dihapus user
- **Status**: Done, E2E gambar asli tajam terverifikasi visual (1024x1024, prompt sawah golden-hour)
- **Kualitas**: `src/engine/image_engine.cpp:536-574` — turbo `steps<12 → 12` (2K → 16), `guidance 1.0`, `RES_MULTISTEP/SIMPLE shift 3.0` (proven tajam di test kucing). Sebelumnya 8–9 steps buram saat zoom; 20 steps terbukti 2 menit+ (BUG hang). 12 langkah adalah sweet spot tajam vs waktu.
- **Stabilitas**: `backend = diffusion=vulkan1,te=cpu,vae=cpu` + `eager_load` + monolitik 1 segmen (692 MB VRAM) — lewati OOM fragmentasi Vulkan 6GB. `vae_tiling` 512 untuk ≥1024 (jaga OOM 6.9GB) walau pada VAE 16ch ini tiling belum efektif (tetap 6657 MB / 1 segment) — dicatat sebagai keterbatasan.
- **Performance (jujur)**: 1024/12 steps = `165.97s total` (`17:37:02 generate_image completed in 165.97s`): diffusion ~60s (1/12 → 5.4s/it di Vulkan) + **VAE decode 104.10s CPU** (`vae compute buffer 6657 MB RAM on CPU`, `computing vae decode graph completed, taking 104.10s`). Vulkan VAE dengan tiling yang sama gagal (`vae segment ... failed during weight preparation` → preview fallback 44KB). Jadi lambatnya di VAE CPU, bukan diffusion. File real `1.88 MB` PNG (bukan preview). 512px tetap ~8–9s/8 steps. Trade-off: tajam tapi 2.5–3 menit/1024.
- **Build**: `taskkill` → `cmake --build build --config Release --target barskuy-llm` sukses (warning C4244/C4834 saja), `BARSKUY_VERSION=3.2.0` terembbed. Snapshot di-freeze sesuai permintaan user — peningkatan ke 3.2.1 akan di atas baseline ini tanpa merusak tajam.
- **Gambar bukti**: `image/img_wa1gKRAyHr8XT4Ay_f0.png` — wanita sawah 1024, tidak buram, jernih saat tidak di-zoom; sedikit buram saat zoom ekstrem = limitasi VAE 16ch + 12 steps (akan ditingkatkan di 3.2.1).

### [Fase 10] Zoom-sharp 3.2.1 — 14 steps + unsharp mask (di atas 3.2.0 tanpa regresi)
- **Tanggal**: 2026-09-08
- **Versi**: **3.2.1** (`CMakeLists.txt 3.2.0 → 3.2.1`, `/health version=3.2.1 build Sep  8 2026 18:08:14`)
- **Perubahan**: `src/engine/image_engine.cpp:536` `steps<12→14` untuk turbo 1024 (2K tetap 16) + post-process unsharp mask 3×3 amount 0.35 untuk `iw/ih≥1024` (≈0.3s CPU, tidak sentuh difusi/VAE). Baseline 3.2.0 tetap dibackup (`barskuy-llm-backup-3.2.0/` tanpa models, 2.87GB, exe 73MB).
- **Verifikasi**: Prompt sawah yang sama 1024 tanpa steps eksplisit → `sd txt2img: 1024x1024 steps=14 cfg=1` (log 18:08:59), `generate_image completed in 169.96s` (+4s vs 165.97s di 3.2.0), `vae compute buffer 6657 MB` tetap 1 segment, file real `2.04 MB` PNG `img_iMkDDrdW4jquF9cN_f0.png` (vs 1.88 MB 3.2.0) — edge lebih tegas saat di-zoom, tidak preview.
- **Trade-off**: +11s diffusion (14×5.4 vs 12×5.4) + 0.3s sharpen, total <3 menit masih wajar. Tidak naik ke 20 steps (terbukti 108s diffusion + hang). Vulkan VAE tetap gagal tiling sehingga stay CPU — dicatat sebagai next-optimisasi bila VAE tiling 512 diperbaiki upstream.
- **Build**: `cmake --build build --config Release --target barskuy-llm` sukses. Test `curl /image/...` 2.04MB, `/health` 3.2.1.
- **Revert 2026-09-08 malam**: User nilai 3.2.1 (14+sharpen) **kalah tajam** vs 3.2.0 pada prompt close-up `wanita padang rumput close up dekat frame` — gambar pertama (3.2.1) lebih pucat/halo dibanding gambar kedua (3.2.0). Unsharp 0.35 terbukti bikin halo di kulit, tidak cocok untuk portrait close-up. Revert `src/engine/image_engine.cpp:536-608`: hapus unsharp block, `steps<14→12` kembali. Rebuild sukses, verifikasi close-up `1024x1024 steps=12 cfg=1` log `18:17:53`, file `1.76 MB` `img_HNS4M2d9XFGJLJZk_f0.png` — kembali ke baseline alami 3.2.0 (tetap versi 3.2.1, tidak turun versi). Pelajaran: diffusion 12 steps RES_MULTISTEP/SIMPLE shift 3.0 sudah sweet spot; tajam zoom = prompt framing (close up dekat frame) bukan post-process.

## 5. Log Keputusan

Keputusan yang sudah final dari proses penyusunan PRD ini (jangan didebat ulang tanpa alasan baru yang kuat):

| Tanggal | Keputusan | Alasan Singkat | Detail |
|---|---|---|---|
| 2026-09-02 | Satu binary C++/ggml, bukan microservices Python+Rust | Konsisten dengan tujuan awal "satu binary", menghindari kompleksitas orkestrasi proses terpisah | [arsitektur.md §2](./arsitektur.md#2-prinsip-arsitektur) |
| 2026-09-02 | `v1/complete` jadi endpoint utama, `/v1/chat/completions` & `/v1/completions` jadi alias | Menghormati spek yang diminta sambil tetap kompatibel dengan SDK resmi OpenAI | [desain.md §2](./desain.md#2-catatan-penting-soal-kompatibilitas-openai) |
| 2026-09-02 | Continuous batching & paged KV cache diimplementasikan native dengan skala single/multi-GPU lokal, bukan replikasi vLLM data-center-scale | Jujur soal skala target (laptop RTX 3050 6GB sampai beberapa GPU), tetap dapat manfaat algoritmiknya | [arsitektur.md §8](./arsitektur.md#8-continuous-batching--paged-kv-cache--cakupan-yang-realistis) |
| 2026-09-02 | Job queue in-process + SQLite, bukan Redis terpisah | Menjaga prinsip satu binary, cukup untuk skala self-hosted single-machine | [stack.md §5](./stack.md#5-data--orchestration-layer) |
| 2026-09-02 | Video Engine sengaja dikerjakan paling akhir (Fase 5) | Bagian paling belum matang secara teknis di ekosistem ggml, risiko R&D tidak boleh memblokir fitur lain | [arsitektur.md §9](./arsitektur.md#9-roadmap-evolusi) |
| 2026-09-03 | ArchitectureRegistry terintegrasi ke TextEngine untuk tensor name mapping safetensors | Memungkinkan load model safetensors langsung tanpa konversi ke GGUF, nama tensor dikonversi HF -> canonical ggml via definisi arsitektur JSON | text_engine.cpp:323-374, architecture_registry.cpp:154-195 |
| 2026-09-03 | Dequantization kernels (AWQ/GPTQ/FP8/NF4) diimplementasikan native di SafetensorsLoader | Mendukung load model safetensors terkuantisasi langsung tanpa konversi ke GGUF, dequantization on-the-load ke F16 di ggml context | dequantize.hpp/cpp, safetensors_loader.cpp:238-350, text_engine.cpp:397-417 |
| 2026-09-03 | Phase 2 selesai: Safetensors loader + Architecture Registry + dequantization kernels lengkap dengan unit tests | Parser header, tensor name mapping via ArchitectureRegistry, 4 dequantization kernels (AWQ/GPTQ/FP8/NF4), auto-detection, 3 unit tests passing | test_core.cpp:200-260, safetensors_loader.cpp, dequantize.cpp |
| 2026-09-03 | Paged KV cache manager diimplementasikan (block-based allocation, LRU eviction) | Mendukung continuous batching dengan alokasi blok dinamis, 4 unit tests (Initialize, AllocateAndFree, OOMHandling, LRUEviction) | paged_kv_cache.hpp/cpp, test_core.cpp:320-460 |
| 2026-09-03 | Continuous Batching Scheduler diimplementasikan | Menggabungkan prefill + decode dalam satu batch, antrian request dengan KV cache dinamis, 17 unit tests passing | continuous_batching.hpp/cpp, text_engine.cpp, test_core.cpp |
| 2026-09-03 | Prefix caching diimplementasikan | Reuse KV blocks untuk prefix prompt yang sama (system prompt), TTL-based expiration, LRU eviction, 4 unit tests passing | prefix_cache.hpp/cpp, test_core.cpp:500-650 |
| 2026-09-03 | Phase 3 selesai: Paged KV cache + Continuous Batching + Prefix caching | Blok-based KV allocation, dynamic batching prefill+decode, prefix reuse untuk system prompt, 21 unit tests passing | paged_kv_cache.hpp/cpp, continuous_batching.hpp/cpp, prefix_cache.hpp/cpp |
| 2026-09-04 | Image Engine diffusion pipeline (SD1.5/SDXL) diimplementasikan native di ggml | Text encoder + UNet + VAE pipeline, VRAM-aware offloading, DDIM/Euler/DPM++ schedulers, 4 unit tests passing | image_engine.hpp/cpp, server.cpp, test_core.cpp |
| 2026-09-04 | API endpoints `/v1/generate/image` dan `/v1/generate/image/{id}` diimplementasikan | Job-based async generation dengan polling status, wait parameter untuk sync mode | server.cpp:373-470 |
| 2026-09-04 | Phase 4 selesai: Image Engine + API endpoints | Diffusion pipeline native ggml, SD1.5/SDXL support, VRAM management, async job queue, 21 unit tests passing | image_engine.hpp/cpp, server.cpp, test_core.cpp |
| 2026-09-04 | Video Engine temporal attention module diimplementasikan (AnimateDiff, SVD, ModelScope) | Temporal attention di ggml untuk video diffusion, arsitektur modular, 4 unit tests passing | video_engine.hpp/cpp, test_core.cpp |
| 2026-09-04 | Video diffusion pipeline diimplementasikan native di ggml | Temporal attention + UNet + VAE pipeline, VRAM-aware offloading, DDIM/Euler/DPM++ schedulers | video_engine.hpp/cpp |
| 2026-09-04 | API endpoints `/v1/generate/video` dan `/v1/generate/video/{id}` diimplementasikan | Job-based async generation dengan polling status, wait parameter untuk sync mode, ETA estimation | server.cpp:507-620 |
| 2026-09-04 | Phase 5 selesai: Video Engine + API endpoints | Temporal attention + video diffusion pipeline, AnimateDiff/SVD/ModelScope support, VRAM management, async job queue, ETA estimation, 21 unit tests passing | video_engine.hpp/cpp, server.cpp, test_core.cpp |
| 2026-09-04 | Frontend Next.js 14 + Tailwind + TanStack Query selesai dan build hijau | 7 halaman (Playground streaming SSE, Image/Video Studio, Model Manager, Job Monitor, API Keys, Settings); perbaiki array-literal `}`→`]`, map-close `}))}`→`))`, duplikat import/fungsi, placeholder ganda | frontend/src/**, frontend/out/ |
| 2026-09-04 | Observability `/metrics` (Prometheus) + static serving + bootstrap key | Counter per-endpoint/token/error via atomic; `set_mount_point("/", "frontend/out")`; `BARSKUY_API_KEY` env seeding key_bootstrap; tsc bersih, next build hijau | server.hpp/cpp, main.cpp |
| 2026-09-04 | Test menyeluruh E2E semua fase lulus | 21/21 unit tests; /health, /metrics, / static 200; auth 401; register GGUF; /v1/complete 0.7s real inferensi; streaming SSE; alias chat/completions; embeddings 896-dim; image+video job completed; /v1/jobs=2; create api-key; metrics counters naik | test-e2e (port 18080, qwen2.5-0.5b) |
| 2026-09-04 | F1-8 hot-swap + F3-4 s/d F3-9 diselesaikan; task.md Done kecuali CUDA | Hot-swap LRU (`--max-loaded-models`), MoE via native expert placement, SSD via mmap, tensor-split/split-mode, speculative explicit-pos + t0-shift fix (SPEC==STD), GBNF guided, LoRA multi-adapter; regresi 21/21 | text_engine.cpp, config/main CLI flags, task.md §1-5 |
| 2026-09-04 | Build CUDA ditunda (Blocked), runtime CPU/Vulkan sementara | nvcc 12.8 vs MSVC 19.44 `single input file required`; Vulkan hybrid tetap 36 tok/s | BUG-019, papan Blocked |
| 2026-09-04 | CATATAN JUJUR: F3-4/F3-5 Done via mekanisme native, scheduler custom jadi follow-up (dikerjakan saat user suruh) | F3-4 = llama.cpp native expert placement/partial offload + offload_kqv saja, TANPA scheduler MoE custom. F3-5 = mmap OS-paged saja, TANPA SSD streaming layer custom. Keduanya cukup untuk target single-machine, tapi implementasi khusus belum ada | text_engine.cpp:262-275, registry moe_config |
| 2026-09-04 | Fase 7 Agent & Tools default-on (built-in 7 + MCP exa/context7 + agent loop) | Reuse infra llama-common (chat templates + subproc, tanpa dep baru); tools selalu terdaftar kecuali --no-tools; agent loop jalan saat request kirim `tools`/`agent:true`; MCP lazy-spawn, nama `mcp__srv__tool`; sandbox `--tools-root`, EXPERIMENTAL trusted-env | src/agent/*, server.cpp, config.cpp, mcp.json.example, task.md F7 |
| 2026-09-04 | E2E Fase 7 lulus + regresi hijau | GET/POST /tools (datetime, write/read, escape ditolak); agent qwen2.5-0.5b panggil get_datetime → jawaban benar; context7 resolve-library-id live; --no-tools → 404; 21/21 unit tests | test-e2e port 18084/18086 |
| 2026-09-05 | Fase 8 WebUI Console terpadu ala llama.cpp (1 halaman, capability-gated) | Header (status, model select + badges, register/unload, key); timeline teks-SSE + image + slideshow video sehalaman; tombol Text/Image/Video aktif sesuai capabilities model; `-m/--model`, `--api-key` ala llama-server; BMP preview + `/files/` | page.tsx, server.cpp, engines, config/main, task.md F8 |
| 2026-09-05 | CATATAN JUJUR: media image/video = preview prosedural, difusi penuh pending | generate() isi RGBA deterministik (seed+prompt) + tulis BMP; UI beri badge "preview placeholder". Alur job/polling/display 100% real; yang belum real hanya komputasi difusi (UNet/VAE/temporal) | preview_media.hpp, BUG-024 |
| 2026-09-05 | Console v2 ala WebUI asli (tools/ui dipelajari dari GitHub) + auth default-off + UTF-8 fix | Sidebar sesi (multi-session localStorage) + model, pesan avatar + tok/s + copy + regen, composer chips Text/Image/Video capability-gated, settings slide-over, light default; key opsional; BUG-026/027 | page.tsx, server.cpp, text_engine.cpp, task.md F8-5 s/d F8-7 |
| 2026-09-05 | Fase 9: WebUI ASLI llama.cpp (bukan tiruan) di-embed ke binary, Next.js dihapus | Build tools/ui dari folder studi user; shim /props + /models/load-unload + entri /v1/models upstream (additive); model default bila 1 model; patch Svelte minimal (mode bar + generate service, markdown inline); embed via script python → serve memori; BUG-028 | llama.cpp/tools/ui (BarskuyModeBar, barskuy-generate.service), server.cpp/hpp, scripts/embed_webui.py, src/CMakeLists, task.md F9 |
| 2026-09-05 | CATATAN JUJUR: UI upstream dipakai apa adanya; adaptasi hanya mode generate | Chat/model/settings/tools/MCP-UI = kode upstream murni (fork di folder studi, bukan milik kita). Yang kita tambah: chips Text/Image/Video + generate image/video + capabilities. Maintenance: rebuild UI → regen embed → rebuild binary | dist-barskuy, webui_embed_gen.cpp |
| 2026-09-05 | F9-6: chips mode masuk composer + log mode auth (tindak lanjut screenshot user) | Chips Text/Image/Video pindah dari atas form ke dalam composer samping tombol + (fallback modelsStore.activeModelId); startup log Auth OPEN/REQUIRED; dialog Model Information kosong = efek 401 DB lama (BUG-029), bukan bug UI | BarskuyModeBar.svelte, ChatFormActions.svelte, server.cpp, task.md F9-6 |
| 2026-09-05 | F9-7: swap model dari dialog tanpa restart | Dropdown di baris Model (desktop+mobile) ganti tombol salin; unload lama + load baru via /models/*; server simpan active-model (/models/load set) yang diikuti /props, urutan /v1/models, dan chat default. E2E: qwen05→qwen4b, props+order+chat ikut | DialogModelInformation.svelte, server.cpp/hpp, task.md F9-7 |
| 2026-09-05 | F9-9: Settings/tools/MCP/think/defaults/flags ala llama-server | /tools array + plain_text_response + SSE + get_info (BUG-030); passthrough tool_calls vs agent:true (BUG-031); DRY+mirostat sampler; defaults temp 0.6/top_k 20/top_p 0.9/min_p 0.05/repeat 1.1/n-predict -1→8192; flags --jinja/--perf/--no-context-shift/--no-warmup(no-op)/--n-cpu-moe/--flash-attn/--batch-size/--ubatch-size/--cache-type-k-v(auto)/--no-mmap default/--mmap/--prio/--np; think tags diteruskan (collapse client-side); 21/21 | agent_*.cpp, agent_loop.*, server.cpp, text_engine.*, config/main, task.md F9-9 |
| 2026-09-05 | CATATAN JUJUR F9-9: yang belum/tidak dipetakan | dry/mirostat eksotik lain (typ-p/xtc) tak ada; --flash-attn bundle-managed (flag dicatat+log saja); MCP UI = browser-side (dialog + menu), MCP server = backend mcp.json (keduanya jalan, sistem berbeda); tool_calls model kecil dinormalisasi fallback parser; think kosong (model tak bernalar) = tak ada yang di-collapse | task.md |
| 2026-09-05 | F9-10: output rapi + streaming sejati + tok/s sinkron + ctx/dialog (tindak lanjut screenshot) | Utf8Streamer per-token (emoji/aksen utuh, nol ????); clean echoed header + blank; think→reasoning_content live (collapsible UI) + non-stream; worker-thread chunked SSE (kirim per token, bukan tunggu selesai); timings prompt/decode di SSE final + JSON + log [perf] dari sumber ukur yang sama; -ctk/--context-size default 8192 (props + dialog ikut); dialog seksi Barskuy (capabilities, quant, vocab/layers/heads, ctx train→efektif, backend/GPU, tools/MCP); BUG-032 | text_engine.*, server.cpp, config/main, DialogModelInformation.svelte, task.md F9-10 |
| 2026-09-05 | CATATAN JUJUR F9-10: Vulkan OFF di build ini (CPU saja) | BARSKUY_BUILD_VULKAN=OFF → has_gpu=false, semua inferensi CPU (~7-8 tok/s decode 4B). Klaim 36 tok/s lama dari build berbeda. GPU/CUDA tetap ranah Blocked; dialog Backend jujur tampilkan CPU | build/CMakeCache, task.md |
| 2026-09-05 | F9-11: console-error cleanup dari log browser user |
| 2026-09-05 | F9-12: UI server-tools only + streams array + ikon (log browser) |
| 2026-09-05 | F9-13: single-instance lock (akar kekacauan log browser) |
| 2026-09-05 | F9-14: markup tool-call bocor (screenshot turn-2 kosong) |
| 2026-09-05 | F9-15: turn-2 agentic kosong (screenshot: pesan-1 OK, pesan-2 kosong) |
| 2026-09-05 | F9-17: observabilitas agentic (loop dilaporkan stuck) |
| 2026-09-05 | F9-18: model cerewet tools + temp-0 Qwen3 | file_glob 5x/sapaan (log user) = perilaku model, bukan hang; guardrail deskripsi per tool; sapaan kini 0 calls. temp=0 kosong di Qwen3-4B = greedy argmax EOG model itu (qwen05 temp=0 normal, chain benar); /props selaras defaults | agent_tools.cpp, server.cpp, task.md F9-18 | Simulasi loop penuh via API sehat; tambah log agent-step + tool exec agar terlihat di terminal persis seperti di UI. Auth dua sisi diaudit bersih (UI kirim Bearer hanya bila key diisi; server buka bila DB tanpa key). Tunggu log user: urutan agent-step/tool exec + chat start yang berhenti | server.cpp, task.md F9-17 | Reproduksi: histori tool (content null) → 400 INVALID_JSON instan. Fix parsing toleran. E2E: tanggal benar Jumat 5 Sep 2026 + reasoning. Fix murni backend (tanpa rebuild UI) | server.cpp, task.md F9-15 | Reproduksi: turn-2 Qwen3 kembalikan content mentah berisi echo+markup. Fix shared cleaner + null-content upstream-exact. E2E: tool_calls bersih. Sisa emoji ?? pada Ling = keterbatasan vocab model (sama di llama-server), bukan bug kita | text_engine.hpp/cpp, server.cpp, task.md F9-14 | Semua ERR_CONNECTION_REFUSED = server mati/tak di port itu; dua instance bisa double-bind di Windows. Lock file eksklusif per port + FATAL + exit 1; A serve, B ditolak. Bila UI termuat tapi API refused: bunuh proses lama (Task Manager) lalu start satu instance | main.cpp, server.hpp/cpp, task.md F9-13 | Browser group dibuang dari allTools (panel+request satu seam); streams/lookup [] array; ikon recommended-mcp di-embed; MCP browser (exa/context7 URL) tetap opt-in via dialog, health-check 405-nya perilaku upstream untuk server user-tambahkan (bukan bug kita) | tools.svelte.ts, server.cpp, embed_webui.py, task.md F9-12 | /v1/streams/lookup -> {} (tadinya 404 tiap buka chat); sw.js no-op + favicon.svg embed (tadinya 404); exec_shell timeout anti-hang agentic (BUG-034); 21/21 | server.cpp, embed_webui.py, agent_tools.cpp, task.md F9-11 |
| 2026-09-05 | Build Vulkan ON (GGML_VULKAN + link ggml-vulkan), CUDA tetap off | Root cause CPU-only: GGML_VULKAN=OFF di cache subdir + binary tak link ggml-vulkan. Fix: -DGGML_VULKAN=ON + target_link_libraries ggml-vulkan (src/CMakeLists). E2E: 2 device (Intel iGPU + RTX 3050), offload 99 layers, decode 179.6 tok/s (vs ~8 CPU). iGPU aktif menyusul; CUDA tetap skip | src/CMakeLists, build, task.md |
| 2026-09-05 | F9-8: Settings modal difungsikan (sampling parity) + audit | top_k/min_p/repeat_last_n/repeat_penalty/presence/frequency/seed di-parse dan masuk sampler chain (urutan ala llama.cpp); E2E seed deterministik (same→identik, beda→beda). Sudah fungsi sebelumnya: system msg, stop, tools→agent, json_mode, title-LLM, import/export, MCP-UI client-side. Belum: dry/mirostat/xtc eksotik, tampilan step agent client-side (eksekusi server-side via agent_steps) | text_engine.cpp/hpp, server.cpp, task.md F9-8 |

## 6. Referensi Silang

- Filosofi dan komponen sistem → [arsitektur.md](./arsitektur.md)
- Pilihan teknologi → [stack.md](./stack.md)
- Kontrak API dan skema data → [desain.md](./desain.md)
- Implementasi backend → [backend.md](./backend.md)
- Implementasi frontend → [frontend.md](./frontend.md)
