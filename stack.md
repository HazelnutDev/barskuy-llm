# Stack — barskuy-llm

> Bagian dari dokumen PRD barskuy-llm. Baca [arsitektur.md](./arsitektur.md) dulu untuk konteks filosofi di balik pilihan-pilihan di bawah ini.

## 1. Prinsip Pemilihan Stack

Tiga pertimbangan yang menentukan hampir semua pilihan di file ini:

- **Satu binary itu harga mati.** Setiap dependency yang mengharuskan proses terpisah (Python runtime, database server, message broker) dipertimbangkan ulang dulu sebelum masuk stack. Kalau ada cara native, native yang menang.
- **Windows adalah target utama, bukan target sekunder.** Toolchain harus benar-benar teruji dengan MSVC + CMake, bukan "seharusnya jalan di Windows juga".
- **Reuse kode yang sudah battle-tested untuk bagian yang berisiko tinggi kalau ditulis ulang dari nol** (parsing GGUF, kernel dequantization, graph komputasi) — tapi bangun sendiri untuk bagian yang jadi nilai tambah proyek ini (architecture registry, unifikasi format, job orchestration, endpoint gaya OpenAI).

## 2. Core Engine Layer

| Komponen | Teknologi | Alasan |
|---|---|---|
| Bahasa inti | C++17/20 | Sama dengan llama.cpp, kompatibel langsung dengan ekosistem ggml, kontrol memori penuh untuk kerja tensor |
| Tensor compute | ggml (embedded/fork) | Tensor library yang sama dipakai llama.cpp dan stable-diffusion.cpp — satu substrat untuk teks, gambar, dan video |
| Loader GGUF | Native, berbasis parser GGUF llama.cpp | Format ini memang "rumah" ggml, tidak ada alasan menulis ulang dari nol |
| Loader safetensors | Parser native (header JSON + `config.json`) yang memetakan tensor ke graph ggml | Logika pemetaan nama tensor mengikuti pola yang sama dipakai skrip `convert_hf_to_gguf.py`, tapi dijalankan saat runtime, bukan menghasilkan file GGUF perantara |
| HTTP server | cpp-httplib (header-only) | Library yang sama dipakai `llama-server`, sudah teruji, tidak menambah dependency berat |
| Build system | CMake + MSVC (Windows) / GCC-Clang (Linux, macOS) | Windows-first sesuai target utama, cross-platform tetap dijaga |

## 3. Backend Compute

| Backend | Prioritas | Catatan |
|---|---|---|
| CUDA | Utama | Untuk GPU NVIDIA, termasuk RTX 3050 6GB sebagai target validasi |
| Vulkan | Utama | Cakupan hardware lebih luas, bisa di-build hybrid bersama CUDA dalam satu binary |
| CPU (AVX2/AVX512) | Utama | Fallback wajib jalan, juga dipakai untuk offload MoE expert |
| ROCm | Sekunder | Dukungan AMD, menyusul setelah jalur CUDA/Vulkan stabil |
| Metal | Sekunder | Dukungan Apple Silicon, menyusul setelah jalur utama stabil |

## 4. Multimodal Generation Layer

| Komponen | Teknologi | Catatan |
|---|---|---|
| Image Engine | Native ggml, mengikuti pola stable-diffusion.cpp | UNet/DiT + VAE + text encoder dalam satu graph ggml |
| Video Engine | Ekstensi Image Engine + temporal/motion module | Paling eksperimental — lihat fase 5 di [arsitektur.md](./arsitektur.md#9-roadmap-evolusi) |
| Scheduler diffusion | Implementasi native (DDIM, Euler, DPM++ minimal di v1) | Cukup untuk cakupan model target awal (SD1.5/SDXL) |

## 5. Data & Orchestration Layer

| Komponen | Teknologi | Alasan |
|---|---|---|
| Model registry | SQLite (embedded) | Tidak butuh database server terpisah, cocok untuk deployment single-machine, konsisten dengan pola yang sudah dipakai di proyek lain (BarskuyAI Desktop) |
| Job queue (image/video) | In-process queue + snapshot ke SQLite | Job history tetap ada setelah restart tanpa menambah dependency Redis |
| Penyimpanan hasil generate | Filesystem lokal, terstruktur per job id | Sederhana untuk self-hosted; opsi object storage (MinIO/S3-compatible) jadi target lanjutan kalau butuh multi-node |
| Konfigurasi | File TOML/YAML + override lewat environment variable | Mudah diedit manual, cocok untuk workflow launcher PowerShell yang sudah biasa dipakai |

## 6. Frontend Layer

| Komponen | Teknologi | Alasan |
|---|---|---|
| Framework | Next.js + React + TypeScript | Ekosistem matang, mudah dapat komponen siap pakai untuk dashboard dan playground |
| Styling | Tailwind CSS | Cepat untuk membangun UI konsisten tanpa banyak file CSS terpisah |
| State/data fetching | TanStack Query | Cocok untuk pola polling job status (image/video) dan streaming (teks) |
| Deployment | Bisa jalan sebagai dev server terpisah, atau di-build jadi static assets yang di-embed/disajikan langsung oleh binary core (opsional) | Fleksibel — dev pakai server terpisah, produksi bisa satu paket kalau mau benar-benar satu executable |

Detail tiap halaman ada di [frontend.md](./frontend.md).

## 7. Matriks Kuantisasi

### 7.1 GGUF (jalur llama.cpp / ggml)

| Tipe | Bit/bobot (approx) | Kategori | Keterangan |
|---|---|---|---|
| F32 | 32 | Full precision | Tidak terkompresi |
| F16 / BF16 | 16 | Half precision | Baseline sebelum kuantisasi |
| Q8_0 | 8.5 | Legacy | Akurasi tinggi, kompresi ringan |
| Q6_K | 6.6 | K-quant | Akurasi tinggi |
| Q5_K_S / Q5_K_M | 5.5 / 5.7 | K-quant | Titik seimbang atas |
| Q5_0 / Q5_1 | 5.5 / 6.0 | Legacy | Pendahulu K-quant |
| Q4_K_S / Q4_K_M | 4.6 / 4.8 | K-quant | Titik seimbang paling umum dipakai |
| Q4_0 / Q4_1 | 4.5 / 5.0 | Legacy | Masih relevan untuk kompatibilitas lama |
| Q3_K_S / Q3_K_M / Q3_K_L | 3.5 / 3.9 / 4.3 | K-quant | Kompresi tinggi |
| Q2_K | 2.6 | K-quant | Kompresi sangat tinggi, akurasi turun signifikan |
| IQ4_XS / IQ4_NL | ~4.25 | I-quant (importance matrix) | Lebih presisi dari legacy 4-bit di ukuran serupa |
| IQ3_XXS / IQ3_S / IQ3_M | ~3.0–3.7 | I-quant | Kompresi ekstra tinggi |
| IQ2_XXS / IQ2_XS / IQ2_S / IQ2_M | ~2.0–2.7 | I-quant | Ekstrem kecil, untuk model sangat besar di VRAM terbatas |
| IQ1_S / IQ1_M | ~1.5–1.75 | I-quant | Eksperimental, degradasi kualitas signifikan |

Termasuk cakupan untuk tipe tensor kustom di luar daftar mainline (contoh kasus nyata: `PQ2_0` ternary milik fork PrismML) — ini yang ditangani lewat Architecture Registry, bukan hardcode di core engine. Lihat [desain.md](./desain.md#pola-arsitektur-plugin).

### 7.2 Safetensors (jalur vLLM-style)

| Tipe | Kategori | Keterangan |
|---|---|---|
| FP16 / BF16 | Unquantized | Baseline, tanpa kompresi |
| AWQ (4-bit) | Post-training quant | Activation-aware, populer di ekosistem vLLM |
| GPTQ (4-bit / 3-bit) | Post-training quant | Berbasis Hessian, salah satu format paling umum di HuggingFace |
| FP8 (E4M3 / E5M2) | Native hardware quant | Butuh dukungan hardware FP8 (arsitektur GPU generasi baru) |
| BitsAndBytes NF4 / INT8 | On-the-fly quant | Dikuantisasi saat model dimuat, bukan disimpan terkuantisasi di file |

## 8. Fitur Wajib dari llama.cpp yang Harus Ada

- Loading GGUF dengan seluruh tipe kuantisasi di tabel 7.1
- Mmap-based model loading (load cepat, overhead memori rendah)
- Grammar-constrained generation (setara GBNF)
- Speculative decoding dengan draft model
- Endpoint embeddings
- Dukungan multimodal vision (setara LLaVA/mtmd)
- Mode server dengan endpoint gaya OpenAI (sudah jadi fondasi desain barskuy-llm)
- LoRA adapter loading
- Hot-swap antar model tanpa restart proses (setara llama-swap)

## 9. Fitur Wajib dari vLLM yang Harus Ada

- Continuous batching (skala disesuaikan — lihat [arsitektur.md](./arsitektur.md#8-continuous-batching--paged-kv-cache--cakupan-yang-realistis))
- Paged KV cache (versi native, skala single/multi-GPU lokal)
- Tensor parallelism untuk multi-GPU
- Prefix caching (reuse KV cache untuk prompt dengan prefix sama)
- Chunked prefill
- Guided decoding / structured output (JSON mode, grammar constraint)
- LoRA adapter serving dengan banyak adapter aktif sekaligus di satu base model
- Dukungan model vision-language

## 10. Hardware & Runtime Target

| Target | Status | Catatan |
|---|---|---|
| Windows + NVIDIA (CUDA) | Primer, target validasi utama | RTX 3050 Laptop 6GB sebagai baseline pengujian |
| Windows + Vulkan | Primer | Cakupan GPU non-NVIDIA di Windows |
| Linux + CUDA | Primer | Target deployment server |
| Linux + ROCm | Sekunder | AMD di Linux |
| macOS + Metal | Sekunder | Apple Silicon |
| CPU-only (semua OS) | Wajib jalan sebagai fallback | Lambat tapi harus tetap fungsional |

## 11. Referensi Silang

- Alasan filosofis di balik struktur ini → [arsitektur.md](./arsitektur.md)
- Kontrak API dan skema data → [desain.md](./desain.md)
- Implementasi detail tiap komponen → [backend.md](./backend.md)
- Implementasi UI → [frontend.md](./frontend.md)
- Breakdown task per fase → [task.md](./task.md)
