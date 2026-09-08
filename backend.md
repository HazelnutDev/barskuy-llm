# Backend — barskuy-llm

> Bagian dari dokumen PRD barskuy-llm. Mengacu ke kontrak di [desain.md](./desain.md) dan komponen di [arsitektur.md](./arsitektur.md).

## 1. Ringkasan

Semua yang dibahas di file ini hidup dalam satu proses C++. Pembagian per "service" di bawah adalah pembagian modul dalam source tree, bukan pembagian proses/container — penting untuk terus diingat supaya tidak ada godaan menambah komunikasi jaringan internal antar modul yang sebenarnya bisa jadi pemanggilan fungsi biasa.

## 2. API Gateway (dalam binary)

- Routing berbasis path + method ke handler yang sesuai, divalidasi terhadap skema di [desain.md](./desain.md#3-kontrak-api) sebelum diteruskan.
- Autentikasi: setiap request dicek terhadap tabel `api_keys` (hash dibandingkan, bukan plaintext). Key yang `revoked_at` terisi langsung ditolak.
- Rate limiting dasar per API key (jumlah request/menit), disimpan sebagai counter in-memory dengan window sliding sederhana — cukup untuk single-node, upgrade ke token bucket terdistribusi kalau nanti multi-node relevan.
- Streaming SSE untuk `/v1/complete` dengan `stream: true` — koneksi ditahan terbuka, dikirim event per token, connection close memicu pembatalan generation di Scheduler (supaya tidak ada compute yang terbuang untuk client yang sudah putus).
- Semua error mengikuti bentuk konsisten: `{ "error": { "type": "...", "message": "...", "code": "..." } }`.

## 3. Text Engine

### 3.1 Loading GGUF
Native lewat ggml — seluruh tipe di matriks kuantisasi [stack.md §7.1](./stack.md#71-gguf-jalur-llamacpp--ggml) harus lolos test loading. Mmap dipakai secara default supaya waktu load tidak proporsional dengan ukuran file.

### 3.2 Loading Safetensors
1. Baca header JSON (8 byte length prefix + JSON metadata) untuk dapat nama tensor, shape, dtype, offset.
2. Baca `config.json` di direktori yang sama untuk dapat hyperparameter arsitektur (jumlah layer, hidden size, jumlah head, dst).
3. Cocokkan nama arsitektur (`config.json` → field `architectures`) ke Architecture Registry.
4. Petakan tiap tensor safetensors ke slot ggml sesuai `tensor_mapping` dari definisi arsitektur.
5. Jalankan dequantization kernel sesuai tipe yang terdeteksi (AWQ/GPTQ/FP8/NF4) kalau bukan FP16/BF16 mentah.

### 3.3 Continuous Batching
Scheduler menjaga daftar sequence aktif, tiap step forward pass memproses satu batch berisi campuran sequence pada tahap berbeda (sebagian masih prefill, sebagian sudah decode) — sequence baru bisa masuk batch di step manapun tanpa menunggu batch sebelumnya selesai total. Ukuran batch maksimum dibatasi oleh VRAM yang tersedia, dihitung dari ukuran KV cache per sequence dikali context length.

### 3.4 Paged KV Cache
KV cache dialokasikan per blok (ukuran blok tetap, misal 16 atau 32 token per blok) bukan per buffer kontinu penuh `context_length`. Sequence yang lebih pendek dari context maksimum tidak mengunci memori untuk panjang penuh yang tidak terpakai. Blok yang sudah tidak dipakai (sequence selesai) dikembalikan ke pool untuk sequence berikutnya.

### 3.5 Prefix Caching
Hash prefix dari token-token awal prompt; kalau ada sequence baru dengan prefix yang identik dengan yang sudah pernah dihitung (skenario umum: system prompt yang sama dipakai berulang), blok KV cache untuk prefix itu di-reuse langsung tanpa forward pass ulang.

### 3.6 Speculative Decoding
Draft model kecil menghasilkan beberapa token kandidat sekaligus, model utama memverifikasi dalam satu forward pass — mempercepat throughput generation tanpa mengubah distribusi output. Konfigurasi draft model per model utama disimpan di Model Registry sebagai pasangan opsional.

### 3.7 Guided Decoding
Constraint berbasis grammar (setara GBNF) atau JSON schema diterapkan dengan memfilter logit token yang tidak valid pada tiap step sebelum sampling — memastikan output selalu mengikuti struktur yang diminta di `response_format`.

### 3.8 LoRA Adapter Serving
Multiple adapter LoRA bisa dimuat bersamaan di atas satu base model, dipilih per-request lewat field `model` yang menunjuk ke kombinasi base+adapter terdaftar (contoh: `qwen3.5-9b-base+adapter-coding`). Penerapan delta weight dilakukan saat forward pass tanpa menggandakan base model di memori untuk tiap adapter.

### 3.9 Hot-Swap Model
Saat Router menerima request untuk model yang belum dimuat dan VRAM tidak cukup untuk memuatnya berdampingan dengan model yang sedang aktif, model yang paling lama idle di-unload dulu (LRU) — pola yang sama dengan llama-swap, tapi terintegrasi langsung ke dalam Scheduler alih-alih proxy eksternal.

### 3.10 Multimodal Vision Input
Untuk model vision-language, input gambar di `messages` (base64 atau URL) diproses lewat encoder vision terpisah, hasil embedding-nya disisipkan ke urutan token sebelum masuk ke decoder — mengikuti pola penanganan yang sama seperti `mtmd`/LLaVA di llama.cpp, termasuk penanganan urutan konten multimodal yang benar.

## 4. Image Engine

- Pipeline: text encoder (CLIP/T5 tergantung arsitektur) → diffusion loop (UNet untuk SD1.5/SDXL, DiT untuk arsitektur lebih baru) → VAE decode → output gambar.
- Scheduler diffusion v1: DDIM dan Euler ancestral minimal, DPM++ 2M sebagai target lanjutan.
- VRAM-aware offloading: komponen yang tidak dipakai di step tertentu (misal text encoder setelah embedding dihitung) dipindah ke RAM sistem selama diffusion loop berjalan, dimuat ulang saat dibutuhkan lagi — krusial untuk tetap muat di VRAM 6GB.
- Dukungan LoRA untuk model diffusion (gaya/karakter) sebagai target lanjutan setelah pipeline dasar stabil.

## 5. Video Engine

- Dibangun di atas Image Engine dengan tambahan temporal attention/motion module di antara frame.
- Karena ini bagian paling belum matang di ekosistem ggml, kemungkinan besar sebagian kernel (khususnya temporal attention) perlu ditulis dari nol, bukan port langsung dari implementasi PyTorch yang ada.
- VRAM jadi kendala paling nyata di sini — resolusi dan jumlah frame default di v1 sengaja dibuat konservatif (lihat contoh di [desain.md §3.6](./desain.md#36-post-v1generatevideo)), dengan opsi menaikkan lewat parameter request untuk hardware yang lebih besar.
- Field `eta_seconds` di response job dihitung dari benchmark internal (waktu per frame per resolusi), diperbarui berkala berdasarkan hardware aktual saat startup.

## 6. Architecture Registry Service

- Memuat semua file `ArchitectureDefinition` (lihat [desain.md §6](./desain.md#6-pola-arsitektur-plugin)) saat startup dari direktori `architectures/`.
- Validasi tiap definisi terhadap skema JSON sebelum diterima — definisi yang tidak valid dicatat sebagai warning di log startup, tidak menghentikan seluruh server.
- Expose fungsi pencarian arsitektur berdasarkan nama dari `config.json` (untuk safetensors) atau metadata GGUF (untuk field arsitektur yang sudah tertanam di header GGUF).

## 7. Model Registry Service

- CRUD dasar terhadap tabel `models` di SQLite.
- `POST /v1/models/register` memicu alur deteksi format penuh (lihat [arsitektur.md §6](./arsitektur.md#6-model-loading--format-detection)) sebelum menyimpan baris baru.
- Endpoint unregister/delete tidak menghapus file model, hanya entri registry — mencegah kehilangan file tidak sengaja.

## 8. Job Queue System

- Struktur data: antrian in-memory (per tipe: image, video) + tabel `generation_jobs` di SQLite sebagai source of truth untuk status dan history.
- Worker pool ukurannya dikonfigurasi berdasarkan VRAM yang tersedia — default 1 worker aktif per tipe di hardware dengan VRAM terbatas (mencegah dua diffusion pipeline berebut memori bersamaan), bisa dinaikkan di config untuk hardware lebih besar.
- Job yang gagal disimpan dengan field `error` terisi pesan yang bisa ditindaklanjuti (out of memory, model tidak ditemukan, parameter tidak valid, dst) — tidak ada job yang "hilang" tanpa jejak.
- Job lama (default: lebih dari 30 hari) dibersihkan dari filesystem hasil secara berkala, tapi metadata di SQLite tetap ada untuk audit.

## 9. Observability

- Logging terstruktur (JSON lines) ke file, dengan level konfigurabel (debug/info/warn/error).
- Metrik dasar yang di-expose lewat endpoint `/metrics` bergaya Prometheus: request per endpoint, token/detik untuk text engine, waktu rata-rata per job image/video, penggunaan VRAM saat ini.
- `usage_logs` di SQLite jadi sumber untuk laporan pemakaian per API key, dikonsumsi frontend untuk halaman API Keys.

## 10. Referensi Silang

- Kontrak API dan skema data → [desain.md](./desain.md)
- Komponen dan alur sistem → [arsitektur.md](./arsitektur.md)
- Pilihan teknologi → [stack.md](./stack.md)
- Implementasi UI yang mengonsumsi service-service ini → [frontend.md](./frontend.md)
- Breakdown task per fase → [task.md](./task.md)
