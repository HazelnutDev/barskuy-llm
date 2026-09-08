# Desain — barskuy-llm

> Bagian dari dokumen PRD barskuy-llm. Baca [arsitektur.md](./arsitektur.md) untuk konteks komponen, dan [stack.md](./stack.md) untuk pilihan teknologi.

## 1. Prinsip Desain

- **Kontrak dulu, implementasi belakangan.** Backend dan frontend sama-sama mengacu ke kontrak di file ini — kalau ada perubahan bentuk request/response, file ini yang diedit lebih dulu, baru kode menyusul.
- **Konsisten dengan konvensi OpenAI di level bentuk data**, walau path endpoint mengikuti penamaan kustom yang diminta (`v1/complete`, `v1/generate/image`, `v1/generate/video`). Field seperti `id`, `object`, `created`, `model`, `usage` mengikuti pola yang sudah familiar di ekosistem itu supaya tooling yang sudah ada (LangChain, SDK resmi, dst) tidak butuh banyak penyesuaian.
- **Job-based untuk semua yang non-instan.** Generate gambar dan video selalu melalui pola job, tidak ada jalur "kadang sync kadang async" yang membingungkan client.
- **Gagal harus jelas.** Error response selalu menyertakan alasan yang bisa ditindaklanjuti (contoh: "arsitektur belum terdaftar" lebih berguna daripada "invalid tensor").

## 2. Catatan Penting Soal Kompatibilitas OpenAI

Ada satu hal yang perlu diluruskan di sini karena mempengaruhi implementasi: endpoint resmi OpenAI untuk teks itu `/v1/chat/completions` (gaya chat) dan `/v1/completions` (gaya prompt lama), bukan `/v1/complete` — path itu sebenarnya lebih dekat ke penamaan API lama Anthropic. Karena permintaan eksplisit memakai `v1/complete`, path itu tetap jadi endpoint utama dan didokumentasikan sebagai demikian, tapi supaya SDK resmi OpenAI (`openai` Python/JS) bisa langsung dipakai tanpa modifikasi base URL yang aneh-aneh, dua alias berikut wajib ada dan mengarah ke handler yang sama:

- `/v1/chat/completions` → alias `v1/complete`
- `/v1/completions` → alias `v1/complete`, dengan konversi bentuk body prompt-string ke format messages secara internal

Untuk gambar, OpenAI aslinya memakai `/v1/images/generations`. Karena diminta path kustom `/v1/generate/image`, path itu yang jadi utama, dengan `/v1/images/generations` sebagai alias opsional di fase belakangan kalau kompatibilitas SDK gambar juga dibutuhkan. Video generation belum punya standar path yang mapan di ekosistem OpenAI (fitur ini masih baru di seluruh industri), jadi `v1/generate/video` sepenuhnya jadi desain sendiri tanpa perlu alias.

## 3. Kontrak API

### 3.1 `GET /v1/models`

Daftar semua model yang terdaftar di registry, bukan hanya yang sedang dimuat ke memori.

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-9b-q4_k_m",
      "object": "model",
      "created": 1735689600,
      "format": "gguf",
      "quantization": "Q4_K_M",
      "capabilities": ["text", "embeddings"],
      "loaded": true
    }
  ]
}
```

### 3.2 `POST /v1/complete`

```json
{
  "model": "qwen3.5-9b-q4_k_m",
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true,
  "stop": ["</s>"],
  "response_format": { "type": "text" }
}
```

Respons non-streaming:

```json
{
  "id": "cmpl_a1b2c3",
  "object": "text_completion",
  "created": 1735689600,
  "model": "qwen3.5-9b-q4_k_m",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 24, "completion_tokens": 118, "total_tokens": 142 }
}
```

Respons streaming: SSE dengan chunk bertipe `text_completion.chunk`, diakhiri baris `data: [DONE]` — mengikuti pola yang sama seperti `/v1/chat/completions` OpenAI supaya parser SSE yang sudah ada di banyak library tetap kompatibel.

### 3.3 `POST /v1/embeddings`

```json
{ "model": "bge-m3", "input": ["teks pertama", "teks kedua"] }
```

```json
{
  "object": "list",
  "data": [{ "object": "embedding", "index": 0, "embedding": [0.012, -0.034, "..."] }],
  "model": "bge-m3",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 }
}
```

### 3.4 `POST /v1/generate/image`

```json
{
  "model": "sdxl-base",
  "prompt": "...",
  "negative_prompt": "...",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidance_scale": 7.0,
  "seed": 42,
  "n": 1,
  "response_format": "url",
  "wait": false
}
```

Respons langsung (job dibuat):

```json
{ "id": "img_x1y2z3", "object": "image.generation.job", "status": "queued", "created": 1735689600 }
```

### 3.5 `GET /v1/generate/image/{job_id}`

```json
{
  "id": "img_x1y2z3",
  "object": "image.generation.job",
  "status": "completed",
  "progress": 100,
  "result": { "data": [{ "url": "/files/img_x1y2z3_0.png" }] },
  "error": null
}
```

`status` bernilai salah satu dari: `queued`, `processing`, `completed`, `failed`.

### 3.6 `POST /v1/generate/video`

```json
{
  "model": "video-diffusion-base",
  "prompt": "...",
  "negative_prompt": "...",
  "num_frames": 48,
  "fps": 24,
  "width": 576,
  "height": 320,
  "seed": 42,
  "guidance_scale": 7.0
}
```

Respons dan pola polling sama seperti image (`object: "video.generation.job"`), lewat `GET /v1/generate/video/{job_id}`. Karena video jauh lebih berat, response job menyertakan estimasi `eta_seconds` kalau engine bisa menghitungnya dari antrian dan resolusi/durasi yang diminta.

### 3.7 `GET /v1/jobs`

Daftar semua job (image + video) dengan filter opsional `status` dan `type` — dipakai dashboard job monitor di frontend.

### 3.8 `POST /v1/models/register`

Mendaftarkan file model baru ke registry (memicu alur deteksi format di [arsitektur.md](./arsitektur.md#6-model-loading--format-detection)).

```json
{ "path": "D:/models/qwen3.5-9b-q4_k_m.gguf", "alias": "qwen3.5-9b-q4_k_m" }
```

### 3.9 Autentikasi

Semua endpoint (kecuali healthcheck) mewajibkan header `Authorization: Bearer <api_key>`. API key dikelola lewat tabel `api_keys` di SQLite (lihat bagian 4), dengan endpoint admin terpisah untuk create/revoke.

## 4. Model Data

| Entitas | Field kunci | Keterangan |
|---|---|---|
| `Model` | `id`, `path`, `format` (gguf/safetensors), `architecture`, `quantization`, `capabilities[]`, `loaded`, `created_at` | Satu baris per file model terdaftar |
| `GenerationJob` | `id`, `type` (image/video), `model_id`, `status`, `progress`, `request_params` (JSON), `result_path`, `error`, `created_at`, `completed_at` | Satu baris per job async |
| `ApiKey` | `id`, `key_hash`, `label`, `created_at`, `revoked_at` | Key disimpan sebagai hash, bukan plaintext |
| `UsageLog` | `id`, `api_key_id`, `endpoint`, `model_id`, `tokens_or_units`, `timestamp` | Untuk observability dan rate limiting dasar |
| `ArchitectureDefinition` | `id`, `name`, `tensor_mapping` (JSON), `graph_template_ref` | Metadata arsitektur yang terdaftar di Architecture Registry — lihat bagian 6 |

## 5. Skema Database (SQLite)

```sql
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    format TEXT NOT NULL CHECK(format IN ('gguf','safetensors')),
    architecture TEXT NOT NULL,
    quantization TEXT,
    capabilities TEXT NOT NULL, -- JSON array
    loaded INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL
);

CREATE TABLE generation_jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('image','video')),
    model_id TEXT REFERENCES models(id),
    status TEXT NOT NULL CHECK(status IN ('queued','processing','completed','failed')),
    progress INTEGER DEFAULT 0,
    request_params TEXT NOT NULL, -- JSON
    result_path TEXT,
    error TEXT,
    created_at INTEGER NOT NULL,
    completed_at INTEGER
);

CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    label TEXT,
    created_at INTEGER NOT NULL,
    revoked_at INTEGER
);

CREATE TABLE usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id TEXT REFERENCES api_keys(id),
    endpoint TEXT NOT NULL,
    model_id TEXT,
    tokens_or_units INTEGER,
    timestamp INTEGER NOT NULL
);

CREATE TABLE architecture_definitions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    tensor_mapping TEXT NOT NULL, -- JSON
    graph_template_ref TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
```

## 6. Pola Arsitektur Plugin

Ini jawaban desain untuk masalah "arsitektur baru belum didukung" yang berulang kali jadi hambatan. Setiap arsitektur didefinisikan sebagai file deklaratif (JSON/TOML), bukan cabang if-else baru di kode inti engine:

```json
{
  "name": "nanbeige4.2",
  "tensor_mapping": {
    "token_embd": "model.embed_tokens.weight",
    "layers.{n}.attn_q": "model.layers.{n}.self_attn.q_proj.weight",
    "layers.{n}.attn_k": "model.layers.{n}.self_attn.k_proj.weight",
    "layers.{n}.ffn_gate": "model.layers.{n}.mlp.gate_proj.weight"
  },
  "attention_type": "grouped_query",
  "moe": { "enabled": false },
  "graph_template": "standard_decoder_only"
}
```

Untuk arsitektur dengan skema tensor yang benar-benar baru (bukan sekadar penamaan berbeda dari template yang ada), `graph_template` bisa menunjuk ke template kustom yang ditulis terpisah. Ini yang membedakan pendekatan ini dari sekadar "mapping nama tensor" — kasus seperti tipe kuantisasi ternary kustom (`PQ2_0`) butuh definisi dequantization kernel baru, bukan cuma pemetaan nama, dan itu didaftarkan sebagai unit terpisah yang direferensikan dari `ArchitectureDefinition`.

## 7. Desain UX Kunci

Detail komponen dan implementasi ada di [frontend.md](./frontend.md); di sini fokus ke alur, bukan kode.

- **Playground** — satu halaman chat dengan pemilih model, slider parameter (temperature, top_p, max_tokens), dan tampilan streaming token real-time. Prioritas: terasa instan, tidak ada loading state yang menghalangi ketikan berikutnya.
- **Studio Gambar/Video** — form prompt + parameter di kiri, hasil (grid untuk gambar, player untuk video) di kanan. Job yang masih `processing` tampil dengan progress bar, bukan spinner generik, supaya user tahu ini bukan hang.
- **Model Manager** — tabel semua model terdaftar dengan kolom format, kuantisasi, status loaded/unloaded, dan tombol register model baru dari path lokal.
- **Job Monitor** — daftar job image/video lintas waktu dengan filter status, dipakai untuk debug ketika job gagal (menampilkan field `error` langsung, bukan disembunyikan).
- **API Keys** — halaman sederhana create/revoke key, menampilkan key penuh hanya sekali saat dibuat (pola standar, bukan disimpan plaintext yang bisa dilihat ulang).

## 8. Pola Desain Arsitektural

- **Adapter pattern untuk compute backend** — Text/Image/Video Engine tidak tahu apakah mereka jalan di CUDA, Vulkan, atau CPU; itu diselesaikan di layer ggml backend selection saat startup berdasarkan hardware yang terdeteksi (dengan override manual lewat config).
- **Plugin pattern untuk arsitektur model** — dijelaskan di bagian 6.
- **Job pattern seragam untuk semua generasi non-instan** — image dan video pakai state machine yang identik (`queued → processing → completed/failed`), supaya Job Queue di backend dan Job Monitor di frontend cukup ditulis sekali untuk dua jenis job.
- **Registry pattern untuk model** — satu sumber kebenaran (SQLite) untuk metadata model, dibaca baik oleh Router (untuk routing request) maupun frontend (untuk Model Manager), supaya tidak ada duplikasi state.

## 9. Referensi Silang

- Konteks komponen dan filosofi → [arsitektur.md](./arsitektur.md)
- Pilihan teknologi → [stack.md](./stack.md)
- Implementasi tiap service → [backend.md](./backend.md)
- Implementasi UI → [frontend.md](./frontend.md)
- Task dan bug tracking → [task.md](./task.md)
