# Frontend — barskuy-llm

> Bagian dari dokumen PRD barskuy-llm. Mengacu ke kontrak di [desain.md](./desain.md) dan endpoint di [backend.md](./backend.md).

## 1. Ringkasan

Frontend adalah dashboard + playground yang mengonsumsi API barskuy-llm lewat HTTP biasa — tidak ada logika inferensi apapun di sisi ini, murni presentasi dan orkestrasi request. Dipisah dari binary core supaya siklus development UI tidak terikat ke waktu compile C++, tapi tetap bisa dipaketkan jadi static assets yang disajikan langsung oleh core engine untuk deployment yang benar-benar satu paket.

## 2. Tech Stack Frontend

Sudah dirangkum di [stack.md §6](./stack.md#6-frontend-layer): Next.js + React + TypeScript, Tailwind CSS untuk styling, TanStack Query untuk data fetching dan polling job status.

## 3. Halaman Playground

- Pemilih model (dropdown, sumber data dari `GET /v1/models`, hanya menampilkan model dengan capability `text`).
- Area chat dengan riwayat percakapan tersimpan di state lokal (tidak perlu persist ke server untuk v1).
- Panel parameter yang bisa di-collapse: temperature, top_p, max_tokens, stop sequences.
- Streaming token ditampilkan langsung saat SSE chunk masuk, dengan indikator "sedang generate" yang hilang otomatis saat `[DONE]` diterima.
- Tombol stop generation yang memutus koneksi SSE dari sisi client — ini yang memicu pembatalan di backend (lihat [backend.md §2](./backend.md#2-api-gateway-dalam-binary)).

## 4. Halaman Studio Gambar/Video

- Tab terpisah untuk Image dan Video, form parameter di kiri (prompt, negative prompt, dimensi, steps/frame count sesuai tipe), preview hasil di kanan.
- Job yang `queued`/`processing` menampilkan progress bar berbasis field `progress` dari polling `GET /v1/generate/{image|video}/{job_id}` — interval polling dimulai dari 1 detik, exponential backoff sampai maksimum 5 detik untuk job yang lama.
- Galeri hasil menyimpan riwayat generate dalam sesi (grid thumbnail), klik untuk buka ukuran penuh atau player video.
- Job yang `failed` menampilkan pesan `error` dari server apa adanya, dengan tombol "coba lagi" yang mengirim ulang request dengan parameter yang sama.

## 5. Halaman Model Manager

- Tabel semua model dari `GET /v1/models`: kolom nama, format, kuantisasi, kapabilitas, status loaded/unloaded.
- Form "Register Model Baru" yang memanggil `POST /v1/models/register` dengan path file lokal.
- Indikator visual kalau ada model yang gagal register karena arsitektur belum dikenali (mengacu ke alur di [arsitektur.md §6](./arsitektur.md#6-model-loading--format-detection)) — pesan error mengarahkan ke dokumentasi cara menambah `ArchitectureDefinition` baru.

## 6. Halaman Job Monitor

- Daftar semua job (`GET /v1/jobs`) lintas image dan video, dengan filter status dan tipe.
- Dipakai terutama untuk debugging — menampilkan `request_params` penuh dan `error` untuk tiap job yang gagal, supaya penyebab bisa dilacak tanpa harus buka log server manual.

## 7. Halaman API Keys & Settings

- Create/revoke API key, dengan key penuh hanya ditampilkan sekali saat pertama dibuat.
- Ringkasan pemakaian per key (dari `usage_logs`) — jumlah request dan token per endpoint dalam rentang waktu yang bisa dipilih.
- Pengaturan runtime dasar yang aman diubah lewat UI (batas worker job queue, default parameter generation) — pengaturan yang lebih sensitif (path model, backend compute) tetap lewat file konfigurasi, bukan UI, supaya tidak ada perubahan tidak sengaja yang butuh restart proses.

## 8. State Management & Data Fetching

- TanStack Query untuk semua request ke API: cache otomatis untuk `GET /v1/models`, invalidasi setelah register model baru.
- Polling job status memakai `refetchInterval` dinamis (lihat bagian 4), berhenti otomatis begitu status `completed` atau `failed`.
- Streaming SSE untuk Playground ditangani terpisah dari TanStack Query (pakai `EventSource` langsung atau `fetch` dengan reader stream), karena pola SSE tidak cocok dengan model request-response biasa yang diasumsikan query library.
- Tidak ada state global yang kompleks (Redux dst) — kebutuhan v1 cukup dengan state lokal per halaman plus cache dari data fetching layer.

## 9. Referensi Silang

- Kontrak API yang dikonsumsi tiap halaman → [desain.md](./desain.md)
- Detail service backend → [backend.md](./backend.md)
- Konteks komponen sistem → [arsitektur.md](./arsitektur.md)
- Pilihan teknologi → [stack.md](./stack.md)
- Breakdown task per fase → [task.md](./task.md)
