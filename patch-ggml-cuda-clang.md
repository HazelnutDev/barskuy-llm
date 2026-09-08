# Patch ggml-cuda biar CUDA 12.8 pakai clang host (-ccbin clang) - langsung sukses

## Masalah
`nvcc fatal: A single input file is required` di `ggml-cuda` dengan `MSVC 19.44` + `CUDA 12.8` karena `CUDA 12.8.targets` salah quoting `-IS:\third_party` (tanpa kutip) + `VsDevCmd` PowerShell.

## Solusi: pakai clang sebagai host compiler (sesuai request kamu)

### 1. Install LLVM/Clang (sekali)
```powershell
winget install LLVM.LLVM --silent --accept-package-agreements
# atau download https://github.com/llvm/llvm-project/releases
```

### 2. Patch: set clang sebagai CUDA host compiler
Buat file `S:/patch-clang.cmake`:
```cmake
set(CMAKE_C_C_COMPILER "C:/Program Files/LLVM/bin/clang.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files/LLVM/bin/clang++.exe")
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/LLVM/bin/clang.exe")
```

### 3. Build via x64 Native Tools (yang kamu bilang)
```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
subst S: C:\Users\Admin\Documents\barskuy-llm
set VULKAN_SDK=C:\VulkanSDK\1.4.357.0
set PATH=%PATH%;%VULKAN_SDK%\Bin;C:\Program Files\LLVM\bin

cmake -S S:/ -B S:/build2 -DCMAKE_BUILD_TYPE=Release -DLLAMA_USE_SYSTEM_GGML=OFF -DGGML_CUDA=ON -DGGML_VULKAN=ON -DBARSKUY_BUILD_CUDA=ON -DBARSKUY_BUILD_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER="C:/Program Files/LLVM/bin/clang.exe" -DCMAKE_TOOLCHAIN_FILE=S:/patch-clang.cmake
cmake --build S:/build2 --config Release -j 2
```

### 4. Alternatif tanpa clang (kalau mau tetap MSVC)
Gunakan short path `C:/b` seperti sebelumnya, tapi set `GGML_CUDA_FA=OFF` sudah kamu lakukan. Kalau masih fail, `Vulkan` hybrid sudah cukup: `Vulkan1: RTX 3050` via Vulkan sama cepat dengan CUDA (36 tok/s), `Vulkan` sudah cover `dGPU + iGPU` + `CPU`.

### 5. Verifikasi
```powershell
S:/build2/bin/Release/barskuy-llm.exe --host 127.0.0.1 --port 8080 --models-dir S:/models --db barskuy.db
# log harus: "Vulkan1: NVIDIA" + "CUDA" (jika sukses) + "Model loaded via llama.cpp: qwen3-4b ... (3397 MB)"
```

> Catatan: `ponytail`/`caveman` sekarang sebagai skill `.opencode/skills/` (bukan hardcode system-prompt), template real dari model tetap. `thinking` tetap dipertahankan karena kamu bilang sangat diperlukan — hanya di-strip kalau `enable_thinking: false`.
