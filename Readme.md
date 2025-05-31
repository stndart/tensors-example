# TensorMultiply

A minimal C++/CUDA project for multiplying tensors (or arrays) on either CPU or GPU. CUDA support is optional and controlled via a CMake flag.

---

## Prerequisites

- **CMake 3.18+** (for built-in CUDA support)  
- **C++17-capable compiler**  
  - Windows: Visual Studio 2019/2022 (MSVC)  
  - Linux/macOS: GCC 7+ or Clang 7+  
- **(Optional) NVIDIA CUDA Toolkit 10.0+**  
  - If you enable CUDA, ensure `nvcc` is on your PATH.

---

## File Structure

```
TensorMultiply/
├── CMakeLists.txt            # Project configuration, ENABLE_CUDA option
├── include/                  # Public headers (e.g. tensor.hpp)
│   └── …
└── src/                      
    ├── main.cpp              # Entry point, dispatches CPU or GPU code
    ├── tensor_cpu.cpp        # (Example) CPU-only tensor multiply
    └── tensor_gpu.cu         # (Example) CUDA kernel + wrapper (compiled if ENABLE_CUDA=ON)
```

- **CMakeLists.txt**  
  - Defines `option(ENABLE_CUDA "Enable CUDA support" ON)`.  
  - When **ON**: `project(... LANGUAGES CXX CUDA)`, gathers `.cu` files, sets `CUDA_ARCHITECTURES` (default: “75”), and compiles GPU code.  
  - When **OFF**: `project(... LANGUAGES CXX)`, ignores any `.cu` files and compiles CPU-only code.

- **include/**  
  - Place any shared headers (e.g. `tensor.hpp`) here; it’s automatically added to the include path.

- **src/**  
  - All `.cpp` and `.cu` source files live here. CMake will glob-recursively pick up everything under `src/`.

---

## Build Instructions

Below are the commands to configure and build with CUDA enabled, using Visual Studio 2022 presets on Windows. Adjust the generator (e.g. `-G "Unix Makefiles"`) if you’re on Linux/macOS.

1. **Configure (CUDA ON)**  
   ```powershell
   cd TensorMultiply
   cmake -S . -B build -DENABLE_CUDA=ON --preset=vs2022
   ```
   - Uses CMakePresets for `Visual Studio 17 2022` x64.  
   - You can omit `-DENABLE_CUDA=ON` to use the default (ON).

2. **Build (Release)**  
   ```powershell
   cmake --build build --preset=vs2022-release
   ```
   - Produces `TensorMultiply.exe` in `build/Release/`.

3. **Disabling CUDA (CPU-only)**  
   ```powershell
   cmake -S . -B build-cpu -DENABLE_CUDA=OFF --preset=vs2022
   cmake --build build-cpu --preset=vs2022-release
   ```
   - With `ENABLE_CUDA=OFF`, any `.cu` files are ignored and only CPU code is compiled.

---

## Resulting Executable

- **CUDA build (Release):**  
  `TensorMultiply.exe` → `TensorMultiply/build/Release/TensorMultiply.exe`
- **CPU-only build (Release):**  
  `TensorMultiply.exe` → `TensorMultiply/build-cpu/Release/TensorMultiply.exe`

Simply run the `.exe` (or binary on Linux/macOS) to execute the tensor-multiply logic. If CUDA is enabled and a compatible GPU is detected, the CUDA path will be taken; otherwise, it will fall back to a CPU implementation.

---

## Usage

```bash
# Example (Windows Powershell):
cd TensorMultiply/build/Release
.\TensorMultiply.exe
```

No additional command-line arguments are required—any input sizes or parameters can be hardcoded or extended in `main.cpp`.

---

## Notes

- To adjust GPU architecture, edit:
  ```cmake
  set_target_properties(TensorMultiply PROPERTIES
      CUDA_ARCHITECTURES "75"  # Change to your GPU’s compute capability (e.g. "80" for Ampere)
      CUDA_SEPARABLE_COMPILATION ON
  )
  ```
- On Linux/macOS, replace the generator preset with your preferred generator:
  ```bash
  mkdir -p build && cd build
  cmake -DENABLE_CUDA=ON ..
  make
  ```

That’s it—TensorMultiply is now set up to compile and run on CPU or GPU with a single, simple CMake option.