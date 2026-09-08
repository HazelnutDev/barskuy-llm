cmake_minimum_required(VERSION 3.20)
project(barskuy-llm VERSION 3.2.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
    add_compile_options(/bigobj)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter -pthread")
endif()

option(BARSKUY_BUILD_CUDA "Build with CUDA backend" ON)
option(BARSKUY_BUILD_VULKAN "Build with Vulkan backend" ON)
option(BARSKUY_BUILD_METAL "Build with Metal backend" OFF)
option(BARSKUY_BUILD_ROCM "Build with ROCm backend" OFF)

set(BUILD_SHARED_LIBS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/third_party)

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/sqlite3)
add_subdirectory(third_party/googletest)

find_package(OpenSSL REQUIRED)

# Single ggml substrate via llama.cpp (includes ggml, ggml-base, ggml-cpu, common)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)
add_subdirectory(src)

include(CTest)
enable_testing()
add_subdirectory(tests)