cmake_minimum_required(VERSION 3.20)

project(barskuy-llm-tests LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common)

add_executable(test_core test_core.cpp
    ${CMAKE_SOURCE_DIR}/src/core/config.cpp
    ${CMAKE_SOURCE_DIR}/src/core/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/registry/model_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/registry/architecture_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/queue/job_queue.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/text_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/safetensors_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/dequantize.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/paged_kv_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/continuous_batching.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/prefix_cache.cpp
)

target_link_libraries(test_core PRIVATE
    gtest_main
    gtest
    fmt::fmt
    OpenSSL::SSL
    OpenSSL::Crypto
    sqlite3
    llama
    llama-common
    ggml
    ggml-base
    ggml-cpu
)

add_test(NAME test_core COMMAND test_core)