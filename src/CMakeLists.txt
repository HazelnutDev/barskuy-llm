file(GLOB_RECURSE BARSKUY_SOURCES
    "core/*.cpp"
    "engine/*.cpp"
    "agent/*.cpp"
    "api/*.cpp"
    "registry/*.cpp"
    "queue/*.cpp"
    "utils/*.cpp"
    "main.cpp"
)

# Fase 9: embed real llama.cpp WebUI (tools/ui dist) into the binary.
# Regenerate: rebuild the UI, then re-run cmake configure (or build target webui-regen).
set(BARSKUY_WEBUI_DIST "${CMAKE_SOURCE_DIR}/llama.cpp/tools/ui/dist-barskuy")
set(BARSKUY_WEBUI_GEN "${CMAKE_CURRENT_BINARY_DIR}/webui_embed_gen.cpp")
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND AND EXISTS "${BARSKUY_WEBUI_DIST}/index.html")
    add_custom_command(
        OUTPUT "${BARSKUY_WEBUI_GEN}"
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/embed_webui.py"
                "${BARSKUY_WEBUI_DIST}" "${BARSKUY_WEBUI_GEN}"
        DEPENDS "${BARSKUY_WEBUI_DIST}/index.html"
        COMMENT "Embedding WebUI into barskuy-llm binary"
        VERBATIM
    )
    add_custom_target(webui-regen DEPENDS "${BARSKUY_WEBUI_GEN}")
    list(APPEND BARSKUY_SOURCES "${BARSKUY_WEBUI_GEN}")
    set(BARSKUY_WEBUI_EMBEDDED ON)
else()
    message(WARNING "WebUI dist not found - binary will serve without embedded UI")
endif()

# Fase 10: stable-diffusion.cpp image backend (real txt2img). Built isolated
# via ExternalProject with its own ggml -> no symbol clash with ours.
# Produces stable-diffusion.dll (+ import lib); DLL ships next to the exe.
# NOTE: Release-only artifact paths (we always ship Release).
option(BARSKUY_WITH_SD "Build stable-diffusion.cpp image backend" ON)
set(BARSKUY_SD_ENABLED OFF)
if(BARSKUY_WITH_SD AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/sd.cpp/CMakeLists.txt")
    include(ExternalProject)
    if(BARSKUY_BUILD_VULKAN)
        set(SD_VULKAN_FLAG ON)
    else()
        set(SD_VULKAN_FLAG OFF)
    endif()
    set(SD_BUILD_DIR "${CMAKE_BINARY_DIR}/sd-build")
    ExternalProject_Add(sd_backend
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/sd.cpp"
        BINARY_DIR "${SD_BUILD_DIR}"
        CMAKE_ARGS
            -DSD_BUILD_SHARED_LIBS=ON
            -DSD_VULKAN=${SD_VULKAN_FLAG}
            -DSD_WEBP=OFF
            -DSD_WEBM=OFF
            -DSD_BUILD_EXAMPLES=OFF
            -DCMAKE_BUILD_TYPE=Release
        BUILD_BYPRODUCTS
            "${SD_BUILD_DIR}/bin/Release/stable-diffusion.dll"
            "${SD_BUILD_DIR}/Release/stable-diffusion.lib"
        INSTALL_COMMAND ""
    )
    set(BARSKUY_SD_ENABLED ON)
endif()

add_executable(barskuy-llm ${BARSKUY_SOURCES})

target_include_directories(barskuy-llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/agent
    ${CMAKE_CURRENT_SOURCE_DIR}/api
    ${CMAKE_CURRENT_SOURCE_DIR}/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/queue
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common
)

target_link_libraries(barskuy-llm PRIVATE
    llama
    llama-common
    ggml
    ggml-base
    ggml-cpu
    fmt::fmt
    OpenSSL::SSL
    OpenSSL::Crypto
    sqlite3
)

if(BARSKUY_BUILD_CUDA)
    find_package(CUDA REQUIRED)
    target_link_libraries(barskuy-llm PRIVATE ${CUDA_LIBRARIES})
    target_compile_definitions(barskuy-llm PRIVATE GGML_USE_CUDA)
endif()

if(BARSKUY_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(barskuy-llm PRIVATE ${Vulkan_LIBRARIES})
    target_compile_definitions(barskuy-llm PRIVATE GGML_USE_VULKAN)
    if(TARGET ggml-vulkan)
        target_link_libraries(barskuy-llm PRIVATE ggml-vulkan)
    endif()
endif()

if(WIN32)
    target_link_libraries(barskuy-llm PRIVATE ws2_32 iphlpapi userenv)
    target_compile_definitions(barskuy-llm PRIVATE _WIN32_WINNT=0x0A00)
endif()

target_compile_definitions(barskuy-llm PRIVATE
    GGML_USE_CPU=1
    GGML_USE_BLAS=0
    BARSKUY_VERSION="${PROJECT_VERSION}"
    _CRT_SECURE_NO_WARNINGS
)

if(BARSKUY_SD_ENABLED)
    add_dependencies(barskuy-llm sd_backend)
    target_include_directories(barskuy-llm PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/sd.cpp/include)
    target_link_libraries(barskuy-llm PRIVATE
        "${SD_BUILD_DIR}/Release/stable-diffusion.lib")
    target_compile_definitions(barskuy-llm PRIVATE BARSKUY_WITH_SD=1)
    add_custom_command(TARGET barskuy-llm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SD_BUILD_DIR}/bin/Release/stable-diffusion.dll"
            $<TARGET_FILE_DIR:barskuy-llm>)
endif()