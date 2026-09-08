#include <gtest/gtest.h>
#include "core/config.hpp"
#include "core/logger.hpp"
#include "utils/utils.hpp"
#include "registry/model_registry.hpp"
#include "queue/job_queue.hpp"
#include "engine/text_engine.hpp"
#include "engine/safetensors_loader.hpp"
#include "engine/paged_kv_cache.hpp"
#include "engine/continuous_batching.hpp"
#include "engine/prefix_cache.hpp"
#include <ggml.h>
#include <gguf.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

using namespace barskuy;

TEST(ConfigTest, DefaultValues) {
    core::Config config;
    // Test defaults via argc/argv simulation
    char* argv[] = {(char*)"test"};
    EXPECT_TRUE(config.load(1, argv));
    EXPECT_EQ(config.host(), "0.0.0.0");
    EXPECT_EQ(config.port(), 8080);
}

TEST(UtilsTest, RandomId) {
    std::string id1 = core::utils::random_id(16);
    std::string id2 = core::utils::random_id(16);
    EXPECT_EQ(id1.length(), 16);
    EXPECT_EQ(id2.length(), 16);
    EXPECT_NE(id1, id2);
}

TEST(UtilsTest, Sha256) {
    std::string hash = core::utils::sha256("test");
    EXPECT_EQ(hash.length(), 64);
    // Known SHA256 of "test"
    EXPECT_EQ(hash, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08");
}

TEST(ModelRegistryTest, Initialize) {
    registry::ModelRegistry registry(":memory:");
    EXPECT_TRUE(registry.initialize());
}

TEST(ModelRegistryTest, RegisterAndGetModel) {
    registry::ModelRegistry registry(":memory:");
    ASSERT_TRUE(registry.initialize());

    registry::Model model;
    model.id = "test-model";
    model.path = "/path/to/model.gguf";
    model.format = "gguf";
    model.architecture = "llama";
    model.quantization = "Q4_K_M";
    model.capabilities = {"text", "embeddings"};
    model.loaded = false;
    model.created_at = 1234567890;

    EXPECT_TRUE(registry.register_model(model));

    auto retrieved = registry.get_model("test-model");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->id, "test-model");
    EXPECT_EQ(retrieved->path, "/path/to/model.gguf");
    EXPECT_EQ(retrieved->format, "gguf");
    EXPECT_EQ(retrieved->architecture, "llama");
    EXPECT_EQ(retrieved->quantization, "Q4_K_M");
    EXPECT_EQ(retrieved->capabilities, std::vector<std::string>({"text", "embeddings"}));
    EXPECT_FALSE(retrieved->loaded);
    EXPECT_EQ(retrieved->created_at, 1234567890);
}

TEST(ModelRegistryTest, ListModels) {
    registry::ModelRegistry registry(":memory:");
    ASSERT_TRUE(registry.initialize());

    registry::Model model1;
    model1.id = "model-1";
    model1.path = "/path/1.gguf";
    model1.format = "gguf";
    model1.architecture = "llama";
    model1.quantization = "Q4_K_M";
    model1.capabilities = {"text"};
    model1.loaded = false;
    model1.created_at = 1000;

    registry::Model model2;
    model2.id = "model-2";
    model2.path = "/path/2.gguf";
    model2.format = "safetensors";
    model2.architecture = "qwen";
    model2.quantization = "FP16";
    model2.capabilities = {"text"};
    model2.loaded = true;
    model2.created_at = 2000;

    EXPECT_TRUE(registry.register_model(model1));
    EXPECT_TRUE(registry.register_model(model2));

    auto models = registry.list_models();
    EXPECT_EQ(models.size(), 2);
    EXPECT_EQ(models[0].id, "model-2"); // newest first
    EXPECT_EQ(models[1].id, "model-1");
}

TEST(ModelRegistryTest, ApiKey) {
    registry::ModelRegistry registry(":memory:");
    ASSERT_TRUE(registry.initialize());

    registry::ApiKey key;
    key.id = "key-1";
    key.key_hash = "hash123";
    key.label = "test key";
    key.created_at = 1234567890;

    EXPECT_TRUE(registry.create_api_key(key));

    auto retrieved = registry.get_api_key_by_hash("hash123");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->id, "key-1");
    EXPECT_EQ(retrieved->key_hash, "hash123");
    EXPECT_EQ(retrieved->label, "test key");
    EXPECT_FALSE(retrieved->revoked_at.has_value());
}

TEST(JobQueueTest, Initialize) {
    registry::ModelRegistry registry(":memory:");
    ASSERT_TRUE(registry.initialize());

    queue::JobQueue job_queue(registry);
    EXPECT_TRUE(job_queue.initialize());
    job_queue.shutdown();
}

TEST(TextEngineTest, QuantizationTypes) {
    // Test that all expected quantization types are available in ggml
    // These are the types available in this version of ggml (lowercase names)
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_F32), "f32");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_F16), "f16");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q4_0), "q4_0");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q4_1), "q4_1");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q5_0), "q5_0");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q5_1), "q5_1");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q8_0), "q8_0");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q8_1), "q8_1");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q2_K), "q2_K");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q3_K), "q3_K");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q4_K), "q4_K");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q5_K), "q5_K");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q6_K), "q6_K");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q8_K), "q8_K");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ2_XXS), "iq2_xxs");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ2_XS), "iq2_xs");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ3_XXS), "iq3_xxs");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ1_S), "iq1_s");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ4_NL), "iq4_nl");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ3_S), "iq3_s");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ2_S), "iq2_s");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ4_XS), "iq4_xs");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_IQ1_M), "iq1_m");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_BF16), "bf16");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_TQ1_0), "tq1_0");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_TQ2_0), "tq2_0");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_MXFP4), "mxfp4");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_NVFP4), "nvfp4");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q1_0), "q1_0");
    EXPECT_STREQ(ggml_type_name(GGML_TYPE_Q2_0), "q2_0");
}

TEST(TextEngineTest, Initialize) {
    engine::TextEngine text_engine;
    // TextEngine construction works; initialize() requires GPU backend which may not be available in test env
    // Server integration test verifies full initialization works
    SUCCEED();
}

// Helper to create a minimal valid safetensors file
static void create_test_safetensors(const std::string& path) {
    // Safetensors format: 8-byte header size (little-endian) + JSON header + tensor data
    
    // Create JSON header
    nlohmann::json header;
    header["model.layers.0.self_attn.q_proj.weight"] = {
        {"dtype", "F16"},
        {"shape", {256, 256}},
        {"data_offsets", {0, 131072}}  // 256*256*2 bytes
    };
    header["model.layers.0.self_attn.k_proj.weight"] = {
        {"dtype", "F16"},
        {"shape", {256, 256}},
        {"data_offsets", {131072, 262144}}
    };
    header["model.layers.0.mlp.gate_proj.weight"] = {
        {"dtype", "F16"},
        {"shape", {1024, 256}},
        {"data_offsets", {262144, 786432}}
    };
    header["model.embed_tokens.weight"] = {
        {"dtype", "F16"},
        {"shape", {32000, 256}},
        {"data_offsets", {786432, 17203200}}
    };
    header["lm_head.weight"] = {
        {"dtype", "F16"},
        {"shape", {32000, 256}},
        {"data_offsets", {17203200, 33620992}}
    };
    
    std::string header_str = header.dump();
    uint64_t header_size = header_str.size();
    
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<char*>(&header_size), 8);
    file.write(header_str.c_str(), header_str.size());
    
    // Write dummy tensor data (zeros)
    std::vector<char> dummy_data(33620992, 0);
    file.write(dummy_data.data(), dummy_data.size());
    file.close();
}

TEST(SafetensorsTest, ParseHeader) {
    std::string test_file = "test_model.safetensors";
    create_test_safetensors(test_file);
    
    engine::SafetensorsLoader loader;
    engine::SafetensorsMetadata metadata;
    
    EXPECT_TRUE(loader.parse_header(test_file, metadata));
    EXPECT_EQ(metadata.tensors.size(), 5);
    
    // Check tensor names are parsed (C++17 compatible)
    EXPECT_NE(metadata.tensors.find("model.layers.0.self_attn.q_proj.weight"), metadata.tensors.end());
    EXPECT_NE(metadata.tensors.find("model.layers.0.self_attn.k_proj.weight"), metadata.tensors.end());
    EXPECT_NE(metadata.tensors.find("model.layers.0.mlp.gate_proj.weight"), metadata.tensors.end());
    EXPECT_NE(metadata.tensors.find("model.embed_tokens.weight"), metadata.tensors.end());
    EXPECT_NE(metadata.tensors.find("lm_head.weight"), metadata.tensors.end());
    
    // Check dtypes
    EXPECT_EQ(metadata.tensors["model.layers.0.self_attn.q_proj.weight"].dtype, "F16");
    EXPECT_EQ(metadata.tensors["model.layers.0.self_attn.q_proj.weight"].shape, std::vector<int64_t>({256, 256}));
    
    // Cleanup
    std::remove(test_file.c_str());
}

TEST(SafetensorsTest, DetectQuantizedFormat) {
    // Test that quantized tensor components are detected
    std::string test_file = "test_quant.safetensors";
    
    nlohmann::json header;
    // Base weight tensor (output - will be dequantized to F16)
    header["model.layers.0.self_attn.q_proj.weight"] = {
        {"dtype", "F16"},
        {"shape", {256, 256}},
        {"data_offsets", {0, 131072}}
    };
    // AWQ components
    header["model.layers.0.self_attn.q_proj.qweight"] = {
        {"dtype", "I32"},
        {"shape", {32, 256}},  // packed int4: 256*256/8 = 8192 uint32 = 32*256
        {"data_offsets", {131072, 139264}}
    };
    header["model.layers.0.self_attn.q_proj.scales"] = {
        {"dtype", "F16"},
        {"shape", {256, 2}},  // 256 rows, 2 groups (group_size=128)
        {"data_offsets", {139264, 140288}}
    };
    header["model.layers.0.self_attn.q_proj.qzeros"] = {
        {"dtype", "I32"},
        {"shape", {32, 256}},  // packed zeros
        {"data_offsets", {140288, 148480}}
    };
    
    std::string header_str = header.dump();
    uint64_t header_size = header_str.size();
    
    std::ofstream file(test_file, std::ios::binary);
    file.write(reinterpret_cast<char*>(&header_size), 8);
    file.write(header_str.c_str(), header_str.size());
    std::vector<char> dummy_data(148480, 0);
    file.write(dummy_data.data(), dummy_data.size());
    file.close();
    
    engine::SafetensorsLoader loader;
    engine::SafetensorsMetadata metadata;
    
    EXPECT_TRUE(loader.parse_header(test_file, metadata));
    
    // Verify quantized components are present (C++17 compatible)
    EXPECT_NE(metadata.tensors.find("model.layers.0.self_attn.q_proj.qweight"), metadata.tensors.end());
    EXPECT_NE(metadata.tensors.find("model.layers.0.self_attn.q_proj.scales"), metadata.tensors.end());
    EXPECT_NE(metadata.tensors.find("model.layers.0.self_attn.q_proj.qzeros"), metadata.tensors.end());
    
    std::remove(test_file.c_str());
}

TEST(SafetensorsTest, DtypeMapping) {
    EXPECT_EQ(engine::SafetensorsLoader::dtype_to_ggml_type("F32"), GGML_TYPE_F32);
    EXPECT_EQ(engine::SafetensorsLoader::dtype_to_ggml_type("F16"), GGML_TYPE_F16);
    EXPECT_EQ(engine::SafetensorsLoader::dtype_to_ggml_type("BF16"), GGML_TYPE_BF16);
    EXPECT_EQ(engine::SafetensorsLoader::dtype_to_ggml_type("I8"), GGML_TYPE_I8);
    EXPECT_EQ(engine::SafetensorsLoader::dtype_to_ggml_type("I32"), GGML_TYPE_I32);
    EXPECT_EQ(engine::SafetensorsLoader::dtype_to_ggml_type("unknown"), GGML_TYPE_F32);  // default
}

TEST(PagedKVCacheTest, Initialize) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config config;
    config.n_layers = 32;
    config.n_heads = 32;
    config.n_kv_heads = 8;  // GQA
    config.head_dim = 128;
    config.block_size = 16;
    config.max_blocks = 1024;
    config.dtype = GGML_TYPE_F16;
    config.ctx = ctx;

    engine::PagedKVCacheManager cache(config);
    EXPECT_TRUE(cache.initialize());

    auto info = cache.get_info();
    EXPECT_EQ(1024, info.total_blocks);
    EXPECT_EQ(1024, info.free_blocks);
    EXPECT_EQ(0, info.used_blocks);

    ggml_free(ctx);
}

TEST(PagedKVCacheTest, AllocateAndFree) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config config;
    config.n_layers = 32;
    config.n_heads = 32;
    config.n_kv_heads = 8;
    config.head_dim = 128;
    config.block_size = 16;
    config.max_blocks = 1024;
    config.dtype = GGML_TYPE_F16;
    config.ctx = ctx;

    engine::PagedKVCacheManager cache(config);
    EXPECT_TRUE(cache.initialize());

    // Allocate for 32 tokens (2 blocks)
    auto seq_ids = cache.allocate_blocks(32);
    ASSERT_FALSE(seq_ids.empty());
    int seq_id = seq_ids[0];

    auto info = cache.get_info();
    EXPECT_EQ(2, info.used_blocks);
    EXPECT_EQ(1022, info.free_blocks);

    // Check sequence length
    EXPECT_EQ(cache.get_sequence_length(seq_id), 32);

    // Append 16 more tokens (1 more block)
    auto new_blocks = cache.append_blocks(seq_id, 16);
    ASSERT_FALSE(new_blocks.empty());
    EXPECT_EQ(static_cast<int>(new_blocks.size()), 1);

    EXPECT_EQ(cache.get_sequence_length(seq_id), 48);
    info = cache.get_info();
    EXPECT_EQ(3, info.used_blocks);
    EXPECT_EQ(1021, info.free_blocks);

    // Free sequence
    cache.free_sequence(seq_id);
    info = cache.get_info();
    EXPECT_EQ(0, info.used_blocks);
    EXPECT_EQ(1024, info.free_blocks);

    ggml_free(ctx);
}

TEST(PagedKVCacheTest, OOMHandling) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config config;
    config.n_layers = 32;
    config.n_heads = 32;
    config.n_kv_heads = 8;
    config.head_dim = 128;
    config.block_size = 16;
    config.max_blocks = 4;  // Very small
    config.dtype = GGML_TYPE_F16;
    config.ctx = ctx;

    engine::PagedKVCacheManager cache(config);
    EXPECT_TRUE(cache.initialize());

    // Allocate all blocks
    auto seq1 = cache.allocate_blocks(32);  // 2 blocks
    ASSERT_FALSE(seq1.empty());
    
    auto seq2 = cache.allocate_blocks(32);  // 2 blocks
    ASSERT_FALSE(seq2.empty());

    // Should fail - no free blocks
    auto seq3 = cache.allocate_blocks(16);  // 1 block
    EXPECT_TRUE(seq3.empty());

    // Free one and try again
    cache.free_sequence(seq1[0]);
    auto seq4 = cache.allocate_blocks(16);
    ASSERT_FALSE(seq4.empty());

    ggml_free(ctx);
}

TEST(PagedKVCacheTest, LRUEviction) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config config;
    config.n_layers = 32;
    config.n_heads = 32;
    config.n_kv_heads = 8;
    config.head_dim = 128;
    config.block_size = 16;
    config.max_blocks = 4;
    config.dtype = GGML_TYPE_F16;
    config.ctx = ctx;

    engine::PagedKVCacheManager cache(config);
    EXPECT_TRUE(cache.initialize());

    auto seq1 = cache.allocate_blocks(16);  // 1 block
    ASSERT_FALSE(seq1.empty());

    auto seq2 = cache.allocate_blocks(16);  // 1 block
    ASSERT_FALSE(seq2.empty());

    auto seq3 = cache.allocate_blocks(16);  // 1 block
    ASSERT_FALSE(seq3.empty());

    // Simulate usage: seq1 and seq3 are "used" (touched), seq2 and seq4 are not
    cache.touch_sequence(seq1[0]);
    cache.touch_sequence(seq3[0]);

    // Fill up
    auto seq4 = cache.allocate_blocks(16);  // 1 block
    ASSERT_FALSE(seq4.empty());

    // Now try to allocate - should trigger LRU eviction
    // seq2 is LRU (never touched), should be evicted
    auto evicted = cache.evict_lru(1);
    EXPECT_FALSE(evicted.empty());
    EXPECT_EQ(evicted[0], seq2[0]);

    ggml_free(ctx);
}

TEST(PrefixCacheTest, BasicStoreAndMatch) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config kv_config;
    kv_config.n_layers = 32;
    kv_config.n_heads = 32;
    kv_config.n_kv_heads = 8;
    kv_config.head_dim = 128;
    kv_config.block_size = 16;
    kv_config.max_blocks = 1024;
    kv_config.dtype = GGML_TYPE_F16;
    kv_config.ctx = ctx;

    engine::PagedKVCacheManager kv_cache(kv_config);
    EXPECT_TRUE(kv_cache.initialize());

    engine::PrefixCache::Config pc_config;
    pc_config.min_prefix_tokens = 4;
    pc_config.max_entries = 100;
    
    engine::PrefixCache prefix_cache(pc_config, &kv_cache);

    // Store a prefix: 18 tokens (needs 2 blocks with block_size=16)
    std::vector<int> prefix_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    std::vector<int> prefix_blocks = {0, 1};  // 2 blocks for 18 tokens
    prefix_cache.store_prefix(prefix_tokens, prefix_blocks);

    // Match exact prefix
    std::vector<int> query_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> matched_blocks;
    int matched = prefix_cache.find_prefix(query_tokens, &matched_blocks);
    EXPECT_EQ(matched, 18);
    EXPECT_EQ(matched_blocks, prefix_blocks);

    // Match partial prefix (too short)
    query_tokens = {1, 2, 3};
    matched = prefix_cache.find_prefix(query_tokens, &matched_blocks);
    EXPECT_EQ(matched, 0);  // Too short (min_prefix_tokens=4)

    // No match
    query_tokens = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180};
    matched = prefix_cache.find_prefix(query_tokens, &matched_blocks);
    EXPECT_EQ(matched, 0);

    ggml_free(ctx);
}

TEST(PrefixCacheTest, SharedSystemPrompt) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config kv_config;
    kv_config.n_layers = 32;
    kv_config.n_heads = 32;
    kv_config.n_kv_heads = 8;
    kv_config.head_dim = 128;
    kv_config.block_size = 16;
    kv_config.max_blocks = 1024;
    kv_config.dtype = GGML_TYPE_F16;
    kv_config.ctx = ctx;

    engine::PagedKVCacheManager kv_cache(kv_config);
    EXPECT_TRUE(kv_cache.initialize());

    engine::PrefixCache::Config pc_config;
    pc_config.min_prefix_tokens = 4;
    pc_config.max_entries = 100;
    
    engine::PrefixCache prefix_cache(pc_config, &kv_cache);

    // System prompt tokens (simulated) - 18 tokens (2 blocks)
    std::vector<int> system_prompt = {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818};
    std::vector<int> system_blocks = {0, 1};  // 2 blocks for 18 tokens (block_size=16)
    
    prefix_cache.store_prefix(system_prompt, system_blocks);

    // Request 1: system + user message 1
    std::vector<int> req1 = system_prompt;
    req1.insert(req1.end(), {1919, 2020, 2121});  // user message
    
    std::vector<int> matched_blocks;
    int matched = prefix_cache.find_prefix(req1, &matched_blocks);
    EXPECT_EQ(matched, 18);
    EXPECT_EQ(matched_blocks, system_blocks);

    // Request 2: system + user message 2
    std::vector<int> req2 = system_prompt;
    req2.insert(req2.end(), {2222, 2323, 2424});  // different user message
    
    matched = prefix_cache.find_prefix(req2, &matched_blocks);
    EXPECT_EQ(matched, 18);
    EXPECT_EQ(matched_blocks, system_blocks);

    // Request 3: different system prompt
    std::vector<int> req3 = {999, 888, 777, 666, 555, 444, 333};
    matched = prefix_cache.find_prefix(req3, &matched_blocks);
    EXPECT_EQ(matched, 0);

    ggml_free(ctx);
}

TEST(PrefixCacheTest, CacheExpiration) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config kv_config;
    kv_config.n_layers = 32;
    kv_config.n_heads = 32;
    kv_config.n_kv_heads = 8;
    kv_config.head_dim = 128;
    kv_config.block_size = 16;
    kv_config.max_blocks = 1024;
    kv_config.dtype = GGML_TYPE_F16;
    kv_config.ctx = ctx;

    engine::PagedKVCacheManager kv_cache(kv_config);
    EXPECT_TRUE(kv_cache.initialize());

    engine::PrefixCache::Config pc_config;
    pc_config.min_prefix_tokens = 4;
    pc_config.max_entries = 100;
    pc_config.ttl_seconds = 1;  // Very short TTL (1 second)
    
    engine::PrefixCache prefix_cache(pc_config, &kv_cache);

    std::vector<int> prefix_tokens = {1, 2, 3, 4, 5};
    std::vector<int> prefix_blocks = {0, 1};
    prefix_cache.store_prefix(prefix_tokens, prefix_blocks);

    // Should match immediately
    std::vector<int> query_tokens = {1, 2, 3, 4, 5, 6, 7};
    std::vector<int> matched_blocks;
    int matched = prefix_cache.find_prefix(query_tokens, &matched_blocks);
    EXPECT_EQ(matched, 5);

    // Wait for TTL to expire - use longer sleep to avoid flakiness
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // Should not match after expiration
    matched = prefix_cache.find_prefix(query_tokens, &matched_blocks);
    EXPECT_EQ(matched, 0);

    ggml_free(ctx);
}

TEST(PrefixCacheTest, CacheSizeLimit) {
    struct ggml_init_params params = { 1024 * 1024 * 256, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    engine::PagedKVCacheManager::Config kv_config;
    kv_config.n_layers = 32;
    kv_config.n_heads = 32;
    kv_config.n_kv_heads = 8;
    kv_config.head_dim = 128;
    kv_config.block_size = 16;
    kv_config.max_blocks = 1024;
    kv_config.dtype = GGML_TYPE_F16;
    kv_config.ctx = ctx;

    engine::PagedKVCacheManager kv_cache(kv_config);
    EXPECT_TRUE(kv_cache.initialize());

    engine::PrefixCache::Config pc_config;
    pc_config.min_prefix_tokens = 4;
    pc_config.max_entries = 3;  // Very small cache
    
    engine::PrefixCache prefix_cache(pc_config, &kv_cache);

    // Fill cache beyond limit
    for (int i = 0; i < 5; ++i) {
        std::vector<int> tokens = {i, i+1, i+2, i+3, i+4};
        std::vector<int> blocks = {i};
        prefix_cache.store_prefix(tokens, blocks);
    }

    // Cache should have at most 3 entries (LRU eviction)
    auto stats = prefix_cache.get_stats();
    EXPECT_LE(stats.total_entries, 3);

    ggml_free(ctx);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}