#include "model_registry.hpp"
#include "core/logger.hpp"
#include <chrono>
#include <functional>

namespace barskuy::registry {

ModelRegistry::ModelRegistry(const std::string& db_path) : db_path_(db_path) {}

ModelRegistry::~ModelRegistry() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool ModelRegistry::initialize() {
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        core::Logger::error("Cannot open database: {}", sqlite3_errmsg(db_));
        return false;
    }

    const char* schema = R"(
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            format TEXT NOT NULL CHECK(format IN ('gguf','safetensors')),
            architecture TEXT NOT NULL,
            quantization TEXT,
            capabilities TEXT NOT NULL,
            loaded INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS generation_jobs (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL CHECK(type IN ('image','video')),
            model_id TEXT REFERENCES models(id),
            status TEXT NOT NULL CHECK(status IN ('queued','processing','completed','failed')),
            progress INTEGER DEFAULT 0,
            request_params TEXT NOT NULL,
            result_path TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            completed_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL UNIQUE,
            label TEXT,
            created_at INTEGER NOT NULL,
            revoked_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_id TEXT REFERENCES api_keys(id),
            endpoint TEXT NOT NULL,
            model_id TEXT,
            tokens_or_units INTEGER,
            timestamp INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS architecture_definitions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            tensor_mapping TEXT NOT NULL,
            graph_template_ref TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_models_loaded ON models(loaded);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON generation_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_type ON generation_jobs(type);
        CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_usage_logs_key_time ON usage_logs(api_key_id, timestamp);
    )";

    char* err_msg = nullptr;
    rc = sqlite3_exec(db_, schema, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        core::Logger::error("Failed to create schema: {}", err_msg ? err_msg : "unknown");
        sqlite3_free(err_msg);
        return false;
    }

    core::Logger::info("Model registry initialized at {}", db_path_);
    return true;
}

bool ModelRegistry::exec(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        core::Logger::error("SQL error: {}", err_msg ? err_msg : "unknown");
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

bool ModelRegistry::exec_prepared(const std::string& sql, std::function<void(sqlite3_stmt*)> binder) {
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        core::Logger::error("Prepare failed: {}", sqlite3_errmsg(db_));
        return false;
    }

    binder(stmt);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        core::Logger::error("Execute failed: {}", sqlite3_errmsg(db_));
        sqlite3_finalize(stmt);
        return false;
    }

    sqlite3_finalize(stmt);
    return true;
}

bool ModelRegistry::register_model(const Model& model) {
    std::string caps_json = nlohmann::json(model.capabilities).dump();
    int64_t created_at = model.created_at > 0 ? model.created_at :
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    return exec_prepared(
        "INSERT OR REPLACE INTO models (id, path, format, architecture, quantization, capabilities, loaded, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, model.id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, model.path.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 3, model.format.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 4, model.architecture.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 5, model.quantization.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 6, caps_json.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int(stmt, 7, model.loaded ? 1 : 0);
            sqlite3_bind_int64(stmt, 8, created_at);
        });
}

bool ModelRegistry::unregister_model(const std::string& id) {
    return exec_prepared("DELETE FROM models WHERE id = ?",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);
        });
}

std::optional<Model> ModelRegistry::get_model(const std::string& id) {
    std::optional<Model> result;
    std::string sql = "SELECT id, path, format, architecture, quantization, capabilities, loaded, created_at FROM models WHERE id = ?";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }

    sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        Model model;
        model.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        model.path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        model.format = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        model.architecture = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        model.quantization = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        model.capabilities = nlohmann::json::parse(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5))).get<std::vector<std::string>>();
        model.loaded = sqlite3_column_int(stmt, 6) != 0;
        model.created_at = sqlite3_column_int64(stmt, 7);
        result = std::move(model);
    }

    sqlite3_finalize(stmt);
    return result;
}

std::vector<Model> ModelRegistry::list_models() {
    std::vector<Model> models;
    std::string sql = "SELECT id, path, format, architecture, quantization, capabilities, loaded, created_at FROM models ORDER BY created_at DESC";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return models;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Model model;
        model.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        model.path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        model.format = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        model.architecture = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        model.quantization = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        model.capabilities = nlohmann::json::parse(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5))).get<std::vector<std::string>>();
        model.loaded = sqlite3_column_int(stmt, 6) != 0;
        model.created_at = sqlite3_column_int64(stmt, 7);
        models.push_back(std::move(model));
    }

    sqlite3_finalize(stmt);
    return models;
}

bool ModelRegistry::update_model_loaded(const std::string& id, bool loaded) {
    return exec_prepared("UPDATE models SET loaded = ? WHERE id = ?",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_int(stmt, 1, loaded ? 1 : 0);
            sqlite3_bind_text(stmt, 2, id.c_str(), -1, SQLITE_STATIC);
        });
}

bool ModelRegistry::create_api_key(const ApiKey& key) {
    return exec_prepared(
        "INSERT INTO api_keys (id, key_hash, label, created_at, revoked_at) VALUES (?, ?, ?, ?, ?)",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, key.id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, key.key_hash.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 3, key.label.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 4, key.created_at);
            if (key.revoked_at) {
                sqlite3_bind_int64(stmt, 5, *key.revoked_at);
            } else {
                sqlite3_bind_null(stmt, 5);
            }
        });
}

bool ModelRegistry::revoke_api_key(const std::string& id) {
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return exec_prepared("UPDATE api_keys SET revoked_at = ? WHERE id = ?",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_int64(stmt, 1, now);
            sqlite3_bind_text(stmt, 2, id.c_str(), -1, SQLITE_STATIC);
        });
}

std::optional<ApiKey> ModelRegistry::get_api_key_by_hash(const std::string& key_hash) {
    std::optional<ApiKey> result;
    std::string sql = "SELECT id, key_hash, label, created_at, revoked_at FROM api_keys WHERE key_hash = ?";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }

    sqlite3_bind_text(stmt, 1, key_hash.c_str(), -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        ApiKey key;
        key.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        key.key_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        key.label = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        key.created_at = sqlite3_column_int64(stmt, 3);
        if (sqlite3_column_type(stmt, 4) != SQLITE_NULL) {
            key.revoked_at = sqlite3_column_int64(stmt, 4);
        }
        result = std::move(key);
    }

    sqlite3_finalize(stmt);
    return result;
}

std::vector<ApiKey> ModelRegistry::list_api_keys() {
    std::vector<ApiKey> keys;
    std::string sql = "SELECT id, key_hash, label, created_at, revoked_at FROM api_keys ORDER BY created_at DESC";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return keys;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ApiKey key;
        key.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        key.key_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        key.label = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        key.created_at = sqlite3_column_int64(stmt, 3);
        if (sqlite3_column_type(stmt, 4) != SQLITE_NULL) {
            key.revoked_at = sqlite3_column_int64(stmt, 4);
        }
        keys.push_back(std::move(key));
    }

    sqlite3_finalize(stmt);
    return keys;
}

bool ModelRegistry::create_job(const GenerationJob& job) {
    std::string params_json = job.request_params.dump();
    return exec_prepared(
        "INSERT INTO generation_jobs (id, type, model_id, status, progress, request_params, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, job.id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, job.type.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 3, job.model_id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 4, job.status.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int(stmt, 5, job.progress);
            sqlite3_bind_text(stmt, 6, params_json.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 7, job.created_at);
        });
}

bool ModelRegistry::update_job_status(const std::string& id, const std::string& status, int progress) {
    if (progress >= 0) {
        return exec_prepared("UPDATE generation_jobs SET status = ?, progress = ? WHERE id = ?",
            [&](sqlite3_stmt* stmt) {
                sqlite3_bind_text(stmt, 1, status.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_int(stmt, 2, progress);
                sqlite3_bind_text(stmt, 3, id.c_str(), -1, SQLITE_STATIC);
            });
    } else {
        return exec_prepared("UPDATE generation_jobs SET status = ? WHERE id = ?",
            [&](sqlite3_stmt* stmt) {
                sqlite3_bind_text(stmt, 1, status.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(stmt, 2, id.c_str(), -1, SQLITE_STATIC);
            });
    }
}

bool ModelRegistry::complete_job(const std::string& id, const std::string& result_path) {
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return exec_prepared(
        "UPDATE generation_jobs SET status = 'completed', progress = 100, result_path = ?, completed_at = ? WHERE id = ?",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, result_path.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 2, now);
            sqlite3_bind_text(stmt, 3, id.c_str(), -1, SQLITE_STATIC);
        });
}

bool ModelRegistry::fail_job(const std::string& id, const std::string& error) {
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return exec_prepared(
        "UPDATE generation_jobs SET status = 'failed', error = ?, completed_at = ? WHERE id = ?",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, error.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 2, now);
            sqlite3_bind_text(stmt, 3, id.c_str(), -1, SQLITE_STATIC);
        });
}

std::optional<GenerationJob> ModelRegistry::get_job(const std::string& id) {
    std::optional<GenerationJob> result;
    std::string sql = "SELECT id, type, model_id, status, progress, request_params, result_path, error, created_at, completed_at FROM generation_jobs WHERE id = ?";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }

    sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        GenerationJob job;
        job.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        job.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        job.model_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        job.status = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        job.progress = sqlite3_column_int(stmt, 4);
        job.request_params = nlohmann::json::parse(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        if (sqlite3_column_type(stmt, 6) != SQLITE_NULL) {
            job.result_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        }
        if (sqlite3_column_type(stmt, 7) != SQLITE_NULL) {
            job.error = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        }
        job.created_at = sqlite3_column_int64(stmt, 8);
        if (sqlite3_column_type(stmt, 9) != SQLITE_NULL) {
            job.completed_at = sqlite3_column_int64(stmt, 9);
        }
        result = std::move(job);
    }

    sqlite3_finalize(stmt);
    return result;
}

std::vector<GenerationJob> ModelRegistry::list_jobs(const std::optional<std::string>& status,
                                                     const std::optional<std::string>& type) {
    std::vector<GenerationJob> jobs;
    std::string sql = "SELECT id, type, model_id, status, progress, request_params, result_path, error, created_at, completed_at FROM generation_jobs";
    bool first = true;

    if (status || type) {
        sql += " WHERE";
        if (status) {
            sql += " status = ?";
            first = false;
        }
        if (type) {
            if (!first) sql += " AND";
            sql += " type = ?";
        }
    }
    sql += " ORDER BY created_at DESC";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return jobs;
    }

    int param_idx = 1;
    if (status) sqlite3_bind_text(stmt, param_idx++, status->c_str(), -1, SQLITE_STATIC);
    if (type) sqlite3_bind_text(stmt, param_idx++, type->c_str(), -1, SQLITE_STATIC);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        GenerationJob job;
        job.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        job.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        job.model_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        job.status = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        job.progress = sqlite3_column_int(stmt, 4);
        job.request_params = nlohmann::json::parse(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        if (sqlite3_column_type(stmt, 6) != SQLITE_NULL) {
            job.result_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        }
        if (sqlite3_column_type(stmt, 7) != SQLITE_NULL) {
            job.error = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        }
        job.created_at = sqlite3_column_int64(stmt, 8);
        if (sqlite3_column_type(stmt, 9) != SQLITE_NULL) {
            job.completed_at = sqlite3_column_int64(stmt, 9);
        }
        jobs.push_back(std::move(job));
    }

    sqlite3_finalize(stmt);
    return jobs;
}

bool ModelRegistry::cleanup_old_jobs(int days) {
    int64_t cutoff = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() - (days * 86400);
    return exec_prepared("DELETE FROM generation_jobs WHERE created_at < ? AND status IN ('completed', 'failed')",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_int64(stmt, 1, cutoff);
        });
}

bool ModelRegistry::log_usage(const UsageLog& log) {
    return exec_prepared(
        "INSERT INTO usage_logs (api_key_id, endpoint, model_id, tokens_or_units, timestamp) VALUES (?, ?, ?, ?, ?)",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, log.api_key_id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, log.endpoint.c_str(), -1, SQLITE_STATIC);
            if (log.model_id) {
                sqlite3_bind_text(stmt, 3, log.model_id->c_str(), -1, SQLITE_STATIC);
            } else {
                sqlite3_bind_null(stmt, 3);
            }
            sqlite3_bind_int64(stmt, 4, log.tokens_or_units);
            sqlite3_bind_int64(stmt, 5, log.timestamp);
        });
}

std::vector<UsageLog> ModelRegistry::get_usage_logs(const std::string& api_key_id, int64_t since) {
    std::vector<UsageLog> logs;
    std::string sql = "SELECT id, api_key_id, endpoint, model_id, tokens_or_units, timestamp FROM usage_logs WHERE api_key_id = ? AND timestamp >= ? ORDER BY timestamp DESC";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return logs;
    }

    sqlite3_bind_text(stmt, 1, api_key_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, since);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        UsageLog log;
        log.id = sqlite3_column_int64(stmt, 0);
        log.api_key_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        log.endpoint = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        if (sqlite3_column_type(stmt, 3) != SQLITE_NULL) {
            log.model_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        }
        log.tokens_or_units = sqlite3_column_int64(stmt, 4);
        log.timestamp = sqlite3_column_int64(stmt, 5);
        logs.push_back(std::move(log));
    }

    sqlite3_finalize(stmt);
    return logs;
}

bool ModelRegistry::register_architecture(const ArchitectureDefinition& def) {
    std::string mapping_json = def.tensor_mapping.dump();
    return exec_prepared(
        "INSERT OR REPLACE INTO architecture_definitions (id, name, tensor_mapping, graph_template_ref, created_at) VALUES (?, ?, ?, ?, ?)",
        [&](sqlite3_stmt* stmt) {
            sqlite3_bind_text(stmt, 1, def.id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 2, def.name.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 3, mapping_json.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmt, 4, def.graph_template_ref.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 5, def.created_at);
        });
}

std::optional<ArchitectureDefinition> ModelRegistry::get_architecture(const std::string& name) {
    std::optional<ArchitectureDefinition> result;
    std::string sql = "SELECT id, name, tensor_mapping, graph_template_ref, created_at FROM architecture_definitions WHERE name = ?";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        ArchitectureDefinition def;
        def.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        def.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        def.tensor_mapping = nlohmann::json::parse(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        def.graph_template_ref = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        def.created_at = sqlite3_column_int64(stmt, 4);
        result = std::move(def);
    }

    sqlite3_finalize(stmt);
    return result;
}

std::vector<ArchitectureDefinition> ModelRegistry::list_architectures() {
    std::vector<ArchitectureDefinition> defs;
    std::string sql = "SELECT id, name, tensor_mapping, graph_template_ref, created_at FROM architecture_definitions ORDER BY created_at DESC";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return defs;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ArchitectureDefinition def;
        def.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        def.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        def.tensor_mapping = nlohmann::json::parse(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        def.graph_template_ref = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        def.created_at = sqlite3_column_int64(stmt, 4);
        defs.push_back(std::move(def));
    }

    sqlite3_finalize(stmt);
    return defs;
}

} // namespace barskuy::registry