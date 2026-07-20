from dataclasses import dataclass

from sqlalchemy import Engine, inspect


@dataclass(frozen=True)
class AdditiveColumn:
    name: str
    ddl: str


ADDITIVE_COLUMNS: dict[str, tuple[AdditiveColumn, ...]] = {
    "jobs": (
        AdditiveColumn("engine_requested", "VARCHAR(64) DEFAULT 'openai-whisper' NOT NULL"),
        AdditiveColumn("engine_used", "VARCHAR(64)"),
        AdditiveColumn("device_requested", "JSON"),
        AdditiveColumn("backend_used", "VARCHAR(32)"),
        AdditiveColumn("device_index_used", "INTEGER"),
        AdditiveColumn("runtime_device_used", "VARCHAR(64)"),
        AdditiveColumn("precision_requested", "VARCHAR(32) DEFAULT 'auto' NOT NULL"),
        AdditiveColumn("precision_used", "VARCHAR(32)"),
        AdditiveColumn("fallback_used", "BOOLEAN DEFAULT FALSE NOT NULL"),
        AdditiveColumn("fallback_reason", "TEXT"),
        AdditiveColumn("worker_id", "VARCHAR(128)"),
        AdditiveColumn("hardware_snapshot", "JSON"),
        AdditiveColumn("runtime_versions", "JSON"),
        AdditiveColumn("error_code", "VARCHAR(64)"),
    ),
    "worker_heartbeats": (
        AdditiveColumn("hostname", "VARCHAR(255)"),
        AdditiveColumn("backend", "VARCHAR(32)"),
        AdditiveColumn("max_concurrency", "INTEGER DEFAULT 1 NOT NULL"),
        AdditiveColumn("active_jobs", "INTEGER DEFAULT 0 NOT NULL"),
        AdditiveColumn("engines", "JSON"),
        AdditiveColumn("devices", "JSON"),
        AdditiveColumn("capabilities", "JSON"),
        AdditiveColumn("loaded_models", "JSON"),
        AdditiveColumn("last_error", "TEXT"),
        AdditiveColumn("started_at", "TIMESTAMP WITH TIME ZONE"),
    ),
}


def apply_additive_migrations(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table, additions in ADDITIVE_COLUMNS.items():
            if table not in tables:
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for column in additions:
                if column.name in existing:
                    continue
                ddl = column.ddl
                if engine.dialect.name == "sqlite":
                    ddl = ddl.replace("TIMESTAMP WITH TIME ZONE", "DATETIME")
                connection.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN "{column.name}" {ddl}')

