from sqlalchemy import create_engine, inspect

from app.migrations import apply_additive_migrations


def test_additive_migration_upgrades_legacy_tables() -> None:
    migration_engine = create_engine("sqlite:///:memory:")
    with migration_engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE jobs (id VARCHAR(36) PRIMARY KEY)")
        connection.exec_driver_sql("CREATE TABLE worker_heartbeats (worker_name VARCHAR(128) PRIMARY KEY)")

    apply_additive_migrations(migration_engine)
    inspector = inspect(migration_engine)
    job_columns = {item["name"] for item in inspector.get_columns("jobs")}
    worker_columns = {item["name"] for item in inspector.get_columns("worker_heartbeats")}

    assert {"engine_requested", "backend_used", "precision_used", "error_code"} <= job_columns
    assert {"backend", "devices", "capabilities", "last_error"} <= worker_columns


def test_additive_migration_is_idempotent() -> None:
    migration_engine = create_engine("sqlite:///:memory:")
    with migration_engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE jobs (id VARCHAR(36) PRIMARY KEY)")
        connection.exec_driver_sql("CREATE TABLE worker_heartbeats (worker_name VARCHAR(128) PRIMARY KEY)")
    apply_additive_migrations(migration_engine)
    apply_additive_migrations(migration_engine)
