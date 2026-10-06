"""
Database Connection & Lifecycle Manager for PostgreSQL 16
Using AsyncPG with PostGIS, pgvector, and JSONB codecs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, Sequence, Union

# pyrefly: ignore [missing-import]
import asyncpg
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Automatically load environment variables from .env
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("civic_db")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/civic_incidents",
)
DB_MIN_CONNECTIONS = int(os.getenv("DB_MIN_CONNECTIONS", "5"))
DB_MAX_CONNECTIONS = int(os.getenv("DB_MAX_CONNECTIONS", "20"))
DB_COMMAND_TIMEOUT = float(os.getenv("DB_COMMAND_TIMEOUT", "60.0"))
DB_MAX_INACTIVE_CONNECTION_LIFETIME = float(
    os.getenv("DB_MAX_INACTIVE_CONNECTION_LIFETIME", "300.0")
)


# -----------------------------------------------------------------------------
# Codec and Connection Initialization
# -----------------------------------------------------------------------------
async def register_codecs(connection: asyncpg.Connection) -> None:
    """
    Registers custom type codecs for automatic conversion:
    - JSON / JSONB <-> Python dict / list
    - Vector text representation for pgvector <-> list[float]
    """
    # Register JSON/JSONB encoders and decoders
    try:
        await connection.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )
        await connection.set_type_codec(
            "json",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )
    except Exception as exc:
        logger.debug("JSON codec registration warning: %s", exc)

    # Optional vector codec support for vector(512)
    # Allows passing Python list[float] or numpy arrays directly
    try:
        def vector_encoder(val: Sequence[float]) -> str:
            if isinstance(val, (list, tuple)):
                return f"[{','.join(str(float(x)) for x in val)}]"
            return str(val)

        def vector_decoder(val: str) -> list[float]:
            # pgvector text format looks like: "[0.123,0.456,...]"
            cleaned = val.strip("[] \t\n\r")
            if not cleaned:
                return []
            return [float(x) for x in cleaned.split(",")]

        await connection.set_type_codec(
            "vector",
            encoder=vector_encoder,
            decoder=vector_decoder,
            schema="public",
            format="text",
        )
    except Exception as exc:
        # Vector type might not exist yet prior to extension initialization
        logger.debug("Vector codec registration deferred until extension exists: %s", exc)


# -----------------------------------------------------------------------------
# Database Manager
# -----------------------------------------------------------------------------
class DatabaseManager:
    """Manages the asyncpg connection pool lifecycle and schema provisioning."""

    def __init__(self, dsn: str = DATABASE_URL) -> None:
        self.dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the connection pool is currently initialized."""
        return self._pool is not None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "Database pool is not initialized. Call 'await db.connect()' first."
            )
        return self._pool

    async def connect(self) -> asyncpg.Pool:
        """Initializes the asyncpg connection pool."""
        if self._pool is not None:
            logger.info("Connection pool already active.")
            return self._pool

        logger.info("Initializing asyncpg connection pool at %s...", self.dsn.split("@")[-1] if "@" in self.dsn else self.dsn)
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=DB_MIN_CONNECTIONS,
            max_size=DB_MAX_CONNECTIONS,
            command_timeout=DB_COMMAND_TIMEOUT,
            max_inactive_connection_lifetime=DB_MAX_INACTIVE_CONNECTION_LIFETIME,
            init=register_codecs,
        )
        logger.info("Database connection pool established successfully.")
        return self._pool

    async def disconnect(self) -> None:
        """Closes the asyncpg connection pool gracefully."""
        if self._pool is not None:
            logger.info("Terminating database connection pool...")
            try:
                await self._pool.close()
            except Exception as exc:
                logger.warning("Error during pool close: %s", exc)
            finally:
                self._pool = None
            logger.info("Database pool closed.")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Context manager to borrow a connection from the pool, auto-connecting if needed."""
        if self._pool is None:
            await self.connect()
        async with self.pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Context manager to acquire a connection wrapped in an atomic transaction."""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args: Any, timeout: Optional[float] = None) -> str:
        """Executes a SQL query and returns the status message."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch(self, query: str, *args: Any, timeout: Optional[float] = None) -> list[asyncpg.Record]:
        """Runs a query and returns the results as a list of Records."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query: str, *args: Any, timeout: Optional[float] = None) -> Optional[asyncpg.Record]:
        """Runs a query and returns the first row as a Record, or None."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(self, query: str, *args: Any, column: int = 0, timeout: Optional[float] = None) -> Any:
        """Runs a query and returns a single value from the first row."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def init_schema(self, schema_path: Optional[Union[str, Path]] = None) -> None:
        """
        Executes schema.sql DDL to ensure extensions, enums, tables,
        and indexes exist on database startup.
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.sql"
        else:
            schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at: {schema_path.resolve()}")

        logger.info("Reading schema definitions from %s...", schema_path.name)
        schema_sql = schema_path.read_text(encoding="utf-8")

        async with self.acquire() as conn:
            # Check DB connectivity
            pg_version = await conn.fetchval("SELECT version();")
            logger.info("Connected to PostgreSQL instance:\n%s", pg_version)

            logger.info("Executing schema migration...")
            # Execute schema script
            await conn.execute(schema_sql)

            # Reload asyncpg type cache for newly created custom types/enums
            try:
                await conn.reload_schema_state()
            except Exception as exc:
                logger.debug("Schema state reload note: %s", exc)

            # Re-register vector codec in case vector extension was newly created
            await register_codecs(conn)

            logger.info(
                "Schema initialized successfully: PostGIS, pgvector, tables & triggers active."
            )

    async def health_check(self) -> dict[str, Any]:
        """Performs a health check ping and returns pool statistics."""
        if not self.is_connected:
            return {
                "status": "disconnected",
                "message": "Database pool is not initialized or database is unreachable.",
            }
        try:
            async with self.acquire() as conn:
                ping_val = await conn.fetchval("SELECT 1;")
                extensions = await conn.fetch(
                    "SELECT extname, extversion FROM pg_extension WHERE extname IN ('postgis', 'vector', 'uuid-ossp');"
                )

            return {
                "status": "healthy" if ping_val == 1 else "unhealthy",
                "active_connections": self.pool.get_size(),
                "idle_connections": self.pool.get_idle_size(),
                "extensions": {rec["extname"]: rec["extversion"] for rec in extensions},
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "error": str(exc),
            }


# Singleton database instance
db = DatabaseManager()


# -----------------------------------------------------------------------------
# Web Framework Lifespan Helper (e.g. FastAPI / Starlette)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: Any = None) -> AsyncGenerator[dict[str, Any], None]:
    """
    Async context manager suitable for FastAPI/Starlette app lifespan:

    Example:
        app = FastAPI(lifespan=lifespan)
    """
    try:
        await db.connect()
        await db.init_schema()
    except Exception as exc:
        logger.warning(
            "Database connection could not be established on startup: %s. "
            "Server will remain online; database operations will retry upon requests.",
            exc,
        )
    try:
        yield {"db": db}
    finally:
        await db.disconnect()


# -----------------------------------------------------------------------------
# Standalone CLI Entrypoint for Testing Initialization
# -----------------------------------------------------------------------------
async def main() -> None:
    logger.info("Starting manual database bootstrap test...")
    try:
        await db.connect()
        await db.init_schema()
        health = await db.health_check()
        logger.info("Health Check Status: %s", json.dumps(health, indent=2))
    except (ConnectionRefusedError, OSError, asyncpg.CannotConnectNowError) as exc:
        logger.warning(
            "Could not connect to PostgreSQL at '%s' (%s). "
            "Please ensure your PostgreSQL server is running and the DATABASE_URL in your .env is set correctly.",
            db.dsn,
            exc,
        )
    except Exception as exc:
        logger.error("Database initialization failed: %s", exc, exc_info=True)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
