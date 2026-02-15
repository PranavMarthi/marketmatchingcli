"""Database engine and session configuration."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from marketmap.config import settings

# Async engine for FastAPI
async_engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=20,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@lru_cache(maxsize=1)
def _get_sync_engine():
    """Lazy-create sync engine (only when Celery workers need it)."""
    return create_engine(
        settings.database_url_sync,
        echo=False,
        pool_size=10,
        max_overflow=5,
    )


def SyncSessionLocal() -> Session:
    """Create a sync session for Celery workers."""
    engine = _get_sync_engine()
    factory = sessionmaker(bind=engine)
    return factory()


async def get_async_session() -> AsyncSession:  # type: ignore[misc]
    """Dependency for FastAPI endpoints."""
    async with AsyncSessionLocal() as session:
        yield session
