"""Pipeline configuration with environment variable loading and feature toggles."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _env(key: str, default: Any = None, cast: type = str) -> Any:
    """Read an environment variable with optional type casting."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        if cast is bool:
            return raw.lower() in ("1", "true", "yes", "on")
        return cast(raw)
    except (ValueError, TypeError):
        return default


DEFAULT_FEATURE_TOGGLES: dict[str, bool] = {
    "dense_retrieval": True,
    "bm25_retrieval": True,
    "hybrid_retrieval": True,
    "cross_encoder_reranking": True,
    "metadata_filtering": True,
    "hyde": False,
    "conversation_memory": True,
    "query_rewriting": True,
    "agent_planning": True,
    "multi_hop_retrieval": False,
    "chunk_enhancement": False,
}


@dataclass
class PipelineConfig:
    """Central configuration for the entire RAG pipeline.

    Values are loaded from environment variables when available, falling back
    to the defaults declared here.  Feature toggles can be updated at runtime
    via :func:`update_feature_toggle`.
    """

    # -- LLM ------------------------------------------------------------------
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 1024

    # -- Embeddings ------------------------------------------------------------
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768

    # -- Chunking --------------------------------------------------------------
    chunk_strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # -- Retrieval -------------------------------------------------------------
    retrieval_top_k: int = 20
    similarity_threshold: float = 0.3
    retriever_type: str = "hybrid"
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3

    # -- Reranking -------------------------------------------------------------
    reranking_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_top_k: int = 5

    # -- Agent -----------------------------------------------------------------
    agent_enabled: bool = True
    agent_max_iterations: int = 3

    # -- HyDE ------------------------------------------------------------------
    hyde_enabled: bool = False

    # -- Memory ----------------------------------------------------------------
    memory_enabled: bool = True
    memory_max_turns: int = 10

    # -- Vector DB -------------------------------------------------------------
    vector_db_path: str = "data/chroma_db"
    vector_db_collection: str = "placement_rag"

    # -- Feature toggles -------------------------------------------------------
    feature_toggles: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_FEATURE_TOGGLES))

    # -- API key (populated from env) ------------------------------------------
    gemini_api_key: str = ""

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Create a config instance populated from environment variables."""
        for candidate in [Path(".env"), _PROJECT_ROOT / ".env"]:
            if candidate.exists():
                try:
                    from dotenv import load_dotenv
                    load_dotenv(candidate)
                except ImportError:
                    for line in candidate.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if not os.getenv(k.strip()):
                                os.environ[k.strip()] = v.strip()
                break

        cfg = cls(
            llm_provider=_env("RAG_LLM_PROVIDER", "gemini"),
            llm_model=_env("RAG_LLM_MODEL", "gemini-2.5-flash"),
            temperature=_env("RAG_TEMPERATURE", 0.3, float),
            max_output_tokens=_env("RAG_MAX_OUTPUT_TOKENS", 1024, int),
            embedding_model=_env("RAG_EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dimension=_env("RAG_EMBEDDING_DIMENSION", 768, int),
            chunk_strategy=_env("RAG_CHUNK_STRATEGY", "recursive"),
            chunk_size=_env("RAG_CHUNK_SIZE", 500, int),
            chunk_overlap=_env("RAG_CHUNK_OVERLAP", 50, int),
            retrieval_top_k=_env("RAG_RETRIEVAL_TOP_K", 20, int),
            similarity_threshold=_env("RAG_SIMILARITY_THRESHOLD", 0.3, float),
            retriever_type=_env("RAG_RETRIEVER_TYPE", "hybrid"),
            hybrid_dense_weight=_env("RAG_HYBRID_DENSE_WEIGHT", 0.7, float),
            hybrid_sparse_weight=_env("RAG_HYBRID_SPARSE_WEIGHT", 0.3, float),
            reranking_enabled=_env("RAG_RERANKING_ENABLED", True, bool),
            reranker_model=_env("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            reranking_top_k=_env("RAG_RERANKING_TOP_K", 5, int),
            agent_enabled=_env("RAG_AGENT_ENABLED", True, bool),
            agent_max_iterations=_env("RAG_AGENT_MAX_ITERATIONS", 3, int),
            hyde_enabled=_env("RAG_HYDE_ENABLED", False, bool),
            memory_enabled=_env("RAG_MEMORY_ENABLED", True, bool),
            memory_max_turns=_env("RAG_MEMORY_MAX_TURNS", 10, int),
            vector_db_path=_env("RAG_VECTOR_DB_PATH", "data/chroma_db"),
            vector_db_collection=_env("RAG_VECTOR_DB_COLLECTION", "placement_rag"),
            gemini_api_key=_env("GEMINI_API_KEY", "") or _env("VITE_GEMINI_API_KEY", "")  )
        # Merge any env-based feature overrides
        for feature in DEFAULT_FEATURE_TOGGLES:
            env_key = f"RAG_FEATURE_{feature.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                cfg.feature_toggles[feature] = env_val.lower() in ("1", "true", "yes", "on")
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load config from a YAML file, then overlay environment variables."""
        p = Path(path)
        if not p.exists():
            logger.warning("Config YAML not found at %s — using defaults + env", p)
            return cls.from_env()

        with open(p, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        flat: dict[str, Any] = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
            # top-level scalars are kept as-is
        # Create with YAML values, env overrides take priority (via from_env logic)
        cfg = cls.from_env()
        for key, value in flat.items():
            if hasattr(cfg, key) and os.getenv(f"RAG_{key.upper()}") is None:
                setattr(cfg, key, value)
        return cfg

    @property
    def vector_db_full_path(self) -> Path:
        """Resolve vector DB path relative to project root."""
        p = Path(self.vector_db_path)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p

    def is_feature_enabled(self, feature: str) -> bool:
        """Check whether a feature toggle is enabled."""
        return self.feature_toggles.get(feature, False)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_config_lock = threading.Lock()
_config_instance: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    """Return the global singleton :class:`PipelineConfig`.

    Thread-safe.  On first call, loads from environment variables.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                yaml_path = _PROJECT_ROOT / "backend" / "config" / "config.yaml"
                _config_instance = PipelineConfig.from_yaml(yaml_path)
                logger.info("PipelineConfig loaded (model=%s, retriever=%s)",
                            _config_instance.llm_model,
                            _config_instance.retriever_type)
    return _config_instance


def update_feature_toggle(feature: str, enabled: bool) -> None:
    """Update a feature toggle at runtime.

    Args:
        feature: The feature name (must be a known toggle).
        enabled: Whether to enable or disable the feature.

    Raises:
        KeyError: If *feature* is not a recognised toggle name.
    """
    cfg = get_config()
    if feature not in cfg.feature_toggles:
        raise KeyError(
            f"Unknown feature toggle '{feature}'. "
            f"Valid toggles: {sorted(cfg.feature_toggles)}"
        )
    cfg.feature_toggles[feature] = enabled
    logger.info("Feature toggle '%s' set to %s", feature, enabled)
