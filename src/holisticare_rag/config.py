"""Settings from environment variables and a local ``.env`` file. Every value has an offline default."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read ``KEY=VALUE`` lines. Empty values are skipped. Real environment variables win."""
    path = Path(path)
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                out[k.strip()] = v.strip()
    return out


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path("data/documents")
    index_dir: Path = Path(".index")
    chunk_chars: int = 900
    chunk_overlap: int = 120
    embedder: str = "hashing"  # hashing (offline), sentence-transformers, ollama
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    min_coverage: float = 0.5
    history_turns: int = 6
    llm: str = "extractive"  # extractive (offline), ollama, openai
    llm_model: str = "llama3.1"
    llm_base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.embedder not in ("hashing", "sentence-transformers", "ollama"):
            raise ValueError("HOLISTICARE_EMBEDDER must be hashing, sentence-transformers or ollama")
        if self.llm not in ("extractive", "ollama", "openai"):
            raise ValueError("HOLISTICARE_LLM must be extractive, ollama or openai")
        if not 0 <= self.temperature <= 0.3:
            raise ValueError("temperature must be between 0 and 0.3 for medical information")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        if self.chunk_overlap >= self.chunk_chars or self.chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200 and larger than chunk_overlap")
        if self.top_k < 1 or self.history_turns < 0 or self.max_tokens < 64:
            raise ValueError("top_k >= 1, history_turns >= 0 and max_tokens >= 64 are necessary")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        e = {**read_dotenv(), **os.environ} if env is None else env
        d = cls()
        return cls(
            docs_dir=Path(e.get("HOLISTICARE_DOCS_DIR", str(d.docs_dir))),
            index_dir=Path(e.get("HOLISTICARE_INDEX_DIR", str(d.index_dir))),
            chunk_chars=int(e.get("HOLISTICARE_CHUNK_CHARS", d.chunk_chars)),
            chunk_overlap=int(e.get("HOLISTICARE_CHUNK_OVERLAP", d.chunk_overlap)),
            embedder=e.get("HOLISTICARE_EMBEDDER", d.embedder),
            embed_model=e.get("HOLISTICARE_EMBED_MODEL", d.embed_model),
            top_k=int(e.get("HOLISTICARE_TOP_K", d.top_k)),
            min_coverage=float(e.get("HOLISTICARE_MIN_COVERAGE", d.min_coverage)),
            history_turns=int(e.get("HOLISTICARE_HISTORY_TURNS", d.history_turns)),
            llm=e.get("HOLISTICARE_LLM", d.llm),
            llm_model=e.get("HOLISTICARE_LLM_MODEL", d.llm_model),
            llm_base_url=e.get("HOLISTICARE_LLM_BASE_URL") or None,
            temperature=float(e.get("HOLISTICARE_TEMPERATURE", d.temperature)),
            max_tokens=int(e.get("HOLISTICARE_MAX_TOKENS", d.max_tokens)),
            timeout_s=float(e.get("HOLISTICARE_TIMEOUT_S", d.timeout_s)),
        )
