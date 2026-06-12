"""remembrane — local-first persistent memory for AI agents."""
from .embedders import HashEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder, cosine_similarity
from .models import Memory, RecallResult
from .scoring import ScoringConfig
from .store import MemoryStore

__version__ = "0.1.0"
__all__ = [
    "MemoryStore",
    "Memory",
    "RecallResult",
    "ScoringConfig",
    "HashEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
]
