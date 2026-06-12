"""remembrane — local-first persistent memory for AI agents."""
from .embedders import HashEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder, cosine_similarity
from .models import JournalEntry, Memory, RecallResult
from .scoring import ScoringConfig
from .store import MemoryStore

__version__ = "0.2.0"
__all__ = [
    "MemoryStore",
    "Memory",
    "RecallResult",
    "JournalEntry",
    "ScoringConfig",
    "HashEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
]
