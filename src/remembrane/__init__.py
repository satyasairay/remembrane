"""remembrane — local-first persistent memory for AI agents."""
from .conflicts import Conflict
from .embedders import HashEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder, cosine_similarity
from .models import JournalEntry, Memory, RecallResult
from .scoring import ScoringConfig
from .store import MemoryStore

__version__ = "0.5.0"
__all__ = [
    "MemoryStore",
    "Memory",
    "RecallResult",
    "JournalEntry",
    "Conflict",
    "ScoringConfig",
    "HashEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
]
