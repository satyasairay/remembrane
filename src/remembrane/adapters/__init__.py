"""Framework adapters for remembrane.

Adapters are duck-typed (or lazily imported) against their target frameworks so
remembrane never hard-depends on them.
"""
from .crewai import RemembraneStorage
from .langchain import RemembraneChatMemory, RemembraneChatMessageHistory

__all__ = ["RemembraneChatMemory", "RemembraneChatMessageHistory", "RemembraneStorage"]
