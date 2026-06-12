"""Framework adapters for remembrane.

Adapters are duck-typed against their target frameworks so remembrane never
hard-depends on them. Import errors only occur if you actually use an adapter
without its framework installed.
"""
from .crewai import RemembraneStorage
from .langchain import RemembraneChatMemory

__all__ = ["RemembraneChatMemory", "RemembraneStorage"]
