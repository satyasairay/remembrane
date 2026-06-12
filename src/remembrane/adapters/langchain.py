"""LangChain-compatible memory adapter.

Duck-typed against LangChain's memory interface (load_memory_variables /
save_context / clear), so it works without importing langchain itself.

Example::

    from remembrane import MemoryStore
    from remembrane.adapters import RemembraneChatMemory

    memory = RemembraneChatMemory(MemoryStore("agent.db"), session_id="user-42")
    # use anywhere LangChain expects a memory object,
    # or call .save_context() / .load_memory_variables() directly.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..store import MemoryStore


class RemembraneChatMemory:
    """Persistent, semantically-recalled conversation memory.

    Instead of replaying the full transcript, ``load_memory_variables`` returns
    the memories most relevant to the *current input*, ranked by
    similarity x recency x importance.
    """

    memory_key: str = "history"

    def __init__(
        self,
        store: MemoryStore,
        session_id: str = "default",
        k: int = 6,
        input_key: str = "input",
        output_key: str = "output",
    ):
        self.store = store
        self.session_id = session_id
        self.k = k
        self.input_key = input_key
        self.output_key = output_key

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Persist one exchange as two memories (human + ai)."""
        human = str(inputs.get(self.input_key, "")).strip()
        ai = str(outputs.get(self.output_key, "")).strip()
        if human:
            self.store.store(human, namespace=self.session_id, metadata={"role": "human"})
        if ai:
            self.store.store(ai, namespace=self.session_id, metadata={"role": "ai"})

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return the k memories most relevant to the current input."""
        query = str(inputs.get(self.input_key, "")).strip()
        if not query:
            memories = self.store.all(self.session_id)[-self.k :]
            lines = [f"{m.metadata.get('role', '?')}: {m.content}" for m in memories]
        else:
            results = self.store.recall(query, k=self.k, namespace=self.session_id)
            lines = [
                f"{r.memory.metadata.get('role', '?')}: {r.memory.content}" for r in results
            ]
        return {self.memory_key: "\n".join(lines)}

    def clear(self) -> None:
        self.store.forget(namespace=self.session_id)
