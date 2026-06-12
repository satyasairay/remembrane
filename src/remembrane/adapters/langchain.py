"""LangChain adapters.

Two integration surfaces:

- :func:`RemembraneChatMessageHistory` — for modern LangChain (>=0.3 / 1.x),
  returns a ``BaseChatMessageHistory`` subclass for use with
  ``RunnableWithMessageHistory``. Requires ``langchain-core`` (imported lazily;
  remembrane itself stays dependency-free).
- :class:`RemembraneChatMemory` — duck-typed against the legacy memory
  interface (``load_memory_variables`` / ``save_context``), needs no langchain
  install at all. Kept for pre-1.x codebases.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..store import MemoryStore


def RemembraneChatMessageHistory(store: MemoryStore, session_id: str = "default"):
    """Persistent, remembrane-backed chat history for current LangChain.

    Example::

        from langchain_core.runnables.history import RunnableWithMessageHistory
        from remembrane import MemoryStore
        from remembrane.adapters import RemembraneChatMessageHistory

        store = MemoryStore("agent.db")
        chain = RunnableWithMessageHistory(
            runnable,
            lambda session_id: RemembraneChatMessageHistory(store, session_id),
        )

    Requires langchain-core: ``pip install langchain-core``.
    """
    try:
        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
        )
    except ImportError as exc:
        raise ImportError(
            "RemembraneChatMessageHistory requires langchain-core. "
            "Run: pip install langchain-core"
        ) from exc

    _BY_ROLE = {"ai": AIMessage, "human": HumanMessage, "system": SystemMessage}

    class _RemembraneHistory(BaseChatMessageHistory):
        def __init__(self, _store: MemoryStore, _session: str):
            self._store = _store
            self._session = _session

        @property
        def messages(self) -> List[BaseMessage]:
            out: List[BaseMessage] = []
            for m in self._store.all(self._session):
                role = m.metadata.get("role", "human")
                out.append(_BY_ROLE.get(role, HumanMessage)(content=m.content))
            return out

        def add_message(self, message: BaseMessage) -> None:
            content = message.content
            if isinstance(content, list):  # structured content blocks: keep the text parts
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ).strip()
            elif not isinstance(content, str):
                content = str(content)
            if not content.strip():
                return
            role = getattr(message, "type", "human")
            self._store.store(content, namespace=self._session, metadata={"role": role})

        def add_messages(self, messages) -> None:
            for m in messages:
                self.add_message(m)

        def clear(self) -> None:
            self._store.forget(namespace=self._session)

    return _RemembraneHistory(store, session_id)


class RemembraneChatMemory:
    """Legacy LangChain memory interface (pre-1.x), duck-typed — no install needed.

    Retrieves the exchanges most relevant to the *current input* instead of
    replaying the transcript, so context stays small as history grows.
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
        human = str(inputs.get(self.input_key, "")).strip()
        ai = str(outputs.get(self.output_key, "")).strip()
        if human:
            self.store.store(human, namespace=self.session_id, metadata={"role": "human"})
        if ai:
            self.store.store(ai, namespace=self.session_id, metadata={"role": "ai"})

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
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
