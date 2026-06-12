"""Verified against real langchain-core (skipped when not installed)."""
import pytest

lc = pytest.importorskip("langchain_core")

from remembrane import MemoryStore  # noqa: E402
from remembrane.adapters import RemembraneChatMessageHistory  # noqa: E402


def test_is_real_base_chat_message_history():
    from langchain_core.chat_history import BaseChatMessageHistory
    store = MemoryStore(":memory:")
    hist = RemembraneChatMessageHistory(store)
    assert isinstance(hist, BaseChatMessageHistory)
    store.close()


def test_roundtrip_messages():
    from langchain_core.messages import AIMessage, HumanMessage
    store = MemoryStore(":memory:")
    hist = RemembraneChatMessageHistory(store, "s1")
    hist.add_messages([HumanMessage(content="hi"), AIMessage(content="hello!")])
    msgs = hist.messages
    assert [m.type for m in msgs] == ["human", "ai"]
    hist.clear()
    assert hist.messages == []
    store.close()


def test_runnable_with_message_history():
    """The exact integration that failed in the round-2 audit."""
    import warnings
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store = MemoryStore(":memory:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain = RunnableWithMessageHistory(
            RunnableLambda(lambda x: AIMessage(content="ack")),
            lambda session_id: RemembraneChatMessageHistory(store, session_id),
        )
        chain.invoke([HumanMessage(content="hello")],
                     config={"configurable": {"session_id": "s2"}})
    assert len(RemembraneChatMessageHistory(store, "s2").messages) == 2
    store.close()
