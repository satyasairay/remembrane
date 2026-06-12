import pytest

from remembrane import MemoryStore
from remembrane.adapters import RemembraneChatMemory, RemembraneStorage


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


class TestLangChainAdapter:
    def test_save_and_load(self, store):
        mem = RemembraneChatMemory(store, session_id="s1")
        mem.save_context({"input": "my favorite color is teal"}, {"output": "Noted, teal it is."})
        mem.save_context({"input": "I work at Acme Corp"}, {"output": "Got it."})
        out = mem.load_memory_variables({"input": "what color do I like?"})
        assert "teal" in out["history"]

    def test_load_without_query_returns_recent(self, store):
        mem = RemembraneChatMemory(store, session_id="s2", k=2)
        mem.save_context({"input": "first"}, {"output": "one"})
        out = mem.load_memory_variables({})
        assert "first" in out["history"]

    def test_clear(self, store):
        mem = RemembraneChatMemory(store, session_id="s3")
        mem.save_context({"input": "to be cleared"}, {"output": "ok"})
        mem.clear()
        assert store.count("s3") == 0

    def test_sessions_isolated(self, store):
        a = RemembraneChatMemory(store, session_id="user-a")
        b = RemembraneChatMemory(store, session_id="user-b")
        a.save_context({"input": "secret of user a"}, {"output": "ok"})
        out = b.load_memory_variables({"input": "secret"})
        assert "user a" not in out["history"]


class TestCrewAIAdapter:
    def test_save_and_search(self, store):
        storage = RemembraneStorage(store)
        storage.save("the deadline is next friday", metadata={"task": "planning"})
        results = storage.search("when is the deadline?")
        assert results
        assert results[0]["context"] == "the deadline is next friday"
        assert results[0]["metadata"]["task"] == "planning"
        assert "score" in results[0]

    def test_reset(self, store):
        storage = RemembraneStorage(store)
        storage.save("temp")
        storage.reset()
        assert storage.search("temp") == []
