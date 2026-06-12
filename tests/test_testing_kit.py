import pytest

from remembrane import MemoryStore
from remembrane.testing import (
    assert_not_recalls,
    assert_recalls,
    assert_recalls_first,
    recalled_contents,
)


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    s.store("user prefers dark mode", importance=0.8)
    s.store("user is allergic to peanuts", importance=1.0)
    yield s
    s.close()


def test_assert_recalls_passes(store):
    assert_recalls(store, "what theme does the user like?", "dark mode")


def test_assert_recalls_fails_helpfully(store):
    with pytest.raises(AssertionError) as exc:
        assert_recalls(store, "what theme does the user like?", "light mode")
    assert "Recalled instead" in str(exc.value)


def test_assert_recalls_first(store):
    assert_recalls_first(store, "any food allergies?", "peanuts")


def test_assert_recalls_first_fails_on_wrong_top(store):
    with pytest.raises(AssertionError):
        assert_recalls_first(store, "peanut allergy info", "dark mode")


def test_assert_not_recalls(store):
    assert_not_recalls(store, "what theme does the user like?", "nonexistent topic", k=1)


def test_recalled_contents_does_not_touch(store):
    recalled_contents(store, "dark mode")
    assert all(m.access_count == 0 for m in store.all())


def test_empty_store_fails_cleanly():
    s = MemoryStore(":memory:")
    with pytest.raises(AssertionError):
        assert_recalls_first(s, "anything", "something")
    s.close()
