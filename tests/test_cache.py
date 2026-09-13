"""Tests for wee.cache.Cache and the @cached decorator."""
import time

from wee.cache import Cache, cached


# ------------------------------------------------------------------
# test_set_get
# ------------------------------------------------------------------


def test_set_get(tmp_path):
    """Basic set/get round-trip."""
    db_path = str(tmp_path / "cache.sqlite")
    c = Cache(path=db_path)

    c.set("greeting", "hello")
    assert c.get("greeting") == "hello"

    # Overwrite
    c.set("greeting", {"msg": "world"})
    assert c.get("greeting") == {"msg": "world"}

    # Missing key
    assert c.get("nonexistent") is None

    c.close()


# ------------------------------------------------------------------
# test_ttl_expiry
# ------------------------------------------------------------------


def test_ttl_expiry(tmp_path):
    """Set with a short TTL, verify it expires."""
    db_path = str(tmp_path / "cache_ttl.sqlite")
    c = Cache(path=db_path)

    c.set("ephemeral", 42, ttl_seconds=0.05)
    # Should be available immediately
    assert c.get("ephemeral") == 42

    # Wait for expiry
    time.sleep(0.1)
    assert c.get("ephemeral") is None

    c.close()


# ------------------------------------------------------------------
# test_cached_decorator
# ------------------------------------------------------------------


def test_cached_decorator(tmp_path):
    """Verify the @cached decorator caches function results."""
    db_path = str(tmp_path / "cache_dec.sqlite")
    c = Cache(path=db_path)

    call_count = 0

    @cached(c, namespace="test", key_fn=lambda x: str(x))
    def expensive_fn(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call -- should compute
    assert expensive_fn(5) == 10
    assert call_count == 1

    # Second call with same arg -- should hit cache
    assert expensive_fn(5) == 10
    assert call_count == 1  # no additional computation

    # Different arg -- should compute again
    assert expensive_fn(7) == 14
    assert call_count == 2

    c.close()
