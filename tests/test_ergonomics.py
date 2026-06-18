"""``__repr__``, value-equality, and hashing for the frozen config/value classes.

All pure config: no daemon, no model download. ``SemanticCache`` is intentionally
excluded — it is not frozen and has no value equality, only identity.
"""

from semisweet import (
    CacheQuery,
    LocalEmbedding,
    Scoring,
    VoyageEmbedding,
)

# --- __repr__ is non-default and informative ---


def test_voyage_embedding_repr_names_class_and_fields():
    text = repr(VoyageEmbedding(model="voyage-3", dim=8))
    assert "VoyageEmbedding" in text
    assert "voyage-3" in text
    assert "8" in text


def test_local_embedding_repr_names_class_and_model():
    text = repr(LocalEmbedding(model="bge-small"))
    assert "LocalEmbedding" in text
    assert "bge-small" in text


def test_cache_query_repr_names_class_query_and_key():
    text = repr(CacheQuery(query="capital of france", keys={"v1"}))
    assert "CacheQuery" in text
    assert "capital of france" in text
    assert "v1" in text


def test_scoring_repr_lists_fields():
    text = repr(Scoring(top_k=5))
    assert "Scoring" in text
    assert "top_k=5" in text


# --- value equality + hashing for frozen classes ---


def test_identical_local_embeddings_are_equal_and_hash_equal():
    a = LocalEmbedding()
    b = LocalEmbedding()
    assert a == b
    assert hash(a) == hash(b)


def test_local_embeddings_with_different_models_are_unequal():
    assert LocalEmbedding(model="a") != LocalEmbedding(model="b")


def test_cache_queries_with_same_query_and_keys_are_equal_and_hash_equal():
    # Keys are an unordered set: insertion order must not affect equality or hash.
    a = CacheQuery(query="what dose", keys={"v1", "v2"})
    b = CacheQuery(query="what dose", keys={"v2", "v1"})
    assert a == b
    assert hash(a) == hash(b)


def test_cache_queries_with_different_keys_are_unequal():
    assert CacheQuery(query="what dose", keys={"v1"}) != CacheQuery(
        query="what dose", keys={"v2"}
    )


def test_cache_queries_with_different_query_text_are_unequal():
    assert CacheQuery(query="dose a") != CacheQuery(query="dose b")


def test_scoring_equality_ignores_construction_order_and_hashes_equal():
    a = Scoring(threshold=0.9, context_gate=0.2, top_k=5)
    b = Scoring(threshold=0.9, context_gate=0.2, top_k=5)
    assert a == b
    assert hash(a) == hash(b)
