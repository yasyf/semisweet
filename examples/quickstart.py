"""Minimal typed usage sample — also the mypy target proving the shipped stubs resolve."""

import semisweet


def roundtrip() -> bytes | None:
    cache = semisweet.SemanticCache(
        namespace="demo",
        embedding=semisweet.LocalEmbedding(),
        vectors=semisweet.MemoryVectors(),
        storage=semisweet.DiskStorage(),
        scoring=semisweet.Scoring(base=0.9, floor=0.86),
    )
    result: bytes | None = None
    with cache:
        accepted: bool = cache.set(
            semisweet.CacheQuery(query="what is aspirin", keys={"patient1"}),
            b"a painkiller",
        )
        assert accepted
        result = cache.get(semisweet.CacheQuery(query="what is aspirin", keys={"patient1"}))
    return result


def handle_errors() -> None:
    try:
        semisweet.CacheQuery(query="")
    except semisweet.ConfigError:
        pass
    except ValueError:
        pass


if __name__ == "__main__":
    print(roundtrip())
