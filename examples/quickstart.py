"""Minimal typed usage sample — also the type-checker target proving the API resolves."""

import asyncio

import semisweet


async def roundtrip() -> object:
    cache = semisweet.SemanticCache(
        namespace="demo",
        embedding=semisweet.LocalEmbedding(),
        vectors=semisweet.MemoryVectors(),
        storage=semisweet.DiskStorage(),
        scoring=semisweet.Scoring(base=0.9, floor=0.86),
    )
    query = semisweet.CacheQuery(query="what is aspirin", keys={"patient1"})
    accepted: bool = await cache.set(query, {"drug": "aspirin", "class": "NSAID"})
    assert accepted
    return await cache.get(query)


@semisweet.cache(query="question")
async def describe(question: str) -> dict[str, str]:
    return {"drug": "aspirin", "class": "NSAID"}  # stand-in for an expensive call


def handle_errors() -> None:
    try:
        semisweet.CacheQuery(query="")
    except semisweet.ConfigError:
        pass


async def main() -> None:
    print(await roundtrip())
    print(await describe("what is aspirin"))
    handle_errors()


if __name__ == "__main__":
    asyncio.run(main())
