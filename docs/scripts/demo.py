"""The README demo: a real paraphrase hit, run against the local offline stack.

`docs/scripts/demo.sh` executes this file and freezes the output into
`docs/assets/demo.png`. Every printed value is the live result of the call above it.
"""

import asyncio

import semisweet


async def main() -> None:
    cache = semisweet.SemanticCache(namespace="capitals")

    stored = "what is the capital of france"
    await cache.set(semisweet.CacheQuery(query=stored), {"answer": "paris"})
    print(f'set("{stored}")')

    reworded = "france's capital?"
    hit = await cache.get(semisweet.CacheQuery(query=reworded))
    print(f'get("{reworded}")')
    print(f"-> {hit!r}  # semantic hit, nothing recomputed")


asyncio.run(main())
