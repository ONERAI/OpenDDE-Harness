"""Round-trip cost of the pi-ai model service over stdio, against faux/echo.

    PYTHONPATH=. python scripts/model_service_latency.py [streams] [spawns]

Prints p50/p95 of: spawn to first event (cold, one stream per spawn);
request to first event and inter-event gap (warm, one process).
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

from opendde_harness.providers.model_service import ModelService

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
CONTEXT = {"messages": [{"role": "user", "content": "latency", "timestamp": 1}]}
ENV = {"OPENDDE_MODEL_SERVICE_FAUX": "1"}


def pct(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100)[int(p) - 1] if len(values) > 1 else values[0]


async def cold(spawns: int) -> list[float]:
    out = []
    for _ in range(spawns):
        svc = ModelService(bundle=BUNDLE, env=ENV)
        t0 = time.perf_counter()
        await svc.start()
        _id, events = await svc.stream_with_id("faux", "echo", CONTEXT)
        await anext(events)
        out.append((time.perf_counter() - t0) * 1000)
        await events.aclose()
        await svc.close()
    return out


async def warm(streams: int) -> tuple[list[float], list[float]]:
    svc = ModelService(bundle=BUNDLE, env=ENV)
    await svc.start()
    first, gaps = [], []
    for _ in range(streams):
        t0 = time.perf_counter()
        last = t0
        async for i, _event in aenumerate(svc.stream("faux", "echo", CONTEXT)):
            now = time.perf_counter()
            if i == 0:
                first.append((now - t0) * 1000)
            else:
                gaps.append((now - last) * 1000)
            last = now
    await svc.close()
    return first, gaps


async def aenumerate(it):
    i = 0
    async for x in it:
        yield i, x
        i += 1


async def main() -> None:
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    spawns = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    c = await cold(spawns)
    f, g = await warm(streams)
    print(f"cold spawn->first event ({spawns} spawns): p50 {pct(c, 50):.1f} ms  p95 {pct(c, 95):.1f} ms")
    print(f"warm request->first event ({streams} streams): p50 {pct(f, 50):.2f} ms  p95 {pct(f, 95):.2f} ms")
    print(f"warm inter-event gap ({len(g)} gaps): p50 {pct(g, 50):.3f} ms  p95 {pct(g, 95):.3f} ms")


if __name__ == "__main__":
    asyncio.run(main())
