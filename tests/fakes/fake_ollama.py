"""A scripted stand-in for the Ollama HTTP API (httpx.MockTransport).

This is a TEST DOUBLE, not a language model: it returns canned JSON so the HTTP client,
schema validation, repair/fallback logic and the LangGraph wiring can be tested offline.
It proves nothing about the quality of a real local model's plans.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

PlannerFn = Callable[[dict[str, Any]], str]


class FakeOllama:
    def __init__(
        self,
        planner: PlannerFn | list[str],
        narrator: PlannerFn | str | None = None,
        models: list[str] | None = None,
        model_missing: bool = False,
    ):
        self.planner = planner
        self.narrator = narrator
        self.models = models or ["qwen2.5:7b"]
        self.model_missing = model_missing
        self.requests: list[dict[str, Any]] = []

    def _next_plan(self, body: dict[str, Any]) -> str:
        if callable(self.planner):
            return self.planner(body)
        return self.planner.pop(0) if len(self.planner) > 1 else self.planner[0]

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0-fake"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            self.requests.append(body)
            if self.model_missing:
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            system = body["messages"][0]["content"]
            if system.startswith("You are the planning module"):
                content = self._next_plan(body)
            else:
                n = self.narrator
                content = (
                    n(body)
                    if callable(n)
                    else (
                        n
                        or json.dumps(
                            {
                                "headline": "Narrated by fake model",
                                "summary": "Summary from the fake narrator.",
                                "findings": [{"kind": "observation", "text": "A fake observation."}],
                                "limitations": [],
                                "suggested_follow_ups": [],
                            }
                        )
                    )
                )
            return httpx.Response(
                200, json={"model": body["model"], "message": {"role": "assistant", "content": content}}
            )
        return httpx.Response(404, json={"error": "not found"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def unreachable_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(handler)
