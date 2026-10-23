"""LLM clients. Default: a local Ollama server.

The LLM is used for two bounded jobs -- writing a JSON *plan* of tool calls, and writing a
JSON *answer* from computed results. Outputs are validated with Pydantic; invalid output is
retried once with the validation error, then the caller falls back to deterministic logic.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agentic_gis.config.settings import Settings
from agentic_gis.connectors.base import ConnectionStatus
from agentic_gis.security.policy import check_llm_endpoint, is_local_endpoint

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    """The model server is unreachable, the model is missing, or the call timed out."""


class LLMOutputError(LLMError):
    """The model returned output that does not match the required schema."""


@dataclass
class LLMCallRecord:
    purpose: str
    model: str
    ok: bool
    elapsed_ms: int
    prompt_chars: int
    response_chars: int = 0
    attempts: int = 1
    error: str | None = None


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref`` definitions (some grammar converters handle refs poorly)."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 20:
            return node
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                target = copy.deepcopy(defs.get(node["$ref"].split("/")[-1], {}))
                rest = {k: v for k, v in node.items() if k != "$ref"}
                return walk({**target, **rest}, depth + 1)
            return {k: walk(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, depth + 1) for v in node]
        return node

    return walk(schema)


def extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise LLMOutputError("No JSON object in model output")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise LLMOutputError("Unterminated JSON object in model output")


class LLMClient:
    provider = "base"

    def __init__(self, model: str, base_url: str, timeout_s: float, temperature: float):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.calls: list[LLMCallRecord] = []

    @property
    def is_local(self) -> bool:
        return is_local_endpoint(self.base_url)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        raise NotImplementedError

    def test_connection(self) -> ConnectionStatus:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []

    def generate_json(self, system: str, user: str, model_cls: type[T], purpose: str, retries: int = 1) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        schema = inline_refs(model_cls.model_json_schema())
        t0 = time.perf_counter()
        last_err = ""
        raw = ""
        for attempt in range(retries + 1):
            raw = self.chat(messages, schema=schema)
            try:
                obj = model_cls.model_validate(extract_json(raw))
                self.calls.append(
                    LLMCallRecord(
                        purpose,
                        self.model,
                        True,
                        int((time.perf_counter() - t0) * 1000),
                        sum(len(m["content"]) for m in messages),
                        len(raw),
                        attempt + 1,
                    )
                )
                return obj
            except (LLMOutputError, ValidationError, json.JSONDecodeError) as exc:
                last_err = str(exc)[:800]
                messages = messages + [
                    {"role": "assistant", "content": raw[:6000]},
                    {
                        "role": "user",
                        "content": f"That output was invalid: {last_err}\nReturn ONLY corrected JSON "
                        "that matches the schema.",
                    },
                ]
        self.calls.append(
            LLMCallRecord(
                purpose,
                self.model,
                False,
                int((time.perf_counter() - t0) * 1000),
                sum(len(m["content"]) for m in messages),
                len(raw),
                retries + 1,
                last_err,
            )
        )
        raise LLMOutputError(f"Model output failed validation after {retries + 1} attempts: {last_err[:300]}")


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 240,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        think: bool | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        super().__init__(model, base_url, timeout_s, temperature)
        self.num_ctx = num_ctx
        self.think = think
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
            trust_env=not is_local_endpoint(base_url),
        )

    def _unavailable(self, exc: Exception) -> LLMUnavailableError:
        if isinstance(exc, httpx.TimeoutException):
            return LLMUnavailableError(
                f"Ollama at {self.base_url} timed out after {self.timeout_s:.0f}s "
                "(try a smaller model or increase AGIS_LLM_TIMEOUT_S)."
            )
        return LLMUnavailableError(
            f"Ollama is not reachable at {self.base_url}. Start the Ollama app or run `ollama serve`."
        )

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            "keep_alive": "15m",
        }
        if schema is not None:
            body["format"] = schema
        if self.think is not None:
            body["think"] = self.think
        try:
            r = self._client.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise self._unavailable(exc) from exc
        if r.status_code == 404 and "not found" in r.text.lower():
            raise LLMUnavailableError(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}")
        if r.status_code >= 400:
            raise LLMUnavailableError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            return str(r.json()["message"]["content"])
        except (KeyError, ValueError) as exc:
            raise LLMOutputError("Unexpected response format from Ollama") from exc

    def list_models(self) -> list[str]:
        try:
            r = self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            return sorted(m.get("name", "") for m in r.json().get("models", []))
        except (httpx.HTTPError, ValueError):
            return []

    def test_connection(self) -> ConnectionStatus:
        try:
            v = self._client.get("/api/version", timeout=5.0)
            v.raise_for_status()
            version = v.json().get("version", "unknown")
        except (httpx.HTTPError, ValueError) as exc:
            return ConnectionStatus(False, "ollama", str(self._unavailable(exc)))
        models = self.list_models()
        names = set(models) | {m.removesuffix(":latest") for m in models}
        if self.model not in names and f"{self.model}:latest" not in names:
            return ConnectionStatus(
                False,
                "ollama",
                f"Ollama {version} is running but model '{self.model}' is not "
                f"installed. Run: ollama pull {self.model}",
                {"version": version, "models": models},
            )
        return ConnectionStatus(
            True,
            "ollama",
            f"Ollama {version} ready with model {self.model}",
            {"version": version, "models": models},
        )


class OpenAICompatibleClient(LLMClient):
    """For organization-approved, OpenAI-compatible servers (vLLM, llama.cpp server, LM Studio ...)."""

    provider = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 240,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ):
        super().__init__(model, base_url, timeout_s, temperature)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            headers=headers,
            transport=transport,
        )

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": False},
            }
        try:
            r = self._client.post("/chat/completions", json=body)
            if r.status_code == 400 and schema is not None:  # server without json_schema support
                body["response_format"] = {"type": "json_object"}
                r = self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"LLM endpoint {self.base_url} is not reachable: {type(exc).__name__}"
            ) from exc
        if r.status_code >= 400:
            raise LLMUnavailableError(f"LLM endpoint returned HTTP {r.status_code}")
        try:
            return str(r.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMOutputError("Unexpected response format from LLM endpoint") from exc

    def list_models(self) -> list[str]:
        try:
            r = self._client.get("/models", timeout=5.0)
            r.raise_for_status()
            return sorted(m.get("id", "") for m in r.json().get("data", []))
        except (httpx.HTTPError, ValueError):
            return []

    def test_connection(self) -> ConnectionStatus:
        models = self.list_models()
        if not models:
            return ConnectionStatus(False, self.provider, f"No models listed at {self.base_url}")
        return ConnectionStatus(
            True, self.provider, f"Endpoint ready ({len(models)} models)", {"models": models}
        )


@dataclass
class LLMSelection:
    client: LLMClient | None
    message: str
    warnings: list[str] = field(default_factory=list)


def build_llm(
    settings: Settings,
    model: str | None = None,
    provider: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> LLMSelection:
    """Create the configured LLM client (or None for the deterministic demo planner)."""
    provider = provider or settings.llm_provider
    if provider == "demo":
        return LLMSelection(None, "Deterministic demo planner (no LLM)")
    if provider == "ollama":
        check_llm_endpoint(settings, settings.ollama_base_url)
        client: LLMClient = OllamaClient(
            settings.ollama_base_url,
            model or settings.ollama_model,
            settings.llm_timeout_s,
            settings.llm_temperature,
            settings.llm_num_ctx,
            settings.ollama_think,
            transport=transport,
        )
        return LLMSelection(client, f"Ollama {client.model} at {settings.ollama_base_url}")
    if provider == "openai_compatible":
        if not settings.openai_compat_base_url or not (model or settings.openai_compat_model):
            raise LLMError("Set AGIS_OPENAI_COMPAT_BASE_URL and AGIS_OPENAI_COMPAT_MODEL")
        check_llm_endpoint(settings, settings.openai_compat_base_url)
        key = settings.openai_compat_api_key.get_secret_value() if settings.openai_compat_api_key else None
        client = OpenAICompatibleClient(
            settings.openai_compat_base_url,
            model or settings.openai_compat_model or "",
            key,
            settings.llm_timeout_s,
            settings.llm_temperature,
            transport=transport,
        )
        warn = (
            []
            if client.is_local
            else ["External LLM endpoint: only public dataset content is sent in prompts."]
        )
        return LLMSelection(
            client, f"OpenAI-compatible {client.model} at {settings.openai_compat_base_url}", warn
        )
    raise LLMError(f"Unknown LLM provider '{provider}'")
