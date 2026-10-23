"""Data-egress policy for LLM prompts.

The default deployment uses a local Ollama server, so prompts never leave the machine.
If an administrator configures a non-local endpoint, ``AGIS_ALLOW_EXTERNAL_LLM=true`` is
required and only *public* dataset metadata/results may be included in prompts.
"""

from __future__ import annotations

import ipaddress
from typing import Literal
from urllib.parse import urlparse

from agentic_gis.config.registry import DatasetSpec
from agentic_gis.config.settings import SENSITIVITY_ORDER, Settings


class ExternalLLMNotAllowedError(RuntimeError):
    pass


def is_local_endpoint(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "host.docker.internal"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback


def check_llm_endpoint(settings: Settings, url: str) -> None:
    if not is_local_endpoint(url) and not settings.allow_external_llm:
        raise ExternalLLMNotAllowedError(
            f"LLM endpoint {urlparse(url).hostname} is not local. Set AGIS_ALLOW_EXTERNAL_LLM=true only if your "
            "organization has approved sending prompts (which may include dataset metadata) to that endpoint."
        )


PromptLevel = Literal["full", "metadata_only", "name_only"]


class DataEgressPolicy:
    def __init__(self, settings: Settings, llm_is_local: bool):
        self.settings = settings
        self.llm_is_local = llm_is_local

    @property
    def max_level(self) -> int:
        level = SENSITIVITY_ORDER[self.settings.llm_max_sensitivity]
        if not self.llm_is_local:
            level = min(level, SENSITIVITY_ORDER["public"])
        return level

    def dataset_level(self, ds: DatasetSpec) -> PromptLevel:
        """How much of a dataset may appear in prompts: schema + result digests, metadata, or name only."""
        s = SENSITIVITY_ORDER[ds.sensitivity]
        if s <= self.max_level:
            return "full"
        if s <= self.max_level + 1:
            return "metadata_only"
        return "name_only"

    def may_include_results(self, datasets: list[DatasetSpec]) -> bool:
        return all(self.dataset_level(d) == "full" for d in datasets)
