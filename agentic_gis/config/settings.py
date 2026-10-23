"""Application settings loaded from environment variables and an optional ``.env`` file.

App-specific variables use the ``AGIS_`` prefix. Databricks and Ollama variables use the
names documented in ``.env.example`` (``DATABRICKS_SERVER_HOSTNAME``, ``OLLAMA_MODEL`` ...).
Secrets are stored as :class:`pydantic.SecretStr` so they never appear in ``repr`` output
or logs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AppMode = Literal["demo", "local", "databricks"]
LLMProvider = Literal["ollama", "demo", "openai_compatible"]
Sensitivity = Literal["public", "internal", "confidential", "restricted"]
SENSITIVITY_ORDER: dict[str, int] = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


def _alias(*names: str) -> AliasChoices:
    return AliasChoices(*names)


class Settings(BaseSettings):
    """All runtime configuration. Construct via :func:`get_settings` in application code."""

    model_config = SettingsConfigDict(
        env_prefix="AGIS_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---- General -------------------------------------------------------------------------
    mode: AppMode = Field("demo", description="Default data mode: demo | local | databricks")
    admin_mode: bool = Field(False, description="Enables dataset registration UI.")
    session_label: str = "Agentic GIS"

    # ---- LLM -----------------------------------------------------------------------------
    llm_provider: LLMProvider = "ollama"
    ollama_base_url: str = Field(
        "http://localhost:11434", validation_alias=_alias("OLLAMA_BASE_URL", "AGIS_OLLAMA_BASE_URL")
    )
    ollama_model: str = Field("qwen2.5:7b", validation_alias=_alias("OLLAMA_MODEL", "AGIS_OLLAMA_MODEL"))
    ollama_think: bool | None = Field(
        None, description="Send `think` to Ollama for reasoning models (None = do not send)."
    )
    openai_compat_base_url: str | None = None
    openai_compat_model: str | None = None
    openai_compat_api_key: SecretStr | None = None
    llm_temperature: float = 0.0
    llm_timeout_s: float = 240.0
    llm_num_ctx: int = 8192
    llm_fallback_to_rules: bool = Field(
        True,
        description="Use the deterministic planner when the LLM is unavailable or returns invalid output.",
    )
    allow_external_llm: bool = Field(
        False,
        description="Permit LLM endpoints that are not on localhost. Requires explicit organizational approval.",
    )
    llm_max_sensitivity: Sensitivity = Field(
        "internal",
        description="Highest dataset sensitivity whose schema/result digests may be placed in LLM prompts.",
    )

    # ---- Databricks ----------------------------------------------------------------------
    databricks_server_hostname: str | None = Field(
        None, validation_alias=_alias("DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HOST")
    )
    databricks_http_path: str | None = Field(None, validation_alias=_alias("DATABRICKS_HTTP_PATH"))
    databricks_token: SecretStr | None = Field(None, validation_alias=_alias("DATABRICKS_TOKEN"))
    databricks_catalog: str | None = Field(None, validation_alias=_alias("DATABRICKS_CATALOG"))
    databricks_schema: str | None = Field(None, validation_alias=_alias("DATABRICKS_SCHEMA"))
    databricks_auth_type: Literal["pat", "oauth-u2m", "oauth-m2m"] = Field(
        "pat", validation_alias=_alias("DATABRICKS_AUTH_TYPE", "AGIS_DATABRICKS_AUTH_TYPE")
    )
    databricks_client_id: str | None = Field(None, validation_alias=_alias("DATABRICKS_CLIENT_ID"))
    databricks_client_secret: SecretStr | None = Field(
        None, validation_alias=_alias("DATABRICKS_CLIENT_SECRET")
    )

    # ---- Query safety --------------------------------------------------------------------
    max_result_rows: int = Field(50_000, ge=1, le=1_000_000)
    query_timeout_s: int = Field(120, ge=1, le=3600)
    sql_approval: Literal["generated", "always", "never"] = Field(
        "generated",
        description=(
            "generated: approve LLM-authored SQL and every Databricks query; "
            "always: approve every query; never: no approval (testing only)."
        ),
    )
    max_plan_steps: int = 12
    max_replans: int = 1

    # ---- Paths ---------------------------------------------------------------------------
    datasets_config: Path = PROJECT_ROOT / "config" / "datasets.yaml"
    datasets_local_config: Path = PROJECT_ROOT / "config" / "datasets.local.yaml"
    data_dir: Path = PROJECT_ROOT / "data"
    sample_data_dir: Path = PROJECT_ROOT / "sample_data"
    audit_log_path: Path = PROJECT_ROOT / "logs" / "audit.jsonl"
    audit_log_sql: bool = True

    # ---- Spatial -------------------------------------------------------------------------
    projected_crs: str = Field(
        "EPSG:2263", description="Projected CRS for NYC distance/area operations (ftUS)."
    )
    geocoder: Literal["reference", "nyc_geosearch"] = "reference"
    duckdb_install_extensions: bool = Field(
        True, description="Allow DuckDB to download the spatial extension if it is not installed."
    )

    # ---- Map -----------------------------------------------------------------------------
    map_renderer: Literal["folium", "pydeck"] = "folium"
    default_basemap: str = "CARTO Dark Matter"
    custom_basemap_url: str | None = None
    custom_basemap_attribution: str | None = None
    max_render_features: int = 6000

    # ---- Public data ---------------------------------------------------------------------
    socrata_domain: str = "data.cityofnewyork.us"
    socrata_app_token: SecretStr | None = Field(
        None, validation_alias=_alias("SOCRATA_APP_TOKEN", "AGIS_SOCRATA_APP_TOKEN")
    )

    @field_validator(
        "datasets_config", "datasets_local_config", "data_dir", "sample_data_dir", "audit_log_path"
    )
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @field_validator("databricks_server_hostname")
    @classmethod
    def _strip_scheme(cls, v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip()
        if "://" in v:
            v = urlparse(v).netloc
        return v.rstrip("/") or None

    # ---- Derived helpers -----------------------------------------------------------------
    @property
    def databricks_configured(self) -> bool:
        if not (self.databricks_server_hostname and self.databricks_http_path):
            return False
        if self.databricks_auth_type == "pat":
            return self.databricks_token is not None and bool(self.databricks_token.get_secret_value())
        if self.databricks_auth_type == "oauth-m2m":
            return bool(self.databricks_client_id and self.databricks_client_secret)
        return True  # oauth-u2m uses an interactive browser login

    def secret_values(self) -> list[str]:
        """Secret strings to scrub from logs and error messages."""
        out: list[str] = []
        for s in (
            self.databricks_token,
            self.databricks_client_secret,
            self.openai_compat_api_key,
            self.socrata_app_token,
        ):
            if s is not None and s.get_secret_value():
                out.append(s.get_secret_value())
        return out

    def public_summary(self) -> dict[str, object]:
        """Settings safe to display in the UI (no secrets)."""
        return {
            "mode": self.mode,
            "llm_provider": self.llm_provider,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "allow_external_llm": self.allow_external_llm,
            "llm_max_sensitivity": self.llm_max_sensitivity,
            "databricks_configured": self.databricks_configured,
            "databricks_server_hostname": self.databricks_server_hostname,
            "databricks_catalog": self.databricks_catalog,
            "databricks_schema": self.databricks_schema,
            "databricks_auth_type": self.databricks_auth_type,
            "max_result_rows": self.max_result_rows,
            "query_timeout_s": self.query_timeout_s,
            "sql_approval": self.sql_approval,
            "projected_crs": self.projected_crs,
            "map_renderer": self.map_renderer,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
