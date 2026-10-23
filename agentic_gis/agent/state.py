"""Agent state and structured models exchanged with the LLM.

Graph state is kept JSON-serializable (it is checkpointed by LangGraph); DataFrames live in
the per-session :class:`~agentic_gis.agent.workspace.ResultStore` and are referenced by id.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

STEP_ID_PATTERN = r"^[sd]\d{1,2}$"


class PlanStep(BaseModel):
    id: str = Field(pattern=STEP_ID_PATTERN, description="Step id: s1, s2, ...")
    tool: str = Field(description="Registered tool name")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    purpose: str = Field("", description="Why this step is needed, in plain language")


class Plan(BaseModel):
    interpretation: str = Field(description="One-sentence restatement of what the user wants")
    geography: str | None = Field(None, description="Geographic focus, e.g. 'Brooklyn' or 'citywide'")
    datasets: list[str] = Field(default_factory=list, description="Dataset ids the plan uses")
    steps: list[PlanStep] = Field(default_factory=list, max_length=16)
    clarification: str | None = Field(
        None, description="If the request is ambiguous, a question for the user instead of steps"
    )
    assumptions: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    kind: Literal["observation", "inference", "verified"] = Field(
        description="observation = directly measured in the data; inference = interpretation; "
        "verified = confirmed by an authoritative verification dataset"
    )
    text: str


class Answer(BaseModel):
    headline: str = Field(description="One-line answer")
    summary: str = Field(description="2-5 sentence narrative in plain language")
    findings: list[Finding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    suggested_follow_ups: list[str] = Field(default_factory=list, max_length=4)


class AgentState(TypedDict, total=False):
    session_id: str
    mode: str
    turn: int
    question: str
    effective_question: str
    messages: Annotated[list[dict[str, Any]], operator.add]
    context: dict[str, Any]
    candidates: list[dict[str, Any]]
    plan: dict[str, Any] | None
    plan_source: str
    intent: dict[str, Any] | None
    sql_reviews: list[dict[str, Any]]
    approval: dict[str, Any] | None
    steps: list[dict[str, Any]]
    layers: list[dict[str, Any]]
    answer: dict[str, Any] | None
    citations: list[dict[str, Any]]
    methodology: list[str]
    trace: list[dict[str, Any]]
    replans: int
    replan_feedback: list[str] | None
    status: str
    error: str | None
    primary_artifact: str | None
    llm_used: bool
