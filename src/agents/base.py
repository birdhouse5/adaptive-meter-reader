"""Base agent class — all agents inherit from this."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    """Structured result returned by every agent."""

    agent_name: str
    output: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base for all agents in the pipeline."""

    name: str = "base"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"agent.{self.name}")

    @abstractmethod
    async def _run(self, payload: dict[str, Any]) -> AgentResult:
        """Implement agent-specific logic. Subclasses override this."""
        ...

    async def process(self, payload: dict[str, Any]) -> AgentResult:
        """Execute the agent with timing and logging."""
        self.logger.info("Starting %s", self.name)
        start = time.perf_counter()
        try:
            result = await self._run(payload)
        except Exception:
            self.logger.exception("Agent %s failed", self.name)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.processing_time_ms = elapsed_ms
        self.logger.info(
            "%s finished in %.1f ms (confidence=%.2f)",
            self.name,
            elapsed_ms,
            result.confidence,
        )
        return result
