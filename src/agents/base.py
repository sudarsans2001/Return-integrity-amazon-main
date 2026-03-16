"""Base agent class with shared Nova integration."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ..nova_client import NovaClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """All pipeline agents inherit from this base."""

    name: str = "BaseAgent"

    def __init__(self, nova: NovaClient) -> None:
        self.nova = nova

    def run(self, **kwargs: Any) -> Any:
        start = time.perf_counter()
        logger.info("[%s] Starting...", self.name)
        result = self.execute(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("[%s] Completed in %.1f ms", self.name, elapsed_ms)
        return result, elapsed_ms

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        ...
