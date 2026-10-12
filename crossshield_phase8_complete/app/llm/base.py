from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Optional[dict] = None


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM provider and return a normalized response.
        """
        pass