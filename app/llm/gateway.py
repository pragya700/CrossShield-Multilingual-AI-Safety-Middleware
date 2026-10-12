from app.llm.base import LLMProvider, LLMRequest, LLMResponse


class LLMGateway:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.provider.generate(request)