from app.llm.base import LLMProvider, LLMRequest, LLMResponse


class MockLLMProvider(LLMProvider):

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=f"Mock LLM response to: {request.prompt}",
            model="mock-model",
            provider="mock",
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": 0
            }
        )