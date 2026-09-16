from abc import ABC, abstractmethod


class AIProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a response from an AI provider.
        """
        raise NotImplementedError