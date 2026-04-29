from typing import Any, Dict

import httpx

from config import Settings
from models import CppGenerateResponse, HealthResponse


class InferenceUnavailableError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InferenceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(self, prompt: str, max_tokens: int) -> CppGenerateResponse:
        url = f"{self.settings.cpp_server_url.rstrip('/')}/generate"
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, json={"prompt": prompt, "max_tokens": max_tokens})
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InferenceUnavailableError("The C++ inference server timed out") from exc
        except httpx.HTTPError as exc:
            raise InferenceUnavailableError(
                f"The C++ inference server is not reachable at {self.settings.cpp_server_url}"
            ) from exc

        try:
            return CppGenerateResponse.model_validate(response.json())
        except ValueError as exc:
            raise InferenceUnavailableError("The C++ inference server returned invalid JSON") from exc

    async def health(self) -> Dict[str, Any]:
        url = f"{self.settings.cpp_server_url.rstrip('/')}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise InferenceUnavailableError(
                f"The C++ inference server is not reachable at {self.settings.cpp_server_url}"
            ) from exc
