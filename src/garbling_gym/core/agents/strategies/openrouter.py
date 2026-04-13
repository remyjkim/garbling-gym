# ABOUTME: Factory for creating OpenRouter API callers compatible with LLM strategy interface
# ABOUTME: Returns Callable[[system_prompt, user_prompt], response_text] using OpenRouter chat API

from typing import Callable

import requests


LLMCaller = Callable[[str, str], str]


def make_openrouter_caller(model: str, api_key: str) -> LLMCaller:
    """
    Return a callable that sends chat completions requests to OpenRouter.

    The returned callable accepts (system_prompt, user_prompt) and returns
    the model's response text.  HTTP errors are propagated so that calling
    strategies can trigger their fallback logic.

    Args:
        model: OpenRouter model identifier, e.g. "nvidia/llama-3.1-nemotron-70b-instruct"
        api_key: OpenRouter API key (OPENROUTER_API_KEY)
    """

    def caller(system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return caller
