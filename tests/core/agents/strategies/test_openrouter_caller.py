# ABOUTME: Unit tests for the OpenRouter LLM caller factory
# ABOUTME: Verifies HTTP request format, response parsing, and error handling

from unittest.mock import MagicMock, patch

import pytest

from garbling_gym.core.agents.strategies.openrouter import make_openrouter_caller


class TestMakeOpenrouterCaller:
    """make_openrouter_caller returns a callable that wraps the OpenRouter chat API."""

    def test_returns_callable(self):
        """Factory returns a callable accepting (system_prompt, user_prompt)."""
        caller = make_openrouter_caller(model="test-model", api_key="test-key")
        assert callable(caller)

    def test_posts_to_openrouter_endpoint(self):
        """Caller sends POST to https://openrouter.ai/api/v1/chat/completions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "BUY"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            caller = make_openrouter_caller(model="test-model", api_key="test-key")
            caller("system prompt", "user prompt")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://openrouter.ai/api/v1/chat/completions"

    def test_sends_bearer_auth_header(self):
        """Caller includes Authorization: Bearer {api_key} header."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "BUY"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            caller = make_openrouter_caller(model="test-model", api_key="my-secret-key")
            caller("sys", "usr")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    def test_sends_system_and_user_messages(self):
        """Caller formats system and user prompts as OpenAI-compatible messages array."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "PASS"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            caller = make_openrouter_caller(model="test-model", api_key="key")
            caller("be a good agent", "should I buy?")

        body = mock_post.call_args[1]["json"]
        messages = body["messages"]
        assert {"role": "system", "content": "be a good agent"} in messages
        assert {"role": "user", "content": "should I buy?"} in messages

    def test_sends_correct_model(self):
        """Caller includes the model identifier in the request body."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "BUY"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            caller = make_openrouter_caller(model="nvidia/llama-3.1-nemotron-70b-instruct", api_key="key")
            caller("sys", "usr")

        body = mock_post.call_args[1]["json"]
        assert body["model"] == "nvidia/llama-3.1-nemotron-70b-instruct"

    def test_returns_response_text(self):
        """Caller returns the text content from choices[0].message.content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "BUY"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            caller = make_openrouter_caller(model="test-model", api_key="key")
            result = caller("sys", "usr")

        assert result == "BUY"

    def test_raises_on_http_error(self):
        """Caller propagates HTTP errors so strategies can trigger their fallback."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 401 Unauthorized")

        with patch("requests.post", return_value=mock_response):
            caller = make_openrouter_caller(model="test-model", api_key="bad-key")
            with pytest.raises(Exception, match="401"):
                caller("sys", "usr")
