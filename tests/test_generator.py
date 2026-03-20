"""Tests for LLM generator service."""

from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import RetrievalResult
from app.services.generator import LLMGenerator, _format_sources


@pytest.fixture
def sample_sources():
    return [
        RetrievalResult(
            image_id="img_001",
            filename="dog_park.jpg",
            filepath="/images/dog_park.jpg",
            score=0.85,
            captions=["A dog playing in the park", "A golden retriever on grass"],
        ),
        RetrievalResult(
            image_id="img_002",
            filename="cat_sofa.jpg",
            filepath="/images/cat_sofa.jpg",
            score=0.72,
            captions=["A cat sitting on a sofa"],
        ),
    ]


class TestFormatSources:
    def test_formats_multiple_sources(self, sample_sources):
        result = _format_sources(sample_sources)
        assert "[来源 1]" in result
        assert "dog_park.jpg" in result
        assert "0.850" in result
        assert "[来源 2]" in result
        assert "cat_sofa.jpg" in result

    def test_empty_captions(self):
        sources = [
            RetrievalResult(
                image_id="img_003",
                filename="empty.jpg",
                filepath="/images/empty.jpg",
                score=0.5,
                captions=[],
            ),
        ]
        result = _format_sources(sources)
        assert "无描述" in result

    def test_empty_sources(self):
        result = _format_sources([])
        assert result == ""


class TestLLMGenerator:
    @patch("app.services.generator.settings")
    @patch("app.services.generator.OpenAI")
    def test_generate_calls_api(self, mock_openai_cls, mock_settings, sample_sources):
        mock_settings.llm_api_key = "sk-test"
        mock_settings.llm_base_url = "https://proxy.example.com/v1"
        mock_settings.llm_model = "test-model"
        mock_settings.llm_max_tokens = 512
        mock_settings.llm_temperature = 0.5

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "这是一只金毛犬在公园里玩耍。"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        generator = LLMGenerator()
        answer = generator.generate("有狗在玩耍吗？", sample_sources)

        assert answer == "这是一只金毛犬在公园里玩耍。"
        mock_client.chat.completions.create.assert_called_once()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert "dog_park.jpg" in call_kwargs["messages"][1]["content"]

    @patch("app.services.generator.settings")
    def test_raises_without_api_key(self, mock_settings):
        mock_settings.llm_api_key = ""
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            LLMGenerator()
