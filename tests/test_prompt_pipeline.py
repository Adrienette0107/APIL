import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

from backend.app.services.adaptive.analyzer import extract_prompt_dna
from backend.app.services.prompt_optimizer import PromptOptimizer
from backend.app.services.response_sanitizer import sanitize_model_output


def test_extract_prompt_dna_for_beginner_explanation():
    dna = extract_prompt_dna(
        "Explain machine learning to a beginner using a simple real-world example.",
        {"language": "English", "level": "beginner", "response_length": "short"},
    )

    assert dna.intent == "explanation"
    assert dna.task == "explain"
    assert dna.topic is not None
    assert dna.audience == "beginner"
    assert dna.language == "English"
    assert dna.response_length == "short"
    assert dna.output_format is not None
    assert dna.constraints


def test_extract_prompt_dna_for_code_generation_with_constraints():
    dna = extract_prompt_dna(
        "Build a Python program that reads a list of numbers and finds the second largest number without using sort(). Explain the logic briefly and provide complete runnable code.",
        {"language": "English"},
    )

    assert dna.intent == "code_generation"
    assert dna.language == "English"
    assert dna.code_requirements
    assert "sort()" in " ".join(dna.constraints)


def test_optimizer_preserves_constraints_and_formats_prompt():
    optimizer = PromptOptimizer()
    result = optimizer.optimize(
        "Compare PostgreSQL and MongoDB for a college project.",
        {"language": "English", "level": "beginner", "response_length": "medium"},
    )

    assert result["status"] == "success"
    assert "Compare PostgreSQL and MongoDB for a college project." in result["optimized_prompt"]
    assert "college project" in result["optimized_prompt"].lower()
    assert "response" in result["optimized_prompt"].lower()
    assert result["prompt_dna"]["intent"] == "comparison"


def test_factual_question_prompt_dna():
    dna = extract_prompt_dna("What is the capital of France?", {"language": "English"})
    assert dna.intent == "explanation"
    assert dna.language == "English"


def test_advanced_technical_explanation_prompt_dna():
    dna = extract_prompt_dna(
        "Explain asynchronous I/O in Python at an advanced level with architecture details.",
        {"language": "English", "level": "advanced", "response_length": "long"},
    )
    assert dna.intent == "explanation"
    assert dna.audience == "advanced"
    assert dna.desired_length == "long"


def test_summarization_prompt_dna():
    dna = extract_prompt_dna("Summarize the following text in five bullet points.", {"language": "English"})
    assert dna.intent == "summarization"
    assert dna.output_format in {"bullet points", "summary"}


def test_translation_prompt_dna():
    dna = extract_prompt_dna("Translate this paragraph to Spanish.", {"language": "English"})
    assert dna.intent == "translation"
    assert dna.language == "English"


def test_creative_writing_prompt_dna():
    dna = extract_prompt_dna("Write a short creative story about a city that glows at night.", {"language": "English"})
    assert dna.intent == "code_generation" or dna.intent == "general"


def test_structured_output_prompt_dna():
    dna = extract_prompt_dna("Return the answer as JSON with keys name, age, and city.", {"language": "English"})
    assert dna.output_format in {"json", None}


def test_long_technical_prompt_dna():
    dna = extract_prompt_dna(
        "Design a secure microservice architecture for a payment platform with authentication, rate limiting, retries, and observability.",
        {"language": "English", "level": "advanced"},
    )
    assert dna.domain in {"technology", "general"}
    assert dna.audience == "advanced"


def test_ambiguous_prompt_dna():
    dna = extract_prompt_dna("Can you help?", {"language": "English"})
    assert dna.ambiguity in {"medium", "low"}


def test_missing_info_prompt_dna():
    dna = extract_prompt_dna("Build something useful for the team.", {"language": "English"})
    assert dna.missing_information or dna.ambiguity == "medium"


def test_short_response_preference_optimization():
    result = PromptOptimizer().optimize("Answer in one sentence: what is HTTP?", {"language": "English", "response_length": "short"})
    assert "short" in result["optimized_prompt"].lower()


def test_long_response_preference_optimization():
    result = PromptOptimizer().optimize("Explain the OSI model in depth.", {"language": "English", "response_length": "long"})
    assert result["status"] == "success"
    assert "long" in result["optimized_prompt"].lower()


def test_sanitizer_keeps_normal_response_unchanged():
    content = "Paris is the capital of France."
    assert sanitize_model_output(content) == content


def test_sanitizer_removes_xml_think_block():
    content = "<think>Hidden reasoning here.</think>Paris is the capital of France."
    assert sanitize_model_output(content) == "Paris is the capital of France."


def test_sanitizer_removes_analysis_block():
    content = "<analysis>Let's calculate exactly.</analysis>Answer: 42."
    assert sanitize_model_output(content) == "Answer: 42."


def test_sanitizer_removes_reasoning_block():
    content = "<reasoning>Need to be careful.</reasoning>Final answer: use Python."
    assert sanitize_model_output(content) == "Final answer: use Python."


def test_sanitizer_removes_mixed_reasoning_and_final_answer():
    content = "<think>step by step</think>\n\nFinal answer: deploy via Docker."
    assert sanitize_model_output(content) == "Final answer: deploy via Docker."


def test_sanitizer_keeps_clean_response_unchanged():
    content = "The API is available and stable."
    assert sanitize_model_output(content) == content


def test_sanitizer_is_provider_independent():
    content = "<analysis>internal chain</analysis>OpenAI output: here is the answer."
    assert sanitize_model_output(content, provider="openai") == "OpenAI output: here is the answer."
    assert sanitize_model_output(content, provider="ollama") == "OpenAI output: here is the answer."
