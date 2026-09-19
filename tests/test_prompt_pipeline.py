import os
import asyncio

from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

from backend.app.main import app
from backend.app.services.adaptive.analyzer import extract_prompt_dna
from backend.app.services.prompt_optimizer import PromptOptimizer
from backend.app.services.response_evaluator import evaluate_response
from backend.app.services.response_improver import improve_response
from backend.app.services.response_sanitizer import sanitize_model_output
from backend.app.services.apil_pipeline import get_generation_budget
from backend.app.services import response_improver as response_improver_module


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


def test_sanitizer_prefers_provider_content_over_thinking_field():
    response = {"message": {"thinking": "private reasoning", "content": "Final answer."}}
    assert sanitize_model_output(response) == "Final answer."


def test_sanitizer_removes_unmarked_reasoning_preamble_conservatively():
    content = (
        "Okay, I need to identify the user's request.\n\n"
        "Let me think about the clearest approach.\n\n"
        "Photosynthesis converts light energy into chemical energy."
    )
    assert sanitize_model_output(content) == "Photosynthesis converts light energy into chemical energy."


def test_sanitizer_returns_empty_for_reasoning_only_output():
    assert sanitize_model_output("<think>private reasoning</think>") == ""


def test_sanitizer_removes_common_delimited_reasoning_marker():
    assert sanitize_model_output("<|begin_of_thought|>private<|end_of_thought|>Final.") == "Final."


def test_sanitizer_is_provider_independent():
    content = "<analysis>internal chain</analysis>OpenAI output: here is the answer."
    assert sanitize_model_output(content, provider="openai") == "OpenAI output: here is the answer."
    assert sanitize_model_output(content, provider="ollama") == "OpenAI output: here is the answer."


def test_universal_prompt_dna_works_for_unseen_prompt_type():
    dna = extract_prompt_dna(
        "Design a one-page ritual for a team whose systems are failing at dusk and keep the tone quiet, observatory-like.",
        {"language": "English"},
    )
    assert dna.intent in {"general", "problem_solving", "explanation"}
    assert dna.audience is None or dna.audience == "beginner"
    assert dna.constraints or dna.requirements
    assert dna.topic is not None


def test_requirement_driven_evaluator_flags_json_format_violation():
    prompt = "Return the answer as valid JSON with fields name and age."
    evaluation = evaluate_response(
        original_prompt=prompt,
        optimized_prompt=prompt,
        response="Alice is 32 years old.",
        prompt_dna=extract_prompt_dna(prompt, {"language": "English"}).to_dict(),
    )
    assert evaluation["improvement_needed"] is True
    assert any("json" in issue.lower() or "format" in issue.lower() for issue in evaluation["issues"]) or any(
        "json" in item.lower() for item in evaluation["missing_requirements"]
    )


def test_requirement_driven_evaluator_flags_sort_constraint_violation():
    prompt = "Write Python code to find the second largest number without using sort()."
    evaluation = evaluate_response(
        original_prompt=prompt,
        optimized_prompt=prompt,
        response="def second_largest(nums):\n    return sorted(nums)[-2]\n",
        prompt_dna=extract_prompt_dna(prompt, {"language": "English"}).to_dict(),
    )
    assert evaluation["improvement_needed"] is True
    assert any("sort" in item.lower() for item in evaluation["missing_requirements"]) or any(
        "sort" in issue.lower() for issue in evaluation["issues"]
    )


def test_requirement_driven_evaluator_flags_reasoning_leakage():
    prompt = "Explain photosynthesis to a beginner using a simple real-world example."
    response = (
        "Okay, the user wants a short explanation.\n"
        "Let me think about a simple example.\n"
        "The answer should be clear."
    )
    evaluation = evaluate_response(
        original_prompt=prompt,
        optimized_prompt=prompt,
        response=response,
        prompt_dna=extract_prompt_dna(prompt, {"language": "English", "response_length": "short"}).to_dict(),
    )
    assert evaluation["improvement_needed"] is True
    assert any("reasoning" in issue.lower() for issue in evaluation["issues"])


def test_no_unnecessary_improvement_when_response_already_satisfies_request():
    prompt = "Return a single sentence answering: what is the capital of France?"
    dna = extract_prompt_dna(prompt, {"language": "English"}).to_dict()
    response = "Paris is the capital of France."
    evaluation = evaluate_response(
        original_prompt=prompt,
        optimized_prompt=prompt,
        response=response,
        prompt_dna=dna,
    )
    assert evaluation["improvement_needed"] is False
    assert evaluation["passed"] is True


def test_improver_attempts_correction_for_json_format_violation():
    prompt = "Return the answer as valid JSON with fields name and age."
    evaluation = evaluate_response(
        original_prompt=prompt,
        optimized_prompt=prompt,
        response="Alice is 32 years old.",
        prompt_dna=extract_prompt_dna(prompt, {"language": "English"}).to_dict(),
    )

    result = __import__('asyncio').run(
        improve_response(
            original_prompt=prompt,
            optimized_prompt=prompt,
            response="Alice is 32 years old.",
            evaluation=evaluation,
            prompt_dna=extract_prompt_dna(prompt, {"language": "English"}).to_dict(),
            provider_name="ollama",
            model_name="llama3.1",
        )
    )
    assert result["improvement_attempted"] is True
    assert result["improvement_applied"] is False
    assert result["response"] == "Alice is 32 years old."


def test_improvement_success_sets_applied_true(monkeypatch):
    class SuccessfulProvider:
        async def generate(self, **kwargs):
            return "Improved final answer."

    monkeypatch.setattr(
        response_improver_module.provider_router,
        "get_provider",
        lambda provider: SuccessfulProvider(),
    )
    result = asyncio.run(
        response_improver_module.improve_response(
            original_prompt="Answer briefly.",
            optimized_prompt="Answer briefly.",
            response="Planning text.",
            evaluation={"improvement_needed": True, "issues": ["planning"]},
            provider_name="ollama",
            model_name="qwen3:4b",
        )
    )
    assert result["improvement_attempted"] is True
    assert result["improvement_applied"] is True
    assert result["response"] == "Improved final answer."


def test_improvement_timeout_preserves_best_response(monkeypatch):
    class SlowProvider:
        async def generate(self, **kwargs):
            await asyncio.sleep(0.05)

    monkeypatch.setattr(
        response_improver_module.provider_router,
        "get_provider",
        lambda provider: SlowProvider(),
    )
    monkeypatch.setattr(response_improver_module.settings, "APIL_IMPROVEMENT_TIMEOUT_SECONDS", 0.001)
    result = asyncio.run(
        response_improver_module.improve_response(
            original_prompt="Answer briefly.",
            optimized_prompt="Answer briefly.",
            response="Best available answer.",
            evaluation={"improvement_needed": True, "issues": ["too long"]},
            provider_name="ollama",
            model_name="qwen3:4b",
        )
    )
    assert result["improvement_attempted"] is True
    assert result["improvement_applied"] is False
    assert result["failure_reason"] == "improvement_timeout"
    assert result["response"] == "Best available answer."


def test_empty_improvement_preserves_best_response(monkeypatch):
    class EmptyProvider:
        async def generate(self, **kwargs):
            return "<think>private reasoning</think>"

    monkeypatch.setattr(
        response_improver_module.provider_router,
        "get_provider",
        lambda provider: EmptyProvider(),
    )
    result = asyncio.run(
        response_improver_module.improve_response(
            original_prompt="Answer briefly.",
            optimized_prompt="Answer briefly.",
            response="Best available answer.",
            evaluation={"improvement_needed": True, "issues": ["too long"]},
            provider_name="ollama",
            model_name="qwen3:4b",
        )
    )
    assert result["improvement_applied"] is False
    assert result["failure_reason"] == "empty_improvement"
    assert result["response"] == "Best available answer."


def test_diagnostic_endpoint_exposes_full_pipeline_snapshot():
    client = TestClient(app)
    payload = {
        "original_prompt": "Return the answer as valid JSON with fields name and age.",
        "prompt_dna": {},
        "optimized_prompt": "Return the answer as valid JSON with fields name and age.",
        "selected_provider": "ollama",
        "selected_model": "llama3.1",
        "raw_provider_response": '{"name": "Alice", "age": 32}',
        "response_evaluation": {"improvement_needed": False},
        "improvement_needed": False,
        "improvement_applied": False,
        "improvement_attempts": 0,
        "improved_response": '{"name": "Alice", "age": 32}',
        "final_quality_gate": {"passed": True},
        "final_response": '{"name": "Alice", "age": 32}'
    }
    response = client.post("/v1/test", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["original_prompt"] == payload["original_prompt"]
    assert body["final_response"] == payload["final_response"]
    assert body["improvement_needed"] is False
    assert body["timing"]["total_ms"] == 0


def test_generation_budget_uses_semantic_prompt_requirements():
    assert get_generation_budget({"desired_length": "short"}) == 384
    assert get_generation_budget({"desired_length": "medium", "desired_depth": "simple"}) == 256
    assert get_generation_budget({"desired_length": "long"}) == 2048
    assert get_generation_budget({"desired_length": "long", "code_requirements": ("complete",)}) == 3072
