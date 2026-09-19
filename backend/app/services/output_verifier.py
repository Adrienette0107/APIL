from __future__ import annotations

import json
import re
from typing import Any

from backend.app.services.response_sanitizer import (
    sanitize_model_output,
)


class OutputVerifier:
    """
    Deterministic APIL output-quality verifier.

    This component does NOT call a GenAI provider.

    It checks for:
        - empty output
        - extremely short output
        - obvious thinking/reasoning leakage
        - obvious provider-error text
        - malformed JSON when JSON was explicitly requested
        - excessive repetition
        - unfinished output
        - basic prompt/output mismatch signals

    Semantic quality evaluation can be added later as an optional
    stage, but it should not run automatically for every request.
    """

    def verify(
        self,
        original_prompt: str,
        response: str,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        preferences = (
            preferences or {}
        )

        issues: list[str] = []

        if not isinstance(
            response,
            str,
        ):
            response = str(
                response
            )

        cleaned_response = response.strip()

        # ----------------------------------------------------------
        # 1. Empty response
        # ----------------------------------------------------------

        if not cleaned_response:

            issues.append(
                "Empty response."
            )

            return {
                "status": "failed",
                "verified": False,
                "original_prompt": original_prompt,
                "response": "",
                "issues": issues,
                "checks": {
                    "non_empty": False,
                    "minimum_length": False,
                    "thinking_leakage": True,
                    "provider_error": True,
                    "repetition": True,
                    "unfinished": True,
                    "json_valid": None,
                },
            }

        # ----------------------------------------------------------
        # 2. Minimum response length
        # ----------------------------------------------------------

        minimum_length_ok = (
            len(cleaned_response) >= 5
        )

        if not minimum_length_ok:

            issues.append(
                "Response is too short."
            )

        # ----------------------------------------------------------
        # 3. Thinking leakage
        # ----------------------------------------------------------

        thinking_patterns = (
            "<think>",
            "</think>",
            "let me think",
            "i need to think",
            "chain of thought",
        )

        lowered_response = (
            cleaned_response.lower()
        )

        sanitized_response = sanitize_model_output(
            cleaned_response
        )

        thinking_leakage = any(
            pattern in lowered_response
            for pattern in thinking_patterns
        ) or sanitized_response != cleaned_response

        if thinking_leakage:

            issues.append(
                "Response contains reasoning/thinking leakage."
            )

        # ----------------------------------------------------------
        # 4. Provider error leakage
        # ----------------------------------------------------------

        provider_error_patterns = (
            "provider error:",
            "api key is invalid",
            "authentication failed",
            "internal server error",
            "traceback (most recent call last)",
        )

        provider_error = any(
            pattern in lowered_response
            for pattern in provider_error_patterns
        )

        if provider_error:

            issues.append(
                "Response appears to contain provider error information."
            )

        # ----------------------------------------------------------
        # 5. Repetition detection
        # ----------------------------------------------------------

        repetition = self._has_excessive_repetition(
            cleaned_response
        )

        if repetition:

            issues.append(
                "Response contains excessive repetition."
            )

        # ----------------------------------------------------------
        # 6. Unfinished output detection
        # ----------------------------------------------------------

        unfinished = self._looks_unfinished(
            cleaned_response
        )

        if unfinished:

            issues.append(
                "Response appears to be unfinished."
            )

        # ----------------------------------------------------------
        # 7. JSON validation when requested
        # ----------------------------------------------------------

        json_requested = (
            self._json_requested(
                original_prompt=original_prompt,
                preferences=preferences,
            )
        )

        json_valid: bool | None = None

        if json_requested:

            json_valid = self._is_valid_json(
                cleaned_response
            )

            if not json_valid:

                issues.append(
                    "Response was expected to be valid JSON."
                )

        # ----------------------------------------------------------
        # 8. Final status
        # ----------------------------------------------------------

        verified = (
            len(issues) == 0
        )

        return {
            "status": (
                "verified"
                if verified
                else "failed"
            ),
            "verified": verified,
            "original_prompt": original_prompt,
            "response": cleaned_response,
            "issues": issues,
            "checks": {
                "non_empty": True,
                "minimum_length": minimum_length_ok,
                "thinking_leakage": (
                    not thinking_leakage
                ),
                "provider_error": (
                    not provider_error
                ),
                "repetition": (
                    not repetition
                ),
                "unfinished": (
                    not unfinished
                ),
                "json_valid": json_valid,
            },
        }

    # ------------------------------------------------------------------
    # Repetition
    # ------------------------------------------------------------------

    @staticmethod
    def _has_excessive_repetition(
        response: str,
    ) -> bool:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            response,
        )

        normalized: list[str] = []

        for sentence in sentences:

            sentence = re.sub(
                r"\s+",
                " ",
                sentence.strip().lower(),
            )

            if sentence:
                normalized.append(
                    sentence
                )

        if len(normalized) < 4:
            return False

        unique_sentences = set(
            normalized
        )

        repetition_ratio = (
            len(unique_sentences)
            / len(normalized)
        )

        return repetition_ratio < 0.60

    # ------------------------------------------------------------------
    # Unfinished output
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_unfinished(
        response: str,
    ) -> bool:

        stripped = response.rstrip()

        unfinished_markers = (
            "...",
            "…",
            "to be continued",
            "and so on",
            "etc",
        )

        lowered = stripped.lower()

        # Only treat a marker as suspicious when it is actually
        # at the end of the response.
        if any(
            lowered.endswith(marker)
            for marker in unfinished_markers
        ):
            return True

        # Obvious unmatched code fence.
        if stripped.count("```") % 2 != 0:
            return True

        return False

    # ------------------------------------------------------------------
    # JSON detection
    # ------------------------------------------------------------------

    @staticmethod
    def _json_requested(
        original_prompt: str,
        preferences: dict[str, Any],
    ) -> bool:

        output_format = preferences.get(
            "output_format"
        )

        if isinstance(
            output_format,
            str,
        ) and output_format.strip().lower() == "json":
            return True

        prompt_lower = (
            original_prompt.lower()
            if isinstance(
                original_prompt,
                str,
            )
            else ""
        )

        return (
            "return as json" in prompt_lower
            or "respond in json" in prompt_lower
            or "output json" in prompt_lower
            or "valid json" in prompt_lower
        )

    @staticmethod
    def _is_valid_json(
        response: str,
    ) -> bool:

        candidate = response.strip()

        # Remove an outer markdown fence only for validation.
        if candidate.startswith(
            "```"
        ) and candidate.endswith(
            "```"
        ):

            lines = candidate.splitlines()

            if len(lines) >= 2:

                candidate = "\n".join(
                    lines[1:-1]
                ).strip()

        try:

            json.loads(
                candidate
            )

            return True

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):

            return False

