from __future__ import annotations


class OutputVerifier:

    def verify(
        self,
        original_prompt: str,
        response: str,
        preferences: dict | None = None,
    ) -> dict:

        issues = []

        if not response:
            issues.append(
                "Empty response."
            )

        cleaned_response = response.strip()

        if len(cleaned_response) < 5:
            issues.append(
                "Response is too short."
            )

        verified = len(issues) == 0

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
        }