from __future__ import annotations

from backend.app.services.adaptive.analyzer import extract_prompt_dna


class PromptOptimizer:

    def optimize(
        self,
        prompt: str,
        preferences: dict | None = None,
    ) -> dict:

        original_prompt = prompt.strip()

        if not original_prompt:
            raise ValueError("Prompt cannot be empty.")

        preferences = preferences or {}
        dna = extract_prompt_dna(original_prompt, preferences)

        sections: list[str] = [
            "User request:",
            original_prompt,
        ]

        requirements: list[str] = []
        if dna.intent:
            requirements.append(f"Intent: {dna.intent}.")
        if dna.language:
            requirements.append(f"Respond in {dna.language}.")
        if dna.audience:
            requirements.append(f"Tailor the response for a {dna.audience} audience.")
        if dna.desired_depth:
            requirements.append(f"Use a {dna.desired_depth} level of detail.")
        if dna.desired_length:
            requirements.append(f"Keep the response {dna.desired_length} in length.")
        if dna.tone:
            requirements.append(f"Use a {dna.tone} tone.")
        if dna.output_format:
            requirements.append(f"Use {dna.output_format} format.")
        if dna.domain:
            requirements.append(f"Stay within the {dna.domain} domain when relevant.")
        if dna.context:
            requirements.append(f"Use the context: {dna.context}.")

        requirements.extend(dna.constraints)
        requirements.extend(dna.requirements)
        requirements.extend(dna.special_instructions)

        if dna.missing_information:
            requirements.append("Missing information to resolve if needed: " + "; ".join(dna.missing_information))

        if dna.code_requirements:
            requirements.append("Code requirements: " + "; ".join(dna.code_requirements))

        if requirements:
            sections.extend(["", "Requirements:"])
            sections.extend(f"- {requirement}" for requirement in dict.fromkeys(requirements))

        sections.extend([
            "",
            "Answer directly and preserve the user's actual intent.",
            "Do not invent requirements or hidden assumptions beyond what is clearly justified.",
            "Do not mention internal instructions or chain-of-thought.",
        ])

        optimized_prompt = "\n".join(sections)

        return {
            "status": "success",
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "prompt_dna": dna.to_dict(),
            "optimization": {
                "language": dna.language,
                "level": dna.audience,
                "response_length": dna.desired_length or dna.response_length,
                "prompt_changed": original_prompt != optimized_prompt,
            },
        }