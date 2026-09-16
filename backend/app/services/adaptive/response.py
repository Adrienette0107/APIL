def adapt_response(
    response: str,
    preferences: dict | None = None,
) -> str:

    preferences = preferences or {}

    response_length = preferences.get(
        "response_length",
        "medium",
    )

    if response_length == "short":
        return response

    if response_length == "long":
        return response

    return response