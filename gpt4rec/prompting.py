"""Prompt construction for GPT4Rec."""

# prompt constants
prompt_prefix = "Previously, the customer has bought:"
prompt_suffix = "In the future, the customer wants to buy"


def build_history_prompt(item_titles):
    """Build a history prompt for the GPT4Rec model."""
    joined = ". ".join([str(x) for x in item_titles if x is not None and str(x) != ""])
    if joined:
        return f"{prompt_prefix} {joined}. {prompt_suffix}"
    return f"{prompt_prefix} {prompt_suffix}"


def build_train_text(item_titles, target_title):
    """Build a training text for the GPT4Rec model."""
    return f"{build_history_prompt(item_titles)} {target_title}"
