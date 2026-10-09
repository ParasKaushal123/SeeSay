"""Prompt management for chest X-ray report generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptConfig:
    """Frozen prompt configuration tied to a version string."""

    version: str
    system_message: str
    user_template: str
    max_new_tokens: int = 384


def chest_xray_prompt(indication: str | None = None) -> PromptConfig:
    """Return the default v1 prompt for chest X-ray report generation.
    
    Args:
        indication: Optional clinical indication to include in the prompt.
        
    Returns:
        Configured PromptConfig instance.
    """
    system_message = (
        "You are an expert radiologist. "
        "Generate a structured chest X-ray report based on the provided image."
    )
    
    user_template = (
        "Please provide a structured radiology report for this chest X-ray image.\n"
        "The report must include exactly two sections:\n"
        "Findings: Detail your visible observations. Express appropriate uncertainty.\n"
        "Impression: Provide a concise summary of the findings.\n"
        "Note: No prior study is provided for comparison."
    )
    
    if indication:
        user_template = f"Clinical Indication: {indication}\n\n" + user_template
        
    return PromptConfig(
        version="chest_xray_v1",
        system_message=system_message,
        user_template=user_template,
    )
