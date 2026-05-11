import os
from typing import Dict, List

from openai import OpenAI


class SkillUpdater:
    """Optional generic LLM client for skill-bank updates.

    Configure with SKILL_CREATOR_BASE_URL, SKILL_CREATOR_API_KEY, and
    SKILL_CREATOR_MODEL. The release code does not ship credentials.
    """

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("SKILL_CREATOR_API_KEY", "")
        base_url = os.environ.get("SKILL_CREATOR_BASE_URL", "")
        if not api_key or not base_url:
            raise RuntimeError(
                "SkillUpdater requires SKILL_CREATOR_API_KEY and SKILL_CREATOR_BASE_URL."
            )
        self.model = model or os.environ.get("SKILL_CREATOR_MODEL", "skill-creator-model")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def update(self, messages: List[Dict], **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
