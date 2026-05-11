import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from openai import OpenAI


DEFAULT_SKILL_CREATOR_BASE_URL = "https://your-skill-creator-endpoint/v1"
DEFAULT_SKILL_CREATOR_API_KEY = ""
DEFAULT_SKILL_CREATOR_MODEL = os.environ.get("SKILL_CREATOR_MODEL", "skill-creator-model")

SKILL_CREATOR_PATH = Path(__file__).resolve().parent / "skill_creator" / "SKILL.md"


class SkillCreatorClient:
    def __init__(
        self,
        max_new_skills_per_update: int = 1,
        max_completion_tokens: int = 4096,
    ):
        api_key = os.environ.get("SKILL_CREATOR_API_KEY", DEFAULT_SKILL_CREATOR_API_KEY)
        base_url = os.environ.get("SKILL_CREATOR_BASE_URL", DEFAULT_SKILL_CREATOR_BASE_URL)
        self.model = os.environ.get("SLIM_SKILL_CREATOR_MODEL", DEFAULT_SKILL_CREATOR_MODEL)
        self.max_completion_tokens = int(max_completion_tokens)
        self.max_new_skills_per_update = int(max_new_skills_per_update)

        if not SKILL_CREATOR_PATH.exists():
            raise FileNotFoundError(f"SLIM skill creator prompt not found: {SKILL_CREATOR_PATH}")

        self.system_prompt = SKILL_CREATOR_PATH.read_text(encoding="utf-8")
        self.client = None
        if api_key and base_url and "your-skill-creator-endpoint" not in base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            print(
                "[SLIMSkillCreator] Skill expansion requires SKILL_CREATOR_BASE_URL "
                "and SKILL_CREATOR_API_KEY; set them before running paper-faithful expansion."
            )

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        focus_skill: Optional[Dict] = None,
        task_type: Optional[str] = None,
    ) -> List[Dict]:
        if not failed_trajectories:
            return []
        if not task_type or task_type == "general":
            return []

        if self.client is None:
            raise RuntimeError(
                "SLIM expansion requested but skill-creator service is not configured. "
                "Set SKILL_CREATOR_BASE_URL, SKILL_CREATOR_API_KEY, and SKILL_CREATOR_MODEL, "
                "or disable expansion by setting ++env.slim_memory.max_new_skills=0."
            )

        user_prompt = self._build_user_prompt(
            failed_trajectories=failed_trajectories,
            current_skills=current_skills,
            focus_skill=focus_skill,
            task_type=task_type,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=self.max_completion_tokens,
        )
        content = response.choices[0].message.content or ""
        parsed = self._parse_skill_markdown(content, failed_trajectories, task_type)
        return [parsed] if parsed else []

    def _build_user_prompt(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        focus_skill: Optional[Dict],
        task_type: Optional[str],
    ) -> str:
        existing_titles = []
        for skill in current_skills.get("general_skills", []):
            existing_titles.append(skill.get("title", ""))
        for group, skills in current_skills.get("task_specific_skills", {}).items():
            for skill in skills:
                existing_titles.append(f"[{group}] {skill.get('title', '')}")

        examples = []
        for idx, traj in enumerate(failed_trajectories[:5], 1):
            recent_steps = traj.get("trajectory", [])[-5:]
            formatted_steps = []
            for step in recent_steps:
                action = step.get("action", "")
                obs = step.get("observation", "")[:300]
                formatted_steps.append(f"- action: {action}\n  observation: {obs}")
            examples.append(
                f"Failure Example {idx}\n"
                f"Task: {traj.get('task', '')}\n"
                f"Task Type: {traj.get('task_type', '')}\n"
                f"Selected Skills: {traj.get('selected_skill_ids', [])}\n"
                f"Recent Steps:\n" + "\n".join(formatted_steps)
            )

        focus_block = ""
        if focus_skill:
            focus_block = (
                "Insufficient existing skill:\n"
                f"- title: {focus_skill.get('title', '')}\n"
                f"- description: {focus_skill.get('description', '')}\n"
                f"- when_to_apply: {focus_skill.get('when_to_apply', '')}\n"
                f"- body: {focus_skill.get('body', focus_skill.get('principle', ''))}\n\n"
            )

        return f"""Create the single best NEW standalone skill artifact for these failures.

Target task type: {task_type or 'general'}

{focus_block}Existing skill titles to avoid duplicating:
{existing_titles}

Representative failed trajectories:
{os.linesep.join(examples)}

Requirements:
- Create ONE new skill as a complete SKILL.md artifact
- The skill MUST be task-specific for target task type "{task_type or 'unknown'}"
- Do NOT create a general/foundational skill
- Do not rewrite, merge, or edit any existing skill
- Make the skill procedural and reusable
- The skill should directly address the missing behavior revealed by these failures
- Include YAML frontmatter with name and description
- Use a specific name; avoid generic names like "goal", "dynamic skill", or "task skill"
- Include clear sections in the body
- Keep it concise but concrete
- Output ONLY the SKILL.md content and nothing else
"""

    def _parse_skill_markdown(
        self,
        content: str,
        failed_trajectories: List[Dict],
        task_type: Optional[str],
    ) -> Optional[Dict]:
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        cleaned = self._strip_code_fence(cleaned)
        if not cleaned:
            return None

        frontmatter = {}
        body = cleaned
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", cleaned, flags=re.DOTALL)
        if frontmatter_match:
            try:
                frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
            except Exception:
                frontmatter = {}
            body = frontmatter_match.group(2).strip()

        title = frontmatter.get("name") or self._extract_heading(body) or "dynamic_skill"
        description = frontmatter.get("description") or self._extract_first_paragraph(body)
        when_to_apply = self._extract_section(body, ["When to Use", "When to Apply", "Use When"]) or description
        principle = self._extract_section(body, ["Core Principles", "Principles", "Guidelines"])
        if not principle:
            principle = self._extract_first_paragraph(body)

        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "dynamic_skill"
        inferred_task_type = task_type or (failed_trajectories[0].get("task_type") if failed_trajectories else "general")
        if self._is_low_quality_skill(slug=slug, title=title, description=description, task_type=inferred_task_type):
            return None
        return {
            "skill_id": f"dyn_{slug}",
            "title": title.replace("-", " ").strip().title(),
            "description": description,
            "principle": principle,
            "when_to_apply": when_to_apply,
            "body": body,
            "tags": [inferred_task_type] if inferred_task_type else [],
        }

    def _strip_code_fence(self, content: str) -> str:
        text = content.strip()
        fence_match = re.match(r"^```(?:[A-Za-z0-9_-]+)?\s*\n(.*?)\n```\s*$", text, flags=re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        text = re.sub(r"^```(?:[A-Za-z0-9_-]+)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
        return text.strip()

    def _is_low_quality_skill(self, slug: str, title: str, description: str, task_type: Optional[str]) -> bool:
        if not task_type or task_type == "general":
            return True
        generic_slugs = {
            "goal",
            "dynamic_skill",
            "task_skill",
            "general_skill",
            "skill",
            "task",
            "alfworld_skill",
        }
        if slug in generic_slugs:
            return True
        title_tokens = [tok for tok in re.split(r"[^a-z0-9]+", title.lower()) if tok]
        if len(title_tokens) < 3:
            return True
        if len((description or "").strip()) < 40:
            return True
        return False

    def _extract_heading(self, body: str) -> Optional[str]:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return None

    def _extract_first_paragraph(self, body: str) -> str:
        parts = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        for part in parts:
            if not part.startswith("#"):
                return re.sub(r"\s+", " ", part)[:240]
        return ""

    def _extract_section(self, body: str, headings: List[str]) -> Optional[str]:
        lines = body.splitlines()
        for idx, line in enumerate(lines):
            normalized = line.strip("# ").strip().lower()
            if normalized in {heading.lower() for heading in headings}:
                collected = []
                for next_line in lines[idx + 1 :]:
                    if next_line.strip().startswith("#"):
                        break
                    collected.append(next_line)
                section = "\n".join(collected).strip()
                if section:
                    return section[:600]
        return None
