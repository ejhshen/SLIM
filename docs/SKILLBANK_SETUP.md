# Skill Bank Setup

SLIM expects a JSON skill bank with two groups:

- `general_skills`: foundational skills inserted together as one virtual
  lifecycle group `__general_skills__`.
- `task_specific_skills`: a mapping from task type to a list of skills.

Each skill should include at least:

```json
{
  "skill_id": "unique_id",
  "title": "Human-readable title",
  "description": "When and why the skill is useful.",
  "principle": "Short procedural guidance.",
  "when_to_apply": "Task condition for retrieval.",
  "body": "Full prompt text inserted when selected.",
  "tags": ["task_type"]
}
```

In the embedding-retrieval release configuration:

- General skills are always included while the general group is active.
- Task-specific skills are retrieved only within the detected task type.
- Retrieval uses cosine similarity with `tau_embedding=0.45` and `top_k=3`.
- Lifecycle audit budget is benchmark-specific in the release scripts:
  ALFWorld audits at most one general group plus four task-specific skills;
  Search-QA audits at most one general group plus eleven task-specific skills.
