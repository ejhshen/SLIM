import copy
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from agent_system.memory.base import BaseMemory


GENERAL_SKILL_GROUP_ID = "__general_skills__"


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "these",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "your",
    "you",
    "under",
    "than",
    "more",
    "less",
    "after",
    "before",
}


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9_\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    return [tok for tok in normalized.split() if tok and tok not in STOPWORDS]


def _normalize_title(text: str) -> str:
    return _normalize_text(text).replace(" ", "_")


def _dot(lhs: List[float], rhs: List[float]) -> float:
    return float(sum(a * b for a, b in zip(lhs, rhs)))


class SLIMMemory(BaseMemory):
    """Dynamic skill memory for SLIM."""

    def __init__(
        self,
        skills_json_path: str,
        top_k: int = 3,
        lambda_utility: float = 0.2,
        tau_route: float = 0.35,
        tau_embedding: float = 0.45,
        retrieval_mode: str = "embedding",
        embedding_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        state_path: Optional[str] = None,
        enable_embedding_dedup: bool = False,
        dedup_similarity_threshold: float = 0.92,
    ):
        if not os.path.exists(skills_json_path):
            raise FileNotFoundError(f"SLIM skills file not found: {skills_json_path}")

        with open(skills_json_path, "r", encoding="utf-8") as f:
            skills = json.load(f)

        if "task_specific_skills" not in skills and "query_type_skills" in skills:
            skills["task_specific_skills"] = skills["query_type_skills"]

        self.top_k = int(top_k)
        self.lambda_utility = float(lambda_utility)
        self.tau_route = float(tau_route)
        self.tau_embedding = float(tau_embedding)
        self.retrieval_mode = str(retrieval_mode or "embedding").lower()
        self.embedding_url = embedding_url or "http://127.0.0.1:6000/v1/embeddings"
        self.embedding_model = embedding_model or "models/embedding_model"
        self.state_path = state_path
        self.enable_embedding_dedup = bool(enable_embedding_dedup)
        self.dedup_similarity_threshold = float(dedup_similarity_threshold)
        self.common_mistakes = list(skills.get("common_mistakes", []))
        self.disabled_skill_ids = set()
        self._suspend_usage = False
        self._embedding_cache: Dict[str, List[float]] = {}
        self._retrieval_calls = 0
        self.general_group_state: Dict[str, Any] = {
            "skill_id": GENERAL_SKILL_GROUP_ID,
            "scope": "general_group",
            "status": "active",
            "utility_ema": 0.0,
            "usage_count": 0,
            "selection_count": 0,
            "low_utility_streak": 0,
        }

        self.records: Dict[str, Dict[str, Any]] = {}
        self._task_specific_keys = list(skills.get("task_specific_skills", {}).keys())
        self._build_from_skill_json(skills)

        if state_path and os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.load_state_dict(state, rebuild_embeddings=False)
                print(f"[SLIMMemory] Restored lifecycle state from {state_path}")
            except Exception as exc:
                print(f"[SLIMMemory] Failed to restore state from {state_path}: {exc}")

        if self.retrieval_mode == "embedding":
            self._rebuild_embedding_cache()

        print(
            f"[SLIMMemory] Loaded {len(self.get_active_skill_ids())} active skills "
            f"({len(self.records)} total, routing={self.retrieval_mode}, top_k={self.top_k}, "
            f"tau_embedding={self.tau_embedding}, tau_route={self.tau_route}, "
            f"embedding_dedup={self.enable_embedding_dedup}, "
            f"dedup_similarity_threshold={self.dedup_similarity_threshold}, "
            f"general_group={self.general_group_state.get('status', 'active')}, "
            f"state_path={self.state_path or 'none'})"
        )

    def _build_from_skill_json(self, skills: Dict[str, Any]):
        for skill in skills.get("general_skills", []):
            self._register_skill(skill=skill, scope="general", task_type=None)

        for task_type, task_skills in skills.get("task_specific_skills", {}).items():
            for skill in task_skills:
                self._register_skill(skill=skill, scope="task_specific", task_type=task_type)

    def _skill_metadata_text(self, skill: Dict[str, Any], task_type: Optional[str]) -> str:
        description = skill.get("description", "")
        body = skill.get("body", skill.get("principle", ""))
        tags = list(skill.get("tags", []))
        return " ".join(
            part
            for part in [
                skill.get("title", ""),
                description or skill.get("principle", ""),
                skill.get("when_to_apply", ""),
                body,
                " ".join(tags),
                task_type or "",
            ]
            if part
        )

    def _skill_trigger_text(self, skill: Dict[str, Any]) -> str:
        return _normalize_text(skill.get("when_to_apply") or skill.get("description", ""))

    def _register_skill(self, skill: Dict[str, Any], scope: str, task_type: Optional[str]):
        skill_id = skill.get("skill_id")
        if not skill_id:
            raise ValueError(f"Skill missing skill_id: {skill}")

        description = skill.get("description", "")
        body = skill.get("body", skill.get("principle", ""))
        tags = list(skill.get("tags", []))
        metadata_text = self._skill_metadata_text(skill, task_type)

        self.records[skill_id] = {
            "skill_id": skill_id,
            "title": skill.get("title", ""),
            "description": description,
            "principle": skill.get("principle", ""),
            "body": body,
            "when_to_apply": skill.get("when_to_apply", ""),
            "tags": tags,
            "scope": scope,
            "task_type": task_type,
            "status": skill.get("status", "active"),
            "utility_ema": float(skill.get("utility_ema", 0.0)),
            "usage_count": int(skill.get("usage_count", 0)),
            "selection_count": int(skill.get("selection_count", 0)),
            "low_utility_streak": int(skill.get("low_utility_streak", 0)),
            "parent_skill_id": skill.get("parent_skill_id"),
            "origin": skill.get("origin", "initial"),
            "failure_bucket": list(skill.get("failure_bucket", [])),
            "metadata_text": metadata_text,
            "metadata_tokens": Counter(_tokenize(metadata_text)),
        }

    def _record_to_skill(self, record: Dict[str, Any], include_state: bool = False) -> Dict[str, Any]:
        skill = {
            "skill_id": record["skill_id"],
            "title": record["title"],
            "description": record.get("description", ""),
            "principle": record["principle"],
            "body": record.get("body", record["principle"]),
            "when_to_apply": record["when_to_apply"],
            "tags": list(record.get("tags", [])),
        }
        if include_state:
            skill.update(
                {
                    "status": record["status"],
                    "utility_ema": record["utility_ema"],
                    "usage_count": record["usage_count"],
                    "selection_count": record["selection_count"],
                    "low_utility_streak": record["low_utility_streak"],
                    "parent_skill_id": record.get("parent_skill_id"),
                    "origin": record.get("origin", "initial"),
                    "failure_bucket": record.get("failure_bucket", []),
                }
            )
        return skill

    def _all_active_records(self) -> List[Dict[str, Any]]:
        return [
            record
            for record in self.records.values()
            if record["status"] == "active" and record["skill_id"] not in self.disabled_skill_ids
        ]

    def _active_general_records(self) -> List[Dict[str, Any]]:
        if not self.is_general_group_active():
            return []
        return [
            record
            for record in self._all_active_records()
            if record.get("scope") == "general"
        ]

    def _active_task_records(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        records = [
            record
            for record in self._all_active_records()
            if record.get("scope") == "task_specific"
        ]
        if task_type is not None:
            records = [record for record in records if record.get("task_type") == task_type]
        return records

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = json.dumps(
            {"model": self.embedding_model, "input": texts},
            ensure_ascii=True,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.embedding_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"SLIM embedding retrieval requires a live embedding service at {self.embedding_url}: {exc}"
            ) from exc
        data = sorted(body.get("data", []), key=lambda item: int(item.get("index", 0)))
        return [list(item["embedding"]) for item in data]

    def _rebuild_embedding_cache(self):
        self._embedding_cache = {}
        task_records = [record for record in self.records.values() if record.get("scope") == "task_specific"]
        missing = [
            record
            for record in task_records
            if record.get("metadata_text") and record["skill_id"] not in self._embedding_cache
        ]
        if not missing:
            return
        vectors = self._embed_texts([record["metadata_text"] for record in missing])
        for record, vector in zip(missing, vectors):
            self._embedding_cache[record["skill_id"]] = vector
        print(
            f"[SLIMMemory] Cached embeddings for {len(self._embedding_cache)} task-specific skills "
            f"from {self.embedding_url}"
        )

    def _ensure_skill_embedding(self, record: Dict[str, Any]):
        skill_id = record["skill_id"]
        if skill_id in self._embedding_cache:
            return
        vectors = self._embed_texts([record.get("metadata_text", "")])
        if vectors:
            self._embedding_cache[skill_id] = vectors[0]

    def is_general_group_active(self) -> bool:
        return (
            self.general_group_state.get("status", "active") == "active"
            and GENERAL_SKILL_GROUP_ID not in self.disabled_skill_ids
        )

    def has_active_general_skills(self) -> bool:
        return any(
            record.get("scope") == "general"
            and record.get("status") == "active"
            and record["skill_id"] not in self.disabled_skill_ids
            for record in self.records.values()
        )

    def _detect_task_type(self, task_description: str) -> str:
        task_specific = set(self._task_specific_keys)
        goal = (task_description or "").lower()

        if "pick_and_place" in task_specific or "clean" in task_specific:
            look_light_markers = [
                "look at",
                "examine",
                "inspect",
            ]
            light_markers = [
                "desklamp",
                "desk lamp",
                "floorlamp",
                "floor lamp",
                "lamp",
                "light",
            ]
            if (
                "look_at_obj_in_light" in task_specific
                and any(marker in goal for marker in look_light_markers)
                and any(marker in goal for marker in light_markers)
            ):
                return "look_at_obj_in_light"
            if "clean" in task_specific and any(kw in goal for kw in ["clean", "cleaned", "wash", "washed", "rinse", "rinsed"]):
                return "clean"
            if "heat" in task_specific and any(kw in goal for kw in ["heat", "heated", "hot", "warm", "warmed", "microwave"]):
                return "heat"
            if "cool" in task_specific and any(kw in goal for kw in ["cool", "cooled", "chill", "chilled", "cold", "fridge", "refrigerator"]):
                return "cool"
            return "pick_and_place"

        if {"direct_retrieval", "multi_hop_reasoning", "entity_attribute_lookup", "compare"} & task_specific:
            if any(
                kw in goal
                for kw in [
                    " which ",
                    "older",
                    "born first",
                    "earlier",
                    "later",
                    "larger",
                    "smaller",
                    "more than",
                    "less than",
                ]
            ):
                return "compare"
            if any(kw in goal for kw in ["occupation", "architect", "nationality", "birthplace", "born in", "located in"]):
                return "entity_attribute_lookup"
            if any(kw in goal for kw in ["director", "author", "producer", "parent", "spouse", "founded", "invented"]):
                return "multi_hop_reasoning"
            return "direct_retrieval"

        return self._task_specific_keys[0] if self._task_specific_keys else "unknown"

    def _lexical_score(self, task_description: str, record: Dict[str, Any], detected_task_type: str) -> float:
        query_tokens = _tokenize(task_description)
        if not query_tokens:
            return 0.0

        query_counter = Counter(query_tokens)
        skill_counter = record["metadata_tokens"]
        if not skill_counter:
            return 0.0

        overlap = sum(min(query_counter[token], skill_counter.get(token, 0)) for token in query_counter)
        coverage = overlap / max(len(query_tokens), 1)

        query_vocab = set(query_counter)
        skill_vocab = set(skill_counter)
        jaccard = len(query_vocab & skill_vocab) / max(len(query_vocab | skill_vocab), 1)

        phrase_bonus = 0.0
        normalized_goal = _normalize_text(task_description)
        for phrase in [record["title"], record["when_to_apply"]]:
            normalized_phrase = _normalize_text(phrase)
            if normalized_phrase and normalized_phrase in normalized_goal:
                phrase_bonus += 0.15

        if record["scope"] == "general":
            phrase_bonus += 0.03
        if record["task_type"] and record["task_type"] == detected_task_type:
            phrase_bonus += 0.25

        return coverage + 0.5 * jaccard + phrase_bonus

    def retrieve(self, task_description: str, top_k: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        if top_k is None:
            top_k = self.top_k
        top_k = int(top_k)
        task_type = self._detect_task_type(task_description)

        general_records = self._active_general_records()
        task_candidates = self._active_task_records(task_type=task_type)
        selected_task_ids: List[str] = []
        similarity_scores: List[Dict[str, Any]] = []
        filtered_below_threshold = 0

        if self.retrieval_mode == "embedding":
            query_vector = self._embed_texts([task_description or ""])[0]
            scored = []
            for record in task_candidates:
                self._ensure_skill_embedding(record)
                similarity = _dot(query_vector, self._embedding_cache.get(record["skill_id"], []))
                scored.append((similarity, record["skill_id"]))
            scored.sort(key=lambda item: (-item[0], item[1]))
            filtered_below_threshold = sum(1 for score, _ in scored if score < self.tau_embedding)
            selected_task_ids = [
                skill_id
                for score, skill_id in scored
                if score >= self.tau_embedding
            ][:top_k]
            similarity_scores = [
                {"skill_id": skill_id, "score": score}
                for score, skill_id in scored[: max(top_k, 10)]
            ]
        elif self.retrieval_mode == "lexical":
            scored = []
            for record in task_candidates:
                relevance = self._lexical_score(task_description, record, task_type)
                scored.append((relevance, record["skill_id"]))
            scored.sort(key=lambda item: (-item[0], item[1]))
            filtered_below_threshold = sum(1 for score, _ in scored if score < self.tau_route)
            selected_task_ids = [
                skill_id
                for score, skill_id in scored
                if score >= self.tau_route
            ][:top_k]
            similarity_scores = [
                {"skill_id": skill_id, "score": score}
                for score, skill_id in scored[: max(top_k, 10)]
            ]
        else:
            raise ValueError(f"Unsupported SLIM retrieval_mode={self.retrieval_mode}")

        general_skill_ids = [record["skill_id"] for record in general_records]
        selected_ids = []
        if general_skill_ids:
            selected_ids.append(GENERAL_SKILL_GROUP_ID)
        selected_ids.extend(selected_task_ids)

        if not self._suspend_usage:
            if general_skill_ids:
                self.general_group_state["usage_count"] = int(self.general_group_state.get("usage_count", 0)) + 1
                self.general_group_state["selection_count"] = int(self.general_group_state.get("selection_count", 0)) + 1
            for skill_id in selected_task_ids:
                self.records[skill_id]["usage_count"] += 1
                self.records[skill_id]["selection_count"] += 1

        task_records = [self.records[skill_id] for skill_id in selected_task_ids]
        general_skills = [self._record_to_skill(record) for record in general_records]
        task_skills = [self._record_to_skill(record) for record in task_records]
        self._retrieval_calls += 1
        if self._retrieval_calls <= 5 or self._retrieval_calls % 100 == 0:
            print(
                "[SLIMMemory] retrieval "
                f"mode={self.retrieval_mode} task_type={task_type} "
                f"general_active={bool(general_skill_ids)} general_count={len(general_skill_ids)} "
                f"task_candidates={len(task_candidates)} selected_task={selected_task_ids} "
                f"filtered_below_threshold={filtered_below_threshold}"
            )

        return {
            "general_skills": general_skills,
            "task_specific_skills": task_skills,
            "mistakes_to_avoid": [],
            "task_type": task_type,
            "task_specific_examples": [],
            "retrieval_mode": f"slim_{self.retrieval_mode}",
            "selected_skill_ids": selected_ids,
            "general_skill_group_id": GENERAL_SKILL_GROUP_ID if general_skill_ids else None,
            "general_group_active": bool(general_skill_ids),
            "general_skill_ids": general_skill_ids,
            "task_specific_skill_ids": selected_task_ids,
            "task_specific_candidate_count": len(task_candidates),
            "task_specific_similarity_scores": similarity_scores,
            "task_specific_filtered_below_threshold": filtered_below_threshold,
            "selected_skills": [self._record_to_skill(record, include_state=True) for record in task_records],
        }

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        sections = []

        general_skills = retrieved_memories.get("general_skills", [])
        if general_skills:
            lines = [
                "### GENERAL SKILLS ###",
                "Here are some useful general strategies for solving the task effectively:",
            ]
            for idx, skill in enumerate(general_skills, 1):
                title = skill.get("title", "")
                principle = self._skill_principle(skill)
                line = f"{idx}. {title}: {principle}" if title else f"{idx}. {principle}"
                when = self._clean_skill_text(skill.get("when_to_apply", ""))
                if when:
                    line += f" Apply this when {when}"
                lines.append(line)
            sections.append("\n".join(lines))

        task_skills = retrieved_memories.get("task_specific_skills", [])
        if task_skills:
            task_type = retrieved_memories.get("task_type", "task")
            readable_task = task_type.replace("_", " ")
            lines = [
                f"### TASK: {task_type} ###",
                f"For {readable_task} tasks, apply these specific strategies:",
            ]
            for idx, skill in enumerate(task_skills, 1):
                title = skill.get("title", "")
                principle = self._skill_principle(skill)
                when = self._clean_skill_text(skill.get("when_to_apply", ""))
                line = f"{idx}. {title}: {principle}" if title else f"{idx}. {principle}"
                if when:
                    line += f" Apply this when {when}"
                lines.append(line)
            sections.append("\n".join(lines))

        if sections:
            return "\n\n".join(sections) + "\n\n"
        return ""

    def _skill_principle(self, skill: Dict[str, Any]) -> str:
        for key in ("principle", "description", "body"):
            text = self._clean_skill_text(skill.get(key, ""))
            if text:
                return text
        return "Use only when it directly helps choose the next admissible action."

    @staticmethod
    def _clean_skill_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def reset(self, batch_size: int):
        return None

    def store(self, record: Dict[str, List[Any]]):
        return None

    def fetch(self, step: int):
        return None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return list(self.records.values())[idx]

    def get_skill_record(self, skill_id: str) -> Dict[str, Any]:
        if skill_id == GENERAL_SKILL_GROUP_ID:
            return self.general_group_state
        return self.records[skill_id]

    def get_active_skill_ids(self) -> List[str]:
        return [record["skill_id"] for record in self.records.values() if record["status"] == "active"]

    def get_retired_skill_ids(self) -> List[str]:
        return [record["skill_id"] for record in self.records.values() if record["status"] == "retired"]

    def get_active_task_specific_skill_ids(self) -> List[str]:
        return [
            record["skill_id"]
            for record in self.records.values()
            if record["status"] == "active" and record.get("scope") == "task_specific"
        ]

    def get_retired_task_specific_skill_ids(self) -> List[str]:
        return [
            record["skill_id"]
            for record in self.records.values()
            if record["status"] == "retired" and record.get("scope") == "task_specific"
        ]

    def set_skill_utility(self, skill_id: str, utility_ema: float, low_utility_streak: int):
        if skill_id == GENERAL_SKILL_GROUP_ID:
            self.general_group_state["utility_ema"] = float(utility_ema)
            self.general_group_state["low_utility_streak"] = int(low_utility_streak)
            return
        record = self.records[skill_id]
        record["utility_ema"] = float(utility_ema)
        record["low_utility_streak"] = int(low_utility_streak)

    def retire_skill(self, skill_id: str):
        if skill_id == GENERAL_SKILL_GROUP_ID:
            self.general_group_state["status"] = "retired"
            return
        if skill_id in self.records:
            self.records[skill_id]["status"] = "retired"

    def activate_skill(self, skill_id: str):
        if skill_id == GENERAL_SKILL_GROUP_ID:
            self.general_group_state["status"] = "active"
            return
        if skill_id in self.records:
            self.records[skill_id]["status"] = "active"

    def record_failure(self, skill_id: str, failure_item: Dict[str, Any], max_bucket_size: int = 32):
        if skill_id == GENERAL_SKILL_GROUP_ID:
            return
        if skill_id not in self.records:
            return
        bucket = self.records[skill_id].setdefault("failure_bucket", [])
        bucket.append(copy.deepcopy(failure_item))
        if len(bucket) > max_bucket_size:
            del bucket[:-max_bucket_size]

    def shrink_failure_bucket(self, skill_id: str, keep_last: int = 8):
        if skill_id == GENERAL_SKILL_GROUP_ID:
            return
        if skill_id not in self.records:
            return
        bucket = self.records[skill_id].get("failure_bucket", [])
        if len(bucket) > keep_last:
            self.records[skill_id]["failure_bucket"] = bucket[-keep_last:]

    def add_skills(self, new_skills: List[Dict[str, Any]], category: str = "general", parent_skill_id: Optional[str] = None) -> int:
        if category == "general":
            print("[SLIMMemory] Skip adding generated general skill; SLIM expand is task-specific only.")
            return 0
        added = 0
        existing_titles = {_normalize_title(record.get("title", "")) for record in self.records.values()}
        existing_triggers = {
            self._skill_trigger_text(record)
            for record in self.records.values()
            if record.get("scope") == "task_specific" and record.get("task_type") == category
        }
        for skill in new_skills:
            skill_id = skill.get("skill_id")
            normalized_title = _normalize_title(skill.get("title", ""))
            if not skill_id or skill_id in self.records or (normalized_title and normalized_title in existing_titles):
                continue
            trigger_text = self._skill_trigger_text(skill)
            if trigger_text and trigger_text in existing_triggers:
                print(
                    f"[SLIMMemory] Skip duplicate generated skill by trigger text: "
                    f"{skill.get('title', skill_id)}"
                )
                continue
            scope = "general" if category == "general" else "task_specific"
            task_type = None if scope == "general" else category
            enriched = dict(skill)
            enriched_metadata_text = self._skill_metadata_text(enriched, task_type)
            if (
                self.enable_embedding_dedup
                and self.retrieval_mode == "embedding"
                and scope == "task_specific"
                and enriched_metadata_text
            ):
                new_vectors = self._embed_texts([enriched_metadata_text])
                new_vector = new_vectors[0] if new_vectors else []
                duplicate_id = None
                duplicate_score = 0.0
                for record in self.records.values():
                    if record.get("scope") != "task_specific" or record.get("task_type") != category:
                        continue
                    self._ensure_skill_embedding(record)
                    score = _dot(new_vector, self._embedding_cache.get(record["skill_id"], []))
                    if score >= self.dedup_similarity_threshold and score > duplicate_score:
                        duplicate_id = record["skill_id"]
                        duplicate_score = score
                if duplicate_id is not None:
                    print(
                        f"[SLIMMemory] Skip duplicate generated skill by embedding similarity: "
                        f"{skill.get('title', skill_id)} ~= {duplicate_id} ({duplicate_score:.3f})"
                    )
                    continue
            enriched["status"] = "active"
            enriched["utility_ema"] = 0.0
            enriched["usage_count"] = 0
            enriched["selection_count"] = 0
            enriched["low_utility_streak"] = 0
            enriched["parent_skill_id"] = parent_skill_id
            enriched["origin"] = "expanded"
            enriched["failure_bucket"] = []
            self._register_skill(enriched, scope=scope, task_type=task_type)
            if self.retrieval_mode == "embedding" and scope == "task_specific":
                self._ensure_skill_embedding(self.records[skill_id])
            existing_titles.add(normalized_title)
            if trigger_text:
                existing_triggers.add(trigger_text)
            added += 1
        return added

    def export_skills(self, include_retired: bool = False, include_state: bool = False) -> Dict[str, Any]:
        general = []
        task_specific = defaultdict(list)
        for record in self.records.values():
            if record["status"] != "active" and not include_retired:
                continue
            skill = self._record_to_skill(record, include_state=include_state)
            if record["scope"] == "general":
                if not include_retired and not self.is_general_group_active():
                    continue
                general.append(skill)
            else:
                task_specific[record["task_type"]].append(skill)
        exported = {
            "general_skills": general,
            "task_specific_skills": dict(task_specific),
            "common_mistakes": copy.deepcopy(self.common_mistakes),
        }
        if include_state:
            exported["general_group"] = copy.deepcopy(self.general_group_state)
        return exported

    def state_dict(self) -> Dict[str, Any]:
        return {
            "skills": self.export_skills(include_retired=True, include_state=True),
            "top_k": self.top_k,
            "lambda_utility": self.lambda_utility,
            "tau_embedding": self.tau_embedding,
            "tau_route": self.tau_route,
            "retrieval_mode": self.retrieval_mode,
            "enable_embedding_dedup": self.enable_embedding_dedup,
            "dedup_similarity_threshold": self.dedup_similarity_threshold,
            "general_group": copy.deepcopy(self.general_group_state),
        }

    def load_state_dict(self, state: Dict[str, Any], rebuild_embeddings: bool = True):
        skills = state.get("skills")
        if not skills:
            return
        self.top_k = int(state.get("top_k", self.top_k))
        self.lambda_utility = float(state.get("lambda_utility", self.lambda_utility))
        self.tau_embedding = float(state.get("tau_embedding", self.tau_embedding))
        self.tau_route = float(state.get("tau_route", self.tau_route))
        self.retrieval_mode = str(state.get("retrieval_mode", self.retrieval_mode)).lower()
        self.enable_embedding_dedup = bool(state.get("enable_embedding_dedup", self.enable_embedding_dedup))
        self.dedup_similarity_threshold = float(state.get("dedup_similarity_threshold", self.dedup_similarity_threshold))
        self.records = {}
        self.common_mistakes = list(skills.get("common_mistakes", []))
        self.general_group_state = copy.deepcopy(
            state.get("general_group") or skills.get("general_group") or self.general_group_state
        )
        self.general_group_state.setdefault("skill_id", GENERAL_SKILL_GROUP_ID)
        self.general_group_state.setdefault("scope", "general_group")
        self.general_group_state.setdefault("status", "active")
        self.general_group_state.setdefault("utility_ema", 0.0)
        self.general_group_state.setdefault("usage_count", 0)
        self.general_group_state.setdefault("selection_count", 0)
        self.general_group_state.setdefault("low_utility_streak", 0)
        self._task_specific_keys = list(skills.get("task_specific_skills", {}).keys())
        self._build_from_skill_json(skills)
        if rebuild_embeddings and self.retrieval_mode == "embedding":
            self._rebuild_embedding_cache()

    def sync_from(self, other: "SLIMMemory"):
        self.load_state_dict(other.state_dict(), rebuild_embeddings=False)
        if self.retrieval_mode == "embedding":
            self._embedding_cache = copy.deepcopy(getattr(other, "_embedding_cache", {}))

    def save_active_skills(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_skills(include_retired=False, include_state=True), f, indent=2)

    def save_state(self, path: Optional[str] = None):
        target_path = path or self.state_path
        if not target_path:
            return
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    @contextmanager
    def temporarily_disabled(self, skill_id: str):
        self.disabled_skill_ids.add(skill_id)
        try:
            yield self
        finally:
            self.disabled_skill_ids.discard(skill_id)

    @contextmanager
    def suspend_usage_tracking(self):
        previous = self._suspend_usage
        self._suspend_usage = True
        try:
            yield self
        finally:
            self._suspend_usage = previous
