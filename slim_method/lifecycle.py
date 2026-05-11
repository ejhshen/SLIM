import json
import os
from collections import Counter
from statistics import mean
from typing import Any, Callable, Dict, List

from .memory import GENERAL_SKILL_GROUP_ID
from .skill_creator_client import SkillCreatorClient


class SlimLifecycleManager:
    def __init__(
        self,
        tau_keep: float = 0.03,
        tau_retire: float = 0.001,
        patience: int = 3,
        n_min: int = 30,
        tau_expand: float = 0.40,
        n_expand: int = 20,
        utility_ema_beta: float = 0.9,
        max_new_skills: int = 2,
        max_bucket_size: int = 32,
        max_audit_skills: int = 8,
        max_audit_general_groups: int = 1,
        max_audit_task_specific_skills: int = 4,
        enforce_expansion_budget: bool = False,
    ):
        self.tau_keep = float(tau_keep)
        self.tau_retire = float(tau_retire)
        self.patience = int(patience)
        self.n_min = int(n_min)
        self.tau_expand = float(tau_expand)
        self.n_expand = int(n_expand)
        self.utility_ema_beta = float(utility_ema_beta)
        self.max_new_skills = int(max_new_skills)
        self.max_bucket_size = int(max_bucket_size)
        self.max_audit_skills = int(max_audit_skills)
        self.max_audit_general_groups = int(max_audit_general_groups)
        self.max_audit_task_specific_skills = int(max_audit_task_specific_skills)
        self.enforce_expansion_budget = bool(enforce_expansion_budget)
        self.skill_creator = SkillCreatorClient(max_new_skills_per_update=max_new_skills)
        self.history: List[Dict[str, Any]] = []
        print(
            "[SLIMLifecycle] Configured lifecycle manager "
            f"(tau_keep={self.tau_keep}, tau_retire={self.tau_retire}, "
            f"patience={self.patience}, n_min={self.n_min}, "
            f"tau_expand={self.tau_expand}, n_expand={self.n_expand}, "
            f"ema_beta={self.utility_ema_beta}, max_new_skills={self.max_new_skills}, "
            f"max_audit_general_groups={self.max_audit_general_groups}, "
            f"max_audit_task_specific_skills={self.max_audit_task_specific_skills}, "
            f"enforce_expansion_budget={self.enforce_expansion_budget})"
        )

    @classmethod
    def from_config(cls, cfg):
        return cls(
            tau_keep=cfg.get("tau_keep", 0.03),
            tau_retire=cfg.get("tau_retire", 0.001),
            patience=cfg.get("patience", 3),
            n_min=cfg.get("n_min", 30),
            tau_expand=cfg.get("tau_expand", 0.40),
            n_expand=cfg.get("n_expand", 20),
            utility_ema_beta=cfg.get("utility_ema_beta", 0.9),
            max_new_skills=cfg.get("max_new_skills", 2),
            max_bucket_size=cfg.get("max_bucket_size", 32),
            max_audit_skills=cfg.get("max_audit_skills", 8),
            max_audit_general_groups=cfg.get("max_audit_general_groups", 1),
            max_audit_task_specific_skills=cfg.get("max_audit_task_specific_skills", 4),
            enforce_expansion_budget=cfg.get("enforce_expansion_budget", False),
        )

    def _update_failure_buckets(self, memory, trajectory_records: List[Dict[str, Any]]):
        for record in trajectory_records:
            if record.get("success", False):
                continue
            failure_item = {
                "task": record.get("input", "")[:500],
                "task_type": record.get("task_type", "unknown"),
                "selected_skill_ids": list(record.get("selected_skill_ids", [])),
                "trajectory": [{"action": record.get("output", "")[:500], "observation": ""}],
            }
            for skill_id in record.get("selected_skill_ids", []):
                if skill_id == GENERAL_SKILL_GROUP_ID:
                    continue
                skill_record = memory.get_skill_record(skill_id) if skill_id in memory.records else None
                if not skill_record or skill_record.get("scope") != "task_specific":
                    continue
                memory.record_failure(skill_id, failure_item, max_bucket_size=self.max_bucket_size)

    def _perf(self, records: List[Dict[str, Any]]) -> float:
        if not records:
            return 0.0
        return float(mean(1.0 if record.get("success", False) else 0.0 for record in records))

    def _has_covering_skill(self, memory, skill_id: str, task_type: str) -> bool:
        if skill_id == GENERAL_SKILL_GROUP_ID:
            return False
        target = memory.get_skill_record(skill_id)
        for candidate_id in memory.get_active_skill_ids():
            if candidate_id == skill_id:
                continue
            candidate = memory.get_skill_record(candidate_id)
            if candidate.get("scope") != target.get("scope"):
                continue
            if target.get("scope") == "task_specific" and candidate.get("task_type") != task_type:
                continue
            if float(candidate.get("utility_ema", 0.0)) >= self.tau_keep:
                return True
        return False

    def _candidate_sort_key(self, memory, usage_counter: Counter, skill_id: str):
        record = memory.get_skill_record(skill_id)
        return (
            -int(usage_counter.get(skill_id, 0)),
            float(record.get("utility_ema", 0.0)),
            -int(record.get("low_utility_streak", 0)),
            skill_id,
        )

    def _audit_skill(
        self,
        memory,
        skill_id: str,
        routed_records: List[Dict[str, Any]],
        audit_callback: Callable[[str, List[int]], float],
    ) -> Dict[str, Any]:
        dataset_indices = sorted({int(record["dataset_index"]) for record in routed_records})
        perf_with = self._perf(routed_records)
        perf_without = float(audit_callback(skill_id, dataset_indices))
        utility = perf_with - perf_without

        skill_record = memory.get_skill_record(skill_id)
        prev_ema = float(skill_record.get("utility_ema", 0.0))
        if skill_record.get("usage_count", 0) == 0:
            utility_ema = utility
        else:
            utility_ema = self.utility_ema_beta * prev_ema + (1.0 - self.utility_ema_beta) * utility

        low_streak = int(skill_record.get("low_utility_streak", 0))
        if utility_ema < self.tau_retire:
            low_streak += 1
        else:
            low_streak = 0

        memory.set_skill_utility(skill_id, utility_ema=utility_ema, low_utility_streak=low_streak)

        updated_record = memory.get_skill_record(skill_id)
        should_retire = (
            utility_ema < self.tau_retire
            and low_streak >= self.patience
            and int(updated_record.get("usage_count", 0)) >= self.n_min
        )
        if should_retire:
            memory.retire_skill(skill_id)

        return {
            "skill_id": skill_id,
            "perf_with": perf_with,
            "perf_without": perf_without,
            "utility": utility,
            "utility_ema": memory.get_skill_record(skill_id).get("utility_ema", 0.0),
            "usage_count": memory.get_skill_record(skill_id).get("usage_count", 0),
            "low_utility_streak": memory.get_skill_record(skill_id).get("low_utility_streak", 0),
            "failure_bucket_size": len(memory.get_skill_record(skill_id).get("failure_bucket", [])),
            "status": memory.get_skill_record(skill_id).get("status", "active"),
            "decision": "retire" if should_retire else "retain",
        }

    def update(
        self,
        memory,
        trajectory_records: List[Dict[str, Any]],
        audit_callback: Callable[[str, List[int]], float],
        output_dir: str,
        global_step: int,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        self._update_failure_buckets(memory, trajectory_records)

        usage_counter = Counter()
        for record in trajectory_records:
            for skill_id in record.get("selected_skill_ids", []):
                usage_counter[skill_id] += 1

        general_group = None
        general_group_skipped_reason = None
        if self.max_audit_general_groups <= 0:
            general_group_skipped_reason = "disabled_by_budget"
        elif not memory.is_general_group_active():
            general_group_skipped_reason = "retired"
        elif not memory.has_active_general_skills():
            general_group_skipped_reason = "no_active_general_skills"
        elif usage_counter.get(GENERAL_SKILL_GROUP_ID, 0) <= 0:
            general_group_skipped_reason = "not_routed"

        audited_task_specific = []
        retained_task_specific = []
        retired_task_specific = []
        expanded_task_specific = []
        task_specific_skipped_reason = None
        expansion_budget_remaining = self.max_new_skills if self.enforce_expansion_budget else None

        if general_group_skipped_reason is None:
            routed_records = [
                record
                for record in trajectory_records
                if GENERAL_SKILL_GROUP_ID in record.get("selected_skill_ids", [])
            ]
            if routed_records:
                general_group = self._audit_skill(
                    memory=memory,
                    skill_id=GENERAL_SKILL_GROUP_ID,
                    routed_records=routed_records,
                    audit_callback=audit_callback,
                )
            else:
                general_group_skipped_reason = "not_routed"

        task_usage_counter = Counter()
        for skill_id, count in usage_counter.items():
            if skill_id == GENERAL_SKILL_GROUP_ID:
                continue
            if skill_id not in memory.records:
                continue
            skill_record = memory.get_skill_record(skill_id)
            if skill_record.get("scope") != "task_specific" or skill_record.get("status") != "active":
                continue
            task_usage_counter[skill_id] = count

        candidate_skill_ids = sorted(
            task_usage_counter.keys(),
            key=lambda skill_id: self._candidate_sort_key(memory, task_usage_counter, skill_id),
        )[: max(0, self.max_audit_task_specific_skills)]
        if not candidate_skill_ids:
            task_specific_skipped_reason = "no_candidates"

        known_task_types = set(memory.export_skills(include_retired=True).get("task_specific_skills", {}).keys())

        for skill_id in candidate_skill_ids:
            routed_records = [record for record in trajectory_records if skill_id in record.get("selected_skill_ids", [])]
            if not routed_records:
                continue

            audit_entry = self._audit_skill(
                memory=memory,
                skill_id=skill_id,
                routed_records=routed_records,
                audit_callback=audit_callback,
            )
            if audit_entry["decision"] == "retire":
                retired_task_specific.append(skill_id)
            else:
                retained_task_specific.append(skill_id)

            bucket = list(memory.get_skill_record(skill_id).get("failure_bucket", []))
            task_counter = Counter(item.get("task_type", "general") for item in bucket)
            dominant_task = task_counter.most_common(1)[0][0] if task_counter else memory.get_skill_record(skill_id).get("task_type", "general")
            needs_expansion = (
                bucket
                and audit_entry["perf_with"] < self.tau_expand
                and len(bucket) >= self.n_expand
                and audit_entry["utility_ema"] < self.tau_keep
                and not self._has_covering_skill(memory, skill_id, dominant_task)
            )
            if needs_expansion and self.max_new_skills <= 0:
                print(f"[SLIMLifecycle] Skip expansion for {skill_id}: max_new_skills <= 0")
            elif (
                needs_expansion
                and self.enforce_expansion_budget
                and expansion_budget_remaining is not None
                and expansion_budget_remaining <= 0
            ):
                print(
                    f"[SLIMLifecycle] Skip expansion for {skill_id}: "
                    f"per-audit expansion budget exhausted (B={self.max_new_skills})"
                )
            elif needs_expansion and dominant_task in known_task_types:
                new_skills = self.skill_creator.analyze_failures(
                    failed_trajectories=bucket,
                    current_skills=memory.export_skills(include_retired=False, include_state=True),
                    focus_skill=memory.get_skill_record(skill_id),
                    task_type=dominant_task,
                )
                if (
                    self.enforce_expansion_budget
                    and expansion_budget_remaining is not None
                ):
                    new_skills = new_skills[: max(0, expansion_budget_remaining)]
                if new_skills:
                    added = memory.add_skills(new_skills, category=dominant_task, parent_skill_id=skill_id)
                    if added:
                        if (
                            self.enforce_expansion_budget
                            and expansion_budget_remaining is not None
                        ):
                            expansion_budget_remaining -= added
                        expanded_task_specific.extend([skill["skill_id"] for skill in new_skills[:added]])
                        memory.shrink_failure_bucket(skill_id, keep_last=max(1, self.n_expand // 2))
            elif needs_expansion:
                print(
                    f"[SLIMLifecycle] Skip expansion for {skill_id}: "
                    f"dominant_task={dominant_task} is not a known task-specific category"
                )

            audited_task_specific.append(audit_entry)

        summary = {
            "global_step": int(global_step),
            "general_group": general_group,
            "general_group_skipped_reason": general_group_skipped_reason,
            "audited_task_specific_skills": audited_task_specific,
            "task_specific_skipped_reason": task_specific_skipped_reason,
            "retained_task_specific_skills": retained_task_specific,
            "retired_task_specific_skills": retired_task_specific,
            "expanded_task_specific_skills": expanded_task_specific,
            "audited_skills": audited_task_specific,
            "retained_skills": retained_task_specific,
            "retired_skills": retired_task_specific,
            "expanded_skills": expanded_task_specific,
            "enforce_expansion_budget": self.enforce_expansion_budget,
            "max_new_skills_per_audit": self.max_new_skills,
            "expansion_budget_remaining": expansion_budget_remaining,
            "active_general_group": memory.is_general_group_active() and memory.has_active_general_skills(),
            "active_skill_ids": memory.get_active_skill_ids(),
            "retired_skill_ids": memory.get_retired_skill_ids(),
            "active_task_specific_skill_ids": memory.get_active_task_specific_skill_ids(),
            "retired_task_specific_skill_ids": memory.get_retired_task_specific_skill_ids(),
        }
        self.history.append(summary)

        report_path = os.path.join(output_dir, f"slim_lifecycle_step{global_step}.json")
        active_path = os.path.join(output_dir, f"slim_active_skills_step{global_step}.json")
        latest_state_path = os.path.join(output_dir, "slim_state_latest.json")
        latest_active_path = os.path.join(output_dir, "slim_active_skills_latest.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        memory.save_active_skills(active_path)
        memory.save_state(latest_state_path)
        memory.save_active_skills(latest_active_path)
        print(
            "[SLIMLifecycle] step="
            f"{global_step} general_group={general_group['decision'] if general_group else general_group_skipped_reason} "
            f"task_audited={len(audited_task_specific)} retained={len(retained_task_specific)} "
            f"retired={len(retired_task_specific)} expanded={len(expanded_task_specific)} "
            f"active_task_specific={len(memory.get_active_task_specific_skill_ids())}"
        )

        metrics = {
            "slim/active_skill_count": len(memory.get_active_skill_ids()),
            "slim/retired_skill_count": len(memory.get_retired_skill_ids()),
            "slim/active_general_group": 1.0 if summary["active_general_group"] else 0.0,
            "slim/active_task_specific_skill_count": len(memory.get_active_task_specific_skill_ids()),
            "slim/retired_task_specific_skill_count": len(memory.get_retired_task_specific_skill_ids()),
            "slim/expanded_skill_count": len(expanded_task_specific),
            "slim/expanded_task_specific_skill_count": len(expanded_task_specific),
            "slim/audited_skill_count": len(audited_task_specific) + (1 if general_group else 0),
            "slim/audited_task_specific_skill_count": len(audited_task_specific),
        }
        if general_group:
            metrics["slim/general_group_utility"] = float(general_group["utility"])
            metrics["slim/general_group_utility_ema"] = float(general_group["utility_ema"])
        if audited_task_specific:
            metrics["slim/mean_utility"] = float(mean(item["utility"] for item in audited_task_specific))
            metrics["slim/mean_utility_ema"] = float(mean(item["utility_ema"] for item in audited_task_specific))
            metrics["slim/task_specific_mean_utility"] = metrics["slim/mean_utility"]
            metrics["slim/task_specific_mean_utility_ema"] = metrics["slim/mean_utility_ema"]

        return {
            "summary": summary,
            "metrics": metrics,
            "paths": {
                "report_path": report_path,
                "active_path": active_path,
                "latest_state_path": latest_state_path,
                "latest_active_path": latest_active_path,
            },
        }
