import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from .bootstrap import configure_imports

configure_imports()
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn

from .lifecycle import SlimLifecycleManager


class SlimRayPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slim_enabled = bool(self.config.env.get("use_slim_memory", False))
        self.slim_lifecycle = None
        if self.slim_enabled:
            self.slim_lifecycle = SlimLifecycleManager.from_config(self.config.env.slim_memory)
            slim_cfg = self.config.env.slim_memory
            lifecycle_freq = int(slim_cfg.get("lifecycle_update_freq", self.config.trainer.test_freq))
            validation_freq = int(self.config.trainer.test_freq)
            if lifecycle_freq != validation_freq:
                print(
                    "[SLIMTrainer] WARNING: env.slim_memory.lifecycle_update_freq is bound to "
                    "the validation interval. SLIM only audits immediately after validation, "
                    f"so trainer.test_freq={validation_freq} is used and lifecycle_update_freq={lifecycle_freq} "
                    "is informational only. Set them equal to match the paper audit interval."
                )
            print(
                "[SLIMTrainer] SLIM enabled "
                f"(routing={slim_cfg.get('retrieval_mode', 'embedding')}, top_k={slim_cfg.get('top_k', 3)}, "
                f"tau_embedding={slim_cfg.get('tau_embedding', 0.45)}, "
                f"validation_audit_freq={validation_freq}, "
                f"general_group_audit={slim_cfg.get('max_audit_general_groups', 1)}, "
                f"task_audit={slim_cfg.get('max_audit_task_specific_skills', 4)}, "
                f"lifecycle={slim_cfg.get('enable_lifecycle_update', True)}, "
                f"probe={slim_cfg.get('enable_internalization_probe', True)})"
            )

    def _run_validation_rollout(self, test_data, envs):
        test_batch = DataProto.from_single_dict(test_data)
        test_batch = test_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
            interleave=True,
        )

        if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            return None

        input_ids = test_batch.batch["input_ids"]
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "env_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")

        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }

        output_batch = self.traj_collector.multi_turn_loop(
            gen_batch=test_gen_batch,
            actor_rollout_wg=self.actor_rollout_wg,
            envs=envs,
            is_train=False,
        )

        output_ids = output_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        result = self.val_reward_fn(output_batch, return_dict=True)
        reward_tensor = result["reward_tensor"]
        scores = reward_tensor.sum(-1).cpu().tolist()
        return {
            "batch": output_batch,
            "reward_tensor": reward_tensor,
            "scores": scores,
            "input_texts": input_texts,
            "output_texts": output_texts,
        }

    def _extract_unique_trajectory_records(self, rollout_result, dataset_offset: int):
        batch = rollout_result["batch"]
        traj_uids = np.asarray(batch.non_tensor_batch["traj_uid"], dtype=object)
        unique_indices = sorted(np.unique(traj_uids, return_index=True)[1].tolist())
        selected_skill_ids = batch.non_tensor_batch.get("selected_skill_ids", np.array([[]] * len(traj_uids), dtype=object))
        task_types = batch.non_tensor_batch.get("slim_task_type", np.array(["unknown"] * len(traj_uids), dtype=object))
        local_indices = batch.non_tensor_batch.get("index", np.arange(len(traj_uids)))

        records = []
        for idx in unique_indices:
            raw_selected = selected_skill_ids[idx]
            if isinstance(raw_selected, np.ndarray):
                raw_selected = raw_selected.tolist()
            elif raw_selected is None:
                raw_selected = []
            local_idx = int(local_indices[idx])
            records.append(
                {
                    "traj_uid": str(traj_uids[idx]),
                    "input": rollout_result["input_texts"][local_idx],
                    "output": rollout_result["output_texts"][idx],
                    "score": float(rollout_result["scores"][idx]),
                    "success": float(rollout_result["scores"][idx]) > 0,
                    "selected_skill_ids": list(raw_selected),
                    "task_type": str(task_types[idx]),
                    "dataset_index": int(dataset_offset + local_idx),
                }
            )
        return records

    def _counterfactual_perf_without_skill(self, skill_id: str, dataset_indices):
        dataset_indices = sorted({int(idx) for idx in dataset_indices})
        if not dataset_indices:
            return 0.0

        target_batch_size = int(getattr(self.config.data, 'val_batch_size', len(dataset_indices)) or len(dataset_indices))
        if len(dataset_indices) >= target_batch_size:
            aligned_indices = dataset_indices[:target_batch_size]
        else:
            repeats = []
            cursor = 0
            while len(dataset_indices) + len(repeats) < target_batch_size:
                repeats.append(dataset_indices[cursor % len(dataset_indices)])
                cursor += 1
            aligned_indices = dataset_indices + repeats

        subset = [self.val_dataset[idx] for idx in aligned_indices]
        test_data = collate_fn(subset)
        memory = getattr(self.val_envs, "retrieval_memory", None)
        if memory is None:
            return 0.0
        with memory.temporarily_disabled(skill_id), memory.suspend_usage_tracking():
            rollout_result = self._run_validation_rollout(test_data, self.val_envs)
        if rollout_result is None:
            return 0.0
        records = self._extract_unique_trajectory_records(rollout_result, dataset_offset=0)
        if not records:
            return 0.0
        return float(np.mean([1.0 if record["success"] else 0.0 for record in records]))

    def _should_run_internalization_probe(self) -> bool:
        if not self.slim_enabled:
            return False
        probe_cfg = self.config.env.slim_memory
        if not probe_cfg.get("enable_internalization_probe", True):
            return False
        probe_freq = int(probe_cfg.get("probe_freq", 20))
        return probe_freq > 0 and self.global_steps > 0 and self.global_steps % probe_freq == 0

    def _run_internalization_probe(self, memory, trajectory_records, lifecycle_result):
        if not trajectory_records:
            return {}, None

        probe_cfg = self.config.env.slim_memory
        probe_top_k = int(probe_cfg.get("probe_top_k", 5))
        probe_min_selection_count = int(probe_cfg.get("probe_min_selection_count", 3))
        probe_min_utility = float(
            probe_cfg.get(
                "probe_min_utility",
                getattr(self.slim_lifecycle, "tau_keep", 0.0),
            )
        )
        probe_success_floor = float(probe_cfg.get("probe_success_floor", 0.5))
        probe_ratio_threshold = float(probe_cfg.get("probe_ratio_threshold", 0.8))

        retained_skill_ids = set(lifecycle_result["summary"].get("retained_skills", []))
        audited_by_skill = {
            item["skill_id"]: item
            for item in lifecycle_result["summary"].get("audited_skills", [])
        }

        usage_counter = Counter()
        records_by_skill = defaultdict(list)
        for record in trajectory_records:
            for skill_id in record.get("selected_skill_ids", []):
                usage_counter[skill_id] += 1
                records_by_skill[skill_id].append(record)

        candidate_rows = []
        for skill_id in retained_skill_ids:
            selection_count = usage_counter.get(skill_id, 0)
            if selection_count < probe_min_selection_count:
                continue
            skill_record = memory.get_skill_record(skill_id)
            utility_ema = float(skill_record.get("utility_ema", 0.0))
            if utility_ema < probe_min_utility:
                continue
            candidate_rows.append(
                (
                    selection_count,
                    int(skill_record.get("selection_count", 0)),
                    utility_ema,
                    skill_id,
                )
            )

        candidate_rows.sort(reverse=True)
        selected_skill_ids = [row[-1] for row in candidate_rows[:probe_top_k]]

        probe_results = []
        for skill_id in selected_skill_ids:
            routed_records = records_by_skill.get(skill_id, [])
            if not routed_records:
                continue

            audit_entry = audited_by_skill.get(skill_id)
            if audit_entry is not None:
                perf_with = float(audit_entry["perf_with"])
                perf_without = float(audit_entry["perf_without"])
                utility = float(audit_entry["utility"])
            else:
                perf_with = float(np.mean([1.0 if record["success"] else 0.0 for record in routed_records]))
                dataset_indices = sorted({int(record["dataset_index"]) for record in routed_records})
                perf_without = float(self._counterfactual_perf_without_skill(skill_id, dataset_indices))
                utility = perf_with - perf_without

            internalization_ratio = 1.0 if perf_with <= 1e-8 else perf_without / max(perf_with, 1e-8)
            external_dependence = utility
            skill_record = memory.get_skill_record(skill_id)
            probe_results.append(
                {
                    "skill_id": skill_id,
                    "title": skill_record.get("title", ""),
                    "task_type": skill_record.get("task_type"),
                    "selection_count_validation": usage_counter.get(skill_id, 0),
                    "selection_count_total": int(skill_record.get("selection_count", 0)),
                    "utility_ema": float(skill_record.get("utility_ema", 0.0)),
                    "perf_with": perf_with,
                    "perf_without": perf_without,
                    "external_dependence": external_dependence,
                    "internalization_ratio": internalization_ratio,
                    "candidate_label": (
                        "closer_to_internalized"
                        if perf_with >= probe_success_floor and internalization_ratio >= probe_ratio_threshold
                        else "still_externally_dependent"
                    ),
                }
            )

        if not probe_results:
            return {}, None

        output_dir = self.config.trainer.default_local_dir
        os.makedirs(output_dir, exist_ok=True)
        report = {
            "global_step": int(self.global_steps),
            "probe_top_k": probe_top_k,
            "probe_min_selection_count": probe_min_selection_count,
            "probe_min_utility": probe_min_utility,
            "probe_success_floor": probe_success_floor,
            "probe_ratio_threshold": probe_ratio_threshold,
            "skills": probe_results,
        }
        report_path = os.path.join(output_dir, f"slim_internalization_probe_step{self.global_steps}.json")
        latest_path = os.path.join(output_dir, "slim_internalization_probe_latest.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(
            "[SLIMProbe] step="
            f"{self.global_steps} skills={len(probe_results)} "
            f"mean_ratio={float(np.mean([item['internalization_ratio'] for item in probe_results])):.3f} "
            f"mean_external_dependence={float(np.mean([item['external_dependence'] for item in probe_results])):.3f}"
        )

        metrics = {
            "slim_probe/skill_count": len(probe_results),
            "slim_probe/mean_internalization_ratio": float(
                np.mean([item["internalization_ratio"] for item in probe_results])
            ),
            "slim_probe/mean_external_dependence": float(
                np.mean([item["external_dependence"] for item in probe_results])
            ),
            "slim_probe/mean_perf_with": float(np.mean([item["perf_with"] for item in probe_results])),
            "slim_probe/mean_perf_without": float(np.mean([item["perf_without"] for item in probe_results])),
        }
        return metrics, report_path

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}
        trajectory_records = []

        dataset_offset = 0

        for test_data in self.val_dataloader:
            rollout_result = self._run_validation_rollout(test_data, self.val_envs)
            if rollout_result is None:
                return {}

            batch = rollout_result["batch"]
            reward_tensor = rollout_result["reward_tensor"]
            trajectory_records.extend(self._extract_unique_trajectory_records(rollout_result, dataset_offset=dataset_offset))

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            tool_calling_list.append(batch.non_tensor_batch["tool_callings"])
            traj_uid_list.append(batch.non_tensor_batch["traj_uid"])

            for key in batch.non_tensor_batch.keys():
                if "success_rate" in key:
                    success_rate_dict.setdefault(key, []).append(batch.non_tensor_batch[key][0])

            dataset_offset += len(test_data["input_ids"])

        if trajectory_records:
            self._maybe_log_val_generations(
                inputs=[record["input"] for record in trajectory_records],
                outputs=[record["output"] for record in trajectory_records],
                scores=[record["score"] for record in trajectory_records],
            )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {key: np.mean(value) for key, value in success_rate_dict.items()}

        data_source_reward = {}
        for idx in range(reward_tensor.shape[0]):
            data_source_reward.setdefault(data_sources[idx], []).append(reward_tensor[idx].item())

        data_source_tool_calling = {}
        unique_idx = np.unique(traj_uids, return_index=True)[1]
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        for idx in range(unique_tool_callings.shape[0]):
            data_source_tool_calling.setdefault(unique_data_sources[idx], []).append(unique_tool_callings[idx].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/{data_source}/test_score"] = np.mean(rewards)
        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f"val/{data_source}/tool_call_count/mean"] = np.mean(tool_calls)
        for key, value in success_rate.items():
            metric_dict[f"val/{key}"] = value

        if self.slim_enabled and self.config.env.slim_memory.get("enable_lifecycle_update", True):
            memory = getattr(self.val_envs, "retrieval_memory", None)
            if memory is not None:
                lifecycle_result = self.slim_lifecycle.update(
                    memory=memory,
                    trajectory_records=trajectory_records,
                    audit_callback=self._counterfactual_perf_without_skill,
                    output_dir=self.config.trainer.default_local_dir,
                    global_step=self.global_steps,
                )
                metric_dict.update(lifecycle_result["metrics"])
                train_memory = getattr(self.envs, "retrieval_memory", None)
                if train_memory is not None:
                    train_memory.sync_from(memory)
                test_memory = getattr(self.test_envs, "retrieval_memory", None)
                if test_memory is not None:
                    test_memory.sync_from(memory)
                if self._should_run_internalization_probe():
                    probe_metrics, _probe_path = self._run_internalization_probe(
                        memory=memory,
                        trajectory_records=trajectory_records,
                        lifecycle_result=lifecycle_result,
                    )
                    metric_dict.update(probe_metrics)

        return metric_dict
