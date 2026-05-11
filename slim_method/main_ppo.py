import os
import time
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

from slim_method.bootstrap import configure_imports, report_import_sources

configure_imports()
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


VERL_CONFIG_DIR = os.environ.get(
    "VERL_CONFIG_DIR",
    str(Path(__file__).resolve().parents[1] / "verl" / "trainer" / "config"),
)

@hydra.main(config_path=VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    print(f"[SLIMMain] Loading trainer config from {VERL_CONFIG_DIR}/ppo_trainer.yaml")
    run_ppo(config)


def run_ppo(config) -> None:
    configure_imports()
    report_import_sources("[SLIMImport]")
    if config.env.get("use_slim_memory", False):
        slim_cfg = config.env.slim_memory
        print(
            "[SLIMMain] Enabled dynamic Skill Lifecycle Management "
            f"(routing={slim_cfg.get('retrieval_mode', 'embedding')}, top_k={slim_cfg.get('top_k', 3)}, "
            f"tau_embedding={slim_cfg.get('tau_embedding', 0.45)}, "
            f"general_group_audit={slim_cfg.get('max_audit_general_groups', 1)}, "
            f"task_audit={slim_cfg.get('max_audit_task_specific_skills', 4)}, "
            f"lifecycle={slim_cfg.get('enable_lifecycle_update', True)}, "
            f"probe={slim_cfg.get('enable_internalization_probe', True)})"
        )
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        env_vars = dict(runtime_env.get("env_vars", {}))
        env_vars["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
        runtime_env["env_vars"] = env_vars
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        configure_imports()
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        from agent_system.environments import make_envs
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from agent_system.reward_manager import EpisodeRewardManager
        from slim_method.trainer import SlimRayPPOTrainer

        report_import_sources("[SLIMTaskRunnerImport]")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        print("[SLIMTaskRunner] Using VERL package from configured PYTHONPATH")

        start_time = time.perf_counter()
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        print(f"[SLIMTaskRunnerTiming] copy_to_local took {time.perf_counter() - start_time:.2f}s")

        env_start = time.perf_counter()
        env_outputs = make_envs(config)
        print(f"[SLIMTaskRunnerTiming] make_envs took {time.perf_counter() - env_start:.2f}s")
        if len(env_outputs) == 2:
            envs, val_envs = env_outputs
            test_envs = None
        else:
            envs, val_envs, test_envs = env_outputs

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id}

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        test_files = config.data.get("test_files", None)
        test_dataset = (
            create_rl_dataset(test_files, config.data, tokenizer, processor)
            if test_files is not None
            else None
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = SlimRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            test_envs=test_envs,
        )
        init_workers_start = time.perf_counter()
        print("[SLIMTaskRunnerTiming] trainer.init_workers starting")
        trainer.init_workers()
        print(f"[SLIMTaskRunnerTiming] trainer.init_workers finished in {time.perf_counter() - init_workers_start:.2f}s")
        trainer.fit()


if __name__ == "__main__":
    main()
