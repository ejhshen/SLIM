#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_REPO_ROOT="${BASE_REPO_ROOT:-}"
VERL_CONFIG_DIR="${VERL_CONFIG_DIR:-${PROJECT_ROOT}/verl/trainer/config}"

: "${MODEL_PATH:?Set MODEL_PATH to the base HuggingFace model path}"
: "${ALFWORLD_DATA_DIR:?Set ALFWORLD_DATA_DIR to prepared ALFWorld parquet directory}"
: "${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH to the embedding model path}"

INITIAL_SKILLS="${INITIAL_SKILLS:-${PROJECT_ROOT}/data/initial_skills/alfworld/skills.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/alfworld_slim_full}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
EXP_NAME="${EXP_NAME:-alfworld_slim_full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:6000/v1/embeddings}"
EMBEDDING_PORT="${EMBEDDING_PORT:-6000}"
SKILL_CREATOR_MODEL="${SKILL_CREATOR_MODEL:-skill-creator-model}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console]}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export PROJECT_ROOT BASE_REPO_ROOT REPO_ROOT VERL_CONFIG_DIR CUDA_VISIBLE_DEVICES
export REPO_ROOT="${BASE_REPO_ROOT:-${PROJECT_ROOT}}"
if [[ -n "${BASE_REPO_ROOT}" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}:${BASE_REPO_ROOT}:${BASE_REPO_ROOT}/agent_system/environments/env_package/alfworld:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/agent_system/environments/env_package/alfworld:${PYTHONPATH:-}"
fi
export SKILL_CREATOR_MODEL

python3 -m slim_method.embedding_service start \
  --model-path "${EMBEDDING_MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${EMBEDDING_PORT}" \
  --workdir "${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"
python3 -m slim_method.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${ALFWORLD_DATA_DIR}/text/train.parquet" \
  data.val_files="${ALFWORLD_DATA_DIR}/text/val.parquet" \
  data.test_files="${ALFWORLD_DATA_DIR}/text/test.parquet" \
  data.train_batch_size=16 \
  data.val_batch_size=32 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=False \
  data.truncation=right \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  algorithm.use_kl_in_reward=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=50 \
  env.history_length=50 \
  env.rollout.n=8 \
  env.resources_per_worker.num_cpus=0.1 \
  +env.use_slim_memory=True \
  +env.slim_memory.skills_json_path="${INITIAL_SKILLS}" \
  +env.slim_memory.state_path="${OUTPUT_DIR}/slim_state_latest.json" \
  +env.slim_memory.log_dir="${OUTPUT_DIR}/lifecycle" \
  +env.slim_memory.retrieval_mode=embedding \
  +env.slim_memory.embedding_url="${EMBEDDING_URL}" \
  +env.slim_memory.embedding_model="${EMBEDDING_MODEL_PATH}" \
  +env.slim_memory.tau_embedding=0.45 \
  +env.slim_memory.top_k=3 \
  +env.slim_memory.enable_lifecycle_update=True \
  +env.slim_memory.lifecycle_update_freq=5 \
  +env.slim_memory.max_audit_general_groups=1 \
  +env.slim_memory.max_audit_task_specific_skills=4 \
  +env.slim_memory.enforce_expansion_budget=False \
  +env.slim_memory.enable_embedding_dedup=False \
  +env.slim_memory.dedup_similarity_threshold=0.92 \
  +env.slim_memory.tau_retire=0.001 \
  +env.slim_memory.patience=3 \
  +env.slim_memory.n_min=30 \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=SLIM_alfworld \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.test_after_train=True \
  trainer.val_before_train=True \
  trainer.total_epochs=60 \
  trainer.total_training_steps=60 \
  ${EXTRA_ARGS} 2>&1 | tee "${LOG_DIR}/${EXP_NAME}.log"
