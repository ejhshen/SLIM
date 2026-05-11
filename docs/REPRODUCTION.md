# Reproduction Commands

## ALFWorld Full SLIM Run

```bash
cd /path/to/slim
cp configs/slim_alfworld.env.example .env.alfworld
# Edit MODEL_PATH, EMBEDDING_MODEL_PATH, BASE_REPO_ROOT, and data paths.
set -a; source .env.alfworld; set +a
bash scripts/run_alfworld_slim_full.sh
```

Default important settings:

- `trainer.total_training_steps=60`
- `trainer.test_freq=5`
- `env.slim_memory.lifecycle_update_freq=5`
- `trainer.save_freq=10`
- `trainer.test_after_train=True`
- `data.train_batch_size=16`
- `data.val_batch_size=32`
- `env.rollout.n=8`
- `env.max_steps=50`

## Search-QA Full SLIM Run

```bash
cd /path/to/slim
cp configs/slim_search_qa.env.example .env.searchqa
# Edit MODEL_PATH, EMBEDDING_MODEL_PATH, BASE_REPO_ROOT, and data paths.
set -a; source .env.searchqa; set +a
bash scripts/run_search_qa_slim_full.sh
```

Default important settings:

- `trainer.total_training_steps=180`
- `trainer.test_freq=10`
- `env.slim_memory.lifecycle_update_freq=10`
- `trainer.save_freq=10`
- `trainer.val_before_train=True`
- `SEARCH_TRAIN_BATCH_SIZE=64`
- `SEARCH_VAL_BATCH_SIZE=512`
- `SEARCH_ROLLOUT_N=4`


## Skill Creator Service

Skill expansion requires an OpenAI-compatible chat service. Set
`SKILL_CREATOR_BASE_URL`, `SKILL_CREATOR_API_KEY`, and
`SKILL_CREATOR_MODEL` in the copied env file before running full
paper-faithful lifecycle expansion. To debug without expansion, set
`EXTRA_ARGS="++env.slim_memory.max_new_skills=0"`.

`lifecycle_update_freq` is tied to validation. SLIM audits immediately after
validation, so set it equal to `trainer.test_freq`; if they differ, the trainer
prints a warning and uses the validation schedule.

By default `enforce_expansion_budget=False`, matching the permissive release
behavior. Set `++env.slim_memory.enforce_expansion_budget=True` to enforce a
global per-audit cap of at most `max_new_skills` new task-specific skills.

By default `enable_embedding_dedup=False`. Set
`++env.slim_memory.enable_embedding_dedup=True` to additionally reject generated
skills whose routing text embedding is too similar to an existing
task-specific skill.

## Logging

The release defaults to console logging to avoid requiring a WandB login.
Set `TRAINER_LOGGER='[console,wandb]'` if you want WandB logging.

## Outputs

Checkpoints and lifecycle state are written under `OUTPUT_DIR`. Logs are
written under `LOG_DIR`. Neither directory is intended to be committed.
