# VERL / SGLang Docker Runtime Setup

This document describes the Docker runtime used to run the SLIM ALFWorld and Search-QA experiments.

## Runtime Image

Use the public SGLang-oriented VERL image:

```bash
verlai/verl:sgl055.latest
```

The image contains the main runtime dependencies used by the actor rollout stack, including PyTorch, VERL, SGLang, FSDP utilities, and common transformer dependencies.

## Directory Layout

Inside the container, use anonymous mount points:

```text
/workspace/slim                 # this repository
/models/base_model              # base instruction model
/models/embedding_model         # embedding model for semantic skill retrieval
/datasets/alfworld_full         # ALFWorld train/val/test parquet files
/datasets/search_qa_processed   # Search-QA train/val/test parquet files
/datasets/search_qa_retriever   # Search-QA retriever index and corpus
```

The benchmark data directories must contain:

```text
/datasets/alfworld_full/text/train.parquet
/datasets/alfworld_full/text/val.parquet
/datasets/alfworld_full/text/test.parquet
/datasets/search_qa_processed/train.parquet
/datasets/search_qa_processed/val_1000.parquet
/datasets/search_qa_processed/test.parquet
/datasets/search_qa_retriever/e5_Flat.index
/datasets/search_qa_retriever/wiki-18.jsonl
```

## Container Creation

```bash
docker rm -f verl-runtime >/dev/null 2>&1 || true

docker create \
  --runtime=nvidia \
  --net=host \
  --shm-size=10g \
  --cap-add=SYS_ADMIN \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /path/to/SLIM:/workspace/slim \
  -v /path/to/models:/models \
  -v /path/to/datasets:/datasets \
  --name verl-runtime \
  verlai/verl:sgl055.latest sleep infinity

docker start verl-runtime
```

## Python Path

The launcher sets:

```bash
PYTHONPATH=/workspace/slim:/workspace/slim/agent_system/environments/env_package/alfworld:$PYTHONPATH
```

This release includes the patched trainer stack used by the experiments:

```text
agent_system.environments
agent_system.multi_turn_rollout
agent_system.reward_manager
agent_system.memory.base
verl.trainer
verl.workers
gigpo.core_gigpo
```

The vendored `verl/` includes the SLIM-required final `test_after_train` path:
after the last RL step it evaluates `data.test_files` with
`metrics_prefix="test"` and `allow_skill_update=False`, so final test evaluation
does not update the lifecycle state.

## Validation Checks

```bash
docker exec verl-runtime python3 - <<'PY'
import torch
import verl
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("verl", verl.__file__)
PY
```

Check ALFWorld imports:

```bash
docker exec verl-runtime python3 - <<'PY'
from agent_system.environments import make_envs
print("agent_system ok")
PY
```

Check the SLIM embedding service:

```bash
docker exec verl-runtime bash -lc '
cd /workspace/slim
python3 -m slim_method.embedding_service health \
  --url http://127.0.0.1:6000/v1/embeddings || true
'
```

## Skill Creator Endpoint

SLIM can expand the skill bank during training. The endpoint is configured only through environment variables:

```bash
export SKILL_CREATOR_BASE_URL="https://your-compatible-chat-endpoint/v1"
export SKILL_CREATOR_API_KEY="your_api_key"
export SKILL_CREATOR_MODEL="your_skill_creator_model"
```

The endpoint must expose an OpenAI-compatible `chat.completions.create` API. If expansion is disabled or never triggered, this endpoint is not used.
