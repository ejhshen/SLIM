# Data Setup

## Container data mount layout

The launch examples assume benchmark data and skill banks are mounted under `/datasets`:

```text
/datasets/alfworld_full/text/train.parquet
/datasets/alfworld_full/text/val.parquet
/datasets/alfworld_full/text/test.parquet
/datasets/search_qa_processed/train.parquet
/datasets/search_qa_processed/val_1000.parquet
/datasets/search_qa_processed/test.parquet
/datasets/search_qa_retriever/
```

## ALFWorld

1. Install runtime dependencies:

```bash
bash tools/prepare_alfworld_runtime.sh
```

2. Generate text-mode parquet splits:

```bash
export ALFWORLD_DATA_DIR=/datasets/alfworld_full
bash tools/prepare_alfworld_text_data.sh
```

The script expects the external base repo to provide
`examples.data_preprocess.prepare`.

## Search-QA

1. Generate Search-R1/Search-QA parquet data:

```bash
export SEARCH_QA_DATA_DIR=/datasets/search_qa_processed
bash tools/prepare_search_qa_data.sh
```

This produces train/test and a validation sample `val_1000.parquet`
sampled from the test split without removing examples from test.

2. Prepare retriever index and corpus:

```bash
export SEARCH_QA_RETRIEVER_DIR=/datasets/search_qa_retriever
bash tools/prepare_search_qa_retriever.sh
```

3. Start or check retriever pool:

```bash
bash tools/start_search_qa_retriever_pool.sh
bash tools/check_search_qa_retriever_pool.sh
```
