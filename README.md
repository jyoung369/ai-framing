# Will AI Save Us or Destroy Us? Comparing AI Media Framing Across Global Powers
This repository contains all experimental code for the AI Media Framing project. In particular, we run experiments to evaluate LLM performance on the task of identifying AI frames and sentiments. It provides few-shot prompts, inference scripts, and evaluation code for three tasks: **Frame Classification (FC)**, **Sentiment Classification (SC)**, and **AI Stance Identification (Identify)**.

The annotated dataset is hosted separately on Hugging Face: [AnonymousAccount2026/anon-dataset](https://huggingface.co/datasets/AnonymousAccount2026/anon-dataset)

---

## Repository Structure

```
submission/
├── test_data/                       # Model-ready test sets
│   ├── {en,ru,zh}_frame_test_ha.jsonl   # FC and SC test spans
│   ├── {en,ru,zh}_identify_test.jsonl   # Identify test sentences
│   ├── {en,ru,zh}_frame_pool_ha.json    # Few-shot example pool (FC & SC)
│   └── {en,ru,zh}_identify_pool.json   # Few-shot example pool (Identify)
│
├── prompts/                         # Prompt templates
│   ├── frame/                       # 3 strategies × 3 languages
│   ├── sentiment/                   # 3 strategies × 3 languages
│   └── identify/                    # 3 strategies × 3 languages
│
├── scripts/                         # Inference and evaluation scripts
│   ├── agreement.py                 # Human agreement calculation
│   ├── run_frame_classification_inference.py
│   ├── run_sentiment_classification_inference.py
│   ├── run_identify_inference.py
│   ├── build_prompt_frame_classification.py
│   ├── build_prompt_sentiment_classification.py
│   ├── build_prompt_identify.py
│   ├── build_paragraph_context.py
│   └── eval_adjudicated.py
│
└── results/                         # Created at runtime by inference scripts
```

---

## Dependencies

- Python 3.10+
- [vLLM](https://github.com/vllm-project/vllm) (tested with 0.4.x)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

```bash
pip install vllm transformers
```

Model weights must be available locally or via the Hugging Face Hub:

| Model key    | Model ID                  |
|--------------|---------------------------|
| `gemma3`     | `google/gemma-3-27b-it`   |
| `qwen3`      | `Qwen/Qwen3-32B`          |
| `gpt-oss-20b`| set `YOUR_GPT20B_MODEL_PATH` in each inference script |

---

## Tasks and Inference

Each inference script writes predictions to `results/` as a `.jsonl` file and a summary `.json`. All paths are resolved relative to the repository root — no changes needed after cloning.

### Prompting Strategies

All three tasks support four strategies controlled by `--strategy`:

| Strategy | Description |
|----------|-------------|
| `1` | Vanilla few-shot (label only) |
| `2` | Few-shot + single confidence score (default) |
| `3` | Few-shot + paragraph context |
| `4` | Full probability distribution over all classes |

---

### Frame Classification (FC)

Classifies each annotated span into one of 25 frame categories.

**Input:** `test_data/{lang}_frame_test_ha.jsonl`  
**Output:** `results/fc_predictions_{run_id}.jsonl`

```bash
python scripts/run_frame_classification_inference.py \
    --lang en \
    --model gemma3 \
    --strategy 2 \
    --run-id gemma3_fc_en_s2
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--lang` | — | `en`, `zh`, or `ru` |
| `--model` | — | `gemma3`, `qwen3`, or `gpt-oss-20b` |
| `--strategy` | `2` | Prompt strategy (1–4) |
| `--n-per-frame` | `5` | Few-shot examples per frame |
| `--seed` | `42` | Random seed for example sampling |
| `--run-id` | timestamp | Output file identifier |

---

### Sentiment Classification (SC)

Classifies each annotated span as positive (`pos`) or negative (`neg`) framing of AI.

**Input:** `test_data/{lang}_frame_test_ha.jsonl` (same spans as FC)  
**Output:** `results/sc_predictions_{run_id}.jsonl`

```bash
python scripts/run_sentiment_classification_inference.py \
    --lang zh \
    --model qwen3 \
    --strategy 2 \
    --run-id qwen3_sc_zh_s2
```

Key arguments mirror FC; `--n-per-frame` is replaced by `--n-per-class` (examples per sentiment class).

---

### Identify

Classifies each sentence as AI-related with explicit positive or negative attitude (`true`) or not (`false`).

**Input:** `test_data/{lang}_identify_test.jsonl`  
**Output:** `results/identify_predictions_{run_id}.jsonl`

```bash
python scripts/run_identify_inference.py \
    --lang ru \
    --model gemma3 \
    --strategy 2 \
    --run-id gemma3_identify_ru_s2
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--lang` | — | `en`, `zh`, or `ru` |
| `--model` | — | `gemma3`, `qwen3`, or `gpt-oss-20b` |
| `--strategy` | `2` | Prompt strategy (1–4) |
| `--n-examples` | `5` | Few-shot examples per class |

---

## Evaluation

`eval_adjudicated.py` evaluates predictions against the adjudicated gold labels from the Hugging Face dataset. Download the dataset into a `dataset/` folder inside this repo before running:

```bash
pip install huggingface_hub
huggingface-cli download AnonymousAccount2026/anon-dataset \
    --repo-type dataset --local-dir dataset
```

```bash
python scripts/eval_adjudicated.py \
    --task fc \
    --lang en \
    --predictions results/fc_predictions_gemma3_fc_en_s2.jsonl
```

```bash
python scripts/eval_adjudicated.py \
    --task sc \
    --lang zh \
    --predictions results/sc_predictions_qwen3_sc_zh_s2.jsonl
```

```bash
python scripts/eval_adjudicated.py \
    --task identify \
    --lang ru \
    --predictions results/identify_predictions_gemma3_identify_ru_s2.jsonl
```

**Arguments:**

| Argument | Choices | Description |
|----------|---------|-------------|
| `--task` | `fc`, `sc`, `identify` | Task to evaluate |
| `--lang` | `en`, `zh`, `ru` | Language |
| `--predictions` | path | `.jsonl` file produced by an inference script |

**Output** (printed and saved to `{predictions}.eval.json`):

```
  Micro F1      : 0.4821  (n=642)
  Best Micro F1 : 0.5103  (threshold >= 72)
```
- **Micro F1**: performance at the default threshold (confidence ≥ 0, i.e., all predictions used).
- **Best Micro F1**: highest Micro F1 found by sweeping confidence thresholds 0–100; predictions below the threshold are suppressed to `none` / `false`.
