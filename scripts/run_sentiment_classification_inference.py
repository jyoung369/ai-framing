"""
Sentiment classification inference for sentiment_confidence_2class_{en,ru,zh}.

Each test sentence is classified as positive or negative framing of AI.
Input:  {lang}_frame_test_ha.jsonl  (reuses same human-annotated data as frame task)
Output: JSON {"Positive": score, "Negative": score, "sentiment": "pos"|"neg"}

Usage
-----
python run_sentiment_classification_inference.py \\
    --lang en --model gemma3 \\
    --run-id "gemma3_sc_en_${SLURM_JOB_ID}"
"""

import argparse
import json
import re
import sys
import time

from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_prompt_sentiment_classification import build_sentiment_classification_prompt
from build_paragraph_context import load_paragraphs, get_context_text

_BASE      = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _BASE / "results"

MODEL_PATHS = {
    "gemma3":      "google/gemma-3-27b-it",
    "qwen3":       "Qwen/Qwen3-32B",
    "gpt-oss-20b": "YOUR_GPT20B_MODEL_PATH",
}

TEST_DATA_PATHS = {
    "zh": _BASE / "test_data/zh_frame_test_ha.jsonl",
    "en": _BASE / "test_data/en_frame_test_ha.jsonl",
    "ru": _BASE / "test_data/ru_frame_test_ha.jsonl",
}

STOP_TOKENS = {
    "gemma3":      ["<end_of_turn>"],
    "qwen3":       ["<|im_end|>"],
    "gpt-oss-20b": ["<|im_end|>"],
}

VANILLA_PROMPTS = {
    "zh": _BASE / "prompts/sentiment/sentiment_vanilla_2class_zh.txt",
    "en": _BASE / "prompts/sentiment/sentiment_vanilla_2class_en.txt",
    "ru": _BASE / "prompts/sentiment/sentiment_vanilla_2class_ru.txt",
}
CONFIDENCE_PROMPTS = {
    "zh": _BASE / "prompts/sentiment/sentiment_confidence_single_zh.txt",
    "en": _BASE / "prompts/sentiment/sentiment_confidence_single_en.txt",
    "ru": _BASE / "prompts/sentiment/sentiment_confidence_single_ru.txt",
}
PROB_DIST_PROMPTS = {
    "zh": _BASE / "prompts/sentiment/sentiment_confidence_2class_zh.txt",
    "en": _BASE / "prompts/sentiment/sentiment_confidence_2class_en.txt",
    "ru": _BASE / "prompts/sentiment/sentiment_confidence_2class_ru.txt",
}


def strip_thinking(text: str) -> str:
    # Qwen3 / Gemma3 thinking tags
    if "<think>" in text and "</think>" in text:
        return text[text.find("</think>") + len("</think>"):].strip()
    # GPT20B harmony format: analysis channel prefix before "assistantfinal"
    if "assistantfinal" in text:
        return text[text.rfind("assistantfinal") + len("assistantfinal"):].strip()
    return text


def _strip_markdown_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_sentiment_output(text: str):
    """Parse model JSON. Returns dict with at least a 'sentiment' key, or None."""
    text = strip_thinking(text)
    text = _strip_markdown_block(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "sentiment" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]{5,}\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "sentiment" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    # Last resort: extract sentiment field
    s = re.search(r'"sentiment"\s*:\s*"(pos|neg)"', text)
    if s:
        return {"sentiment": s.group(1)}
    return None


def extract_sentiment(parsed) -> str | None:
    if not parsed:
        return None
    s = parsed.get("sentiment")
    if s in {"pos", "neg"}:
        return s
    # Fall back to higher numeric score
    pos_score = parsed.get("Positive", -1)
    neg_score = parsed.get("Negative", -1)
    if isinstance(pos_score, (int, float)) and isinstance(neg_score, (int, float)):
        return "pos" if pos_score >= neg_score else "neg"
    return None


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",        required=True, choices=["zh", "en", "ru"])
    parser.add_argument("--model",       required=True, choices=["gemma3", "qwen3", "gpt-oss-20b"])
    parser.add_argument("--thinking",    action="store_true")
    parser.add_argument("--strategy",    type=int, default=2, choices=[1, 2, 3, 4],
                        help="1=vanilla, 2=vanilla+confidence, 3=vanilla+context, 4=prob-distribution")
    parser.add_argument("--run-id",      default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--pool-file",   type=Path, default=None)
    parser.add_argument("--test-data",   type=Path, default=None)
    parser.add_argument("--n-per-class", type=int,  default=5,
                        help="Example sentences per sentiment class in the prompt")
    parser.add_argument("--seed",        type=int,  default=42)
    args = parser.parse_args()

    # S1: vanilla (label only)
    # S2: vanilla + single confidence score
    # S3: vanilla + context (label only, same output as S1)
    # S4: full probability distribution over pos/neg
    use_context = args.strategy == 3
    use_dist    = args.strategy == 4

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path     = OUTPUT_DIR / f"sc_predictions_{args.run_id}.jsonl"
    summary_path = OUTPUT_DIR / f"sc_summary_{args.run_id}.json"

    test_data_path = args.test_data or TEST_DATA_PATHS[args.lang]
    if args.strategy == 1 or args.strategy == 3:
        prompt_file = args.prompt_file or VANILLA_PROMPTS[args.lang]
    elif args.strategy == 2:
        prompt_file = args.prompt_file or CONFIDENCE_PROMPTS[args.lang]
    else:  # strategy 4
        prompt_file = args.prompt_file or PROB_DIST_PROMPTS[args.lang]

    print(f"Run ID      : {args.run_id}")
    print(f"Lang        : {args.lang}")
    print(f"Model       : {args.model} ({MODEL_PATHS[args.model]})")
    print(f"Strategy    : {args.strategy}")
    print(f"N per class : {args.n_per_class}")
    print(f"Seed        : {args.seed}")
    print(f"Test data   : {test_data_path}")
    print(f"Output      : {out_path}")

    # ── Load test data ─────────────────────────────────────────────────────────
    print("\nLoading test data...")
    with open(test_data_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"  {len(records)} sentences")

    ctx_dir    = args.test_data.parent if args.test_data else None
    paragraphs = load_paragraphs(args.lang, data_dir=ctx_dir) if use_context else {}

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    model_path = MODEL_PATHS[args.model]
    tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── Build prompts ──────────────────────────────────────────────────────────
    print("Building prompts...")
    nothink = args.model in ("qwen3", "gpt-oss-20b")
    prompts = []

    for rec in records:
        span = rec["span_text"]
        if use_context:
            para_text = paragraphs.get(rec.get("paragraph_id", ""), "")
            if para_text:
                span = get_context_text(span, para_text, args.lang)
        if nothink:
            span = span + "\n/no_think"
        prompt = build_sentiment_classification_prompt(
            span,
            lang=args.lang,
            tokenizer=tokenizer,
            template_path=prompt_file,
            pool_path=args.pool_file,
            n_per_class=args.n_per_class,
            seed=args.seed,
        )
        prompts.append(prompt)

    print(f"  {len(prompts)} prompts built")
    if prompts:
        print(f"  Sample prompt length: {len(prompts[0])} chars")

    # ── vLLM inference ─────────────────────────────────────────────────────────
    print("\nLoading vLLM model...")
    from vllm import LLM, SamplingParams

    stop = STOP_TOKENS[args.model]

    max_tokens_short = 64    # {"sentiment": "pos"} or {"sentiment": "pos", "confidence": 85}
    max_tokens_dist  = 1024  # {"Positive": X, "Negative": Y, "sentiment": "pos"} + GPT20B reasoning prefix
    if args.model == "qwen3":
        max_tokens = max_tokens_dist if use_dist else max_tokens_short
        llm = LLM(
            model=model_path, tensor_parallel_size=1, dtype="bfloat16",
            max_model_len=4096, gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, stop=stop,
        )
    elif args.model == "gpt-oss-20b":
        # GPT-OSS-20B always outputs reasoning text before JSON; use large max_tokens for all strategies
        max_tokens = max_tokens_dist
        llm = LLM(
            model=model_path, tensor_parallel_size=1, dtype="bfloat16",
            max_model_len=8192, gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, stop=stop,
        )
    else:  # gemma3
        max_tokens = max_tokens_dist if use_dist else max_tokens_short
        llm = LLM(
            model=model_path, tensor_parallel_size=1, dtype="bfloat16",
            max_model_len=4096, gpu_memory_utilization=0.92,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, stop=stop,
        )

    print(f"Running inference on {len(prompts)} sentences...")
    t0      = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(prompts)/elapsed:.1f} sents/s)")

    # ── Parse & save ───────────────────────────────────────────────────────────
    print(f"\nSaving predictions → {out_path}")
    parse_errors = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for rec, output in zip(records, outputs):
            raw_text = output.outputs[0].text.strip()
            parsed   = parse_sentiment_output(raw_text)
            pred     = extract_sentiment(parsed)

            if parsed is None:
                parse_errors += 1

            result = {
                "id":                   rec["id"],
                "source":               rec["source"],
                "article_id":           rec["article_id"],
                "span_text":            rec["span_text"],
                "expected_output":      rec["expected_output"],
                "raw_output":           raw_text,
                "parsed_output":        parsed,
                "predicted_sentiment":  pred,
                "_meta": {
                    "lang":        args.lang,
                    "model":       args.model,
                    "strategy":    args.strategy,
                    "run_id":      args.run_id,
                    "n_per_class": args.n_per_class,
                    "prompt_file": str(prompt_file) if prompt_file else None,
                },
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"  Saved {len(records)} records  ({parse_errors} parse errors)")

    with open(summary_path, "w") as f:
        json.dump({
            "run_id":      args.run_id,
            "lang":        args.lang,
            "model":       args.model,
            "strategy":    args.strategy,
            "thinking":    args.thinking,
            "n_per_class": args.n_per_class,
            "n_sentences": len(records),
            "elapsed_s":   round(elapsed, 1),
            "parse_errors": parse_errors,
        }, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print("Done.")
