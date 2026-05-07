"""
Sentence-level AI-sentiment identification inference.

For each sentence in the test JSONL, asks the model whether it is AI-related
with explicit positive or negative attitude (True) or not (False).

Usage
-----
python run_identify_inference.py \\
    --lang zh --model gemma3 \\
    --run-id "gemma3_identify_zh_n5_${SLURM_JOB_ID}" \\
    --n-examples 5 \\
    --prompt-file .../identify_zh.txt \\
    --pool-file   .../zh_identify_pool.json \\
    --test-data   .../zh_identify_test.jsonl
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_prompt_identify import build_identify_prompt
from build_paragraph_context import load_paragraphs, get_context_text

_BASE      = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _BASE / "results"

MODEL_PATHS = {
    "gemma3":      "google/gemma-3-27b-it",
    "qwen3":       "Qwen/Qwen3-32B",
    "gpt-oss-20b": "YOUR_GPT20B_MODEL_PATH",
}
STOP_TOKENS = {
    "gemma3":      ["<end_of_turn>"],
    "qwen3":       ["<|im_end|>"],
    "gpt-oss-20b": ["<|im_end|>"],
}
DEFAULT_PROMPTS = {
    "zh": _BASE / "prompts/identify/identify_zh.txt",
    "en": _BASE / "prompts/identify/identify_en.txt",
    "ru": _BASE / "prompts/identify/identify_ru.txt",
}
VANILLA_PROMPTS = {
    "zh": _BASE / "prompts/identify/identify_vanilla_zh.txt",
    "en": _BASE / "prompts/identify/identify_vanilla_en.txt",
    "ru": _BASE / "prompts/identify/identify_vanilla_ru.txt",
}
PROB_PROMPTS = {
    "zh": _BASE / "prompts/identify/identify_prob_zh.txt",
    "en": _BASE / "prompts/identify/identify_prob_en.txt",
    "ru": _BASE / "prompts/identify/identify_prob_ru.txt",
}
DEFAULT_POOLS = {
    "zh": _BASE / "test_data/zh_identify_pool.json",
    "en": _BASE / "test_data/en_identify_pool.json",
    "ru": _BASE / "test_data/ru_identify_pool.json",
}
DEFAULT_TESTS = {
    "zh": _BASE / "test_data/zh_identify_test.jsonl",
    "en": _BASE / "test_data/en_identify_test.jsonl",
    "ru": _BASE / "test_data/ru_identify_test.jsonl",
}


def parse_output(text: str) -> tuple[bool | None, int | None]:
    """Extract (is_target, confidence) from strategy 1/2/3 output.
    Tries all {...} matches from last to first so GPT20B's reasoning-prefix
    template JSON (e.g. {"is_target":false,"confidence":...}) is skipped in
    favour of the actual final JSON at the end of the output.
    """
    text = text.strip()
    for match in reversed(list(re.finditer(r'\{[^}]+\}', text))):
        try:
            obj = json.loads(match.group())
            is_target = obj.get("is_target")
            confidence = obj.get("confidence")
            if isinstance(is_target, bool):
                return is_target, int(confidence) if isinstance(confidence, (int, float)) else 0
            if isinstance(is_target, str):
                is_target = is_target.lower() == "true"
                return is_target, int(confidence) if isinstance(confidence, (int, float)) else 0
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None, None


def parse_output_prob(text: str) -> tuple[bool | None, int | None]:
    """
    Extract (is_target, true_score) from strategy 4 output.
    Expected: {"True": 65, "False": 35, "is_target": true}
    true_score is used as the confidence for threshold sweeping.
    Tries matches last-to-first for the same GPT20B reasoning-prefix reason.
    """
    text = text.strip()
    for match in reversed(list(re.finditer(r'\{[^}]+\}', text))):
        try:
            obj = json.loads(match.group())
            true_score  = obj.get("True")
            false_score = obj.get("False")
            is_target   = obj.get("is_target")
            if true_score is None or false_score is None:
                continue
            true_score  = int(true_score)
            false_score = int(false_score)
            if is_target is None:
                is_target = true_score > false_score
            elif isinstance(is_target, str):
                is_target = is_target.lower() == "true"
            return bool(is_target), true_score
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",        required=True, choices=["zh", "en", "ru"])
    parser.add_argument("--model",       required=True, choices=["gemma3", "qwen3", "gpt-oss-20b"])
    parser.add_argument("--thinking",    action="store_true")
    parser.add_argument("--strategy",    type=int, default=2, choices=[1, 2, 3, 4],
                        help="1=vanilla, 2=confidence, 3=paragraph-context, 4=prob-distribution")
    parser.add_argument("--run-id",      default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--n-examples",  type=int, default=5)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--pool-file",   type=Path, default=None)
    parser.add_argument("--test-data",   type=Path, default=None)
    args = parser.parse_args()

    use_vanilla   = args.strategy in (1, 3)
    use_context   = args.strategy == 3
    use_confidence = args.strategy == 2
    use_prob_dist  = args.strategy == 4

    if args.strategy == 4:
        prompt_file = args.prompt_file or PROB_PROMPTS[args.lang]
    elif use_vanilla:
        prompt_file = args.prompt_file or VANILLA_PROMPTS[args.lang]
    else:
        prompt_file = args.prompt_file or DEFAULT_PROMPTS[args.lang]
    pool_file   = args.pool_file   or DEFAULT_POOLS[args.lang]
    test_file   = args.test_data   or DEFAULT_TESTS[args.lang]
    model_path  = MODEL_PATHS[args.model]
    run_id      = args.run_id

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path     = OUTPUT_DIR / f"identify_predictions_{run_id}.jsonl"
    summary_path = OUTPUT_DIR / f"identify_summary_{run_id}.json"

    print(f"Run ID     : {run_id}")
    print(f"Lang       : {args.lang}")
    print(f"Model      : {args.model} ({model_path})")
    print(f"Strategy   : {args.strategy}")
    print(f"N examples : {args.n_examples} per class ({2*args.n_examples} total)")
    print(f"Test data  : {test_file}")
    print(f"Output     : {out_path}")

    # ── Load test data ─────────────────────────────────────────────────────────
    print("\nLoading test data...")
    with open(test_file, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    print(f"  {len(records)} sentences")

    ctx_dir    = args.test_data.parent if args.test_data else None
    paragraphs = load_paragraphs(args.lang, data_dir=ctx_dir) if use_context else {}

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── Build prompts ──────────────────────────────────────────────────────────
    print("Building prompts...")
    nothink = args.model in ("qwen3", "gpt-oss-20b")
    prompts = []
    for rec in records:
        sentence = rec["sentence_text"]
        if use_context:
            para_text = paragraphs.get(rec.get("paragraph_id", ""), "")
            if para_text:
                sentence = get_context_text(sentence, para_text, args.lang)
        prompt = build_identify_prompt(
            sentence,
            n_examples=args.n_examples,
            tokenizer=tokenizer,
            template_path=prompt_file,
            pool_path=pool_file,
            nothink=nothink,
            use_confidence=use_confidence,
            use_prob_dist=use_prob_dist,
        )
        prompts.append(prompt)
    print(f"  {len(prompts)} prompts built")
    print(f"  Sample prompt length: {len(prompts[0])} chars")

    # ── vLLM inference ─────────────────────────────────────────────────────────
    print("\nLoading vLLM model...")
    from vllm import LLM, SamplingParams

    stop = STOP_TOKENS[args.model]

    max_tokens_s1   = 32   # vanilla: just {"is_target": true}
    max_tokens_s2s4 = 64   # confidence or prob-dist: small fixed JSON
    if args.model == "qwen3":
        max_tokens = max_tokens_s2s4 if (use_confidence or use_prob_dist) else max_tokens_s1
        llm = LLM(
            model=model_path, tensor_parallel_size=1, dtype="bfloat16",
            max_model_len=8192, gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, stop=stop,
        )
    elif args.model == "gpt-oss-20b":
        # GPT-OSS-20B always outputs reasoning text before JSON; use large max_tokens for all strategies
        max_tokens = 1024
        llm = LLM(
            model=model_path, tensor_parallel_size=1, dtype="bfloat16",
            max_model_len=8192, gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, stop=stop,
        )
    else:  # gemma3
        max_tokens = max_tokens_s2s4 if (use_confidence or use_prob_dist) else max_tokens_s1
        llm = LLM(
            model=model_path, tensor_parallel_size=1, dtype="bfloat16",
            max_model_len=8192, gpu_memory_utilization=0.92,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, stop=stop,
        )

    print(f"Running inference on {len(prompts)} prompts...")
    t0      = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(prompts)/elapsed:.1f} sentences/s)")

    # ── Parse & save ───────────────────────────────────────────────────────────
    print(f"\nSaving predictions to {out_path} ...")
    _parser = parse_output_prob if use_prob_dist else parse_output
    parse_errors = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec, out in zip(records, outputs):
            raw_text   = out.outputs[0].text.strip()
            predicted, confidence = _parser(raw_text)
            if predicted is None:
                parse_errors += 1
                predicted  = False
                confidence = 0
            result = {
                "sentence_text":   rec["sentence_text"],
                "label":           rec["label"],
                "paragraph_id":    rec.get("paragraph_id", ""),
                "predicted_label": predicted,
                "confidence":      confidence,
                "raw_output":      raw_text,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"  Saved {len(records)} records  ({parse_errors} parse errors)")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id":       run_id,
            "lang":         args.lang,
            "model":        args.model,
            "strategy":     args.strategy,
            "n_examples":   args.n_examples,
            "n_sentences":  len(records),
            "elapsed_s":    round(elapsed, 1),
            "parse_errors": parse_errors,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
